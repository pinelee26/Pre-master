"""
Hopfield-Compressed Attention: KV Cache Compression via TokenRank & Modern Hopfield Networks
(v3 — Observe Window + Soft β)

Key improvements over v2:
  1. Observe Window: the last N tokens are NEVER compressed — only older tokens
     are chunked and merged via Hopfield prototypes, preventing repetition loops
  2. Softer β=2.0 by default — reduces over-attraction to dominant patterns
  3. 1-step Hopfield update by default — faster with minimal quality loss
  4. One-shot compression with window-aware split

Reference equations:
  - TokenRank: π = πP   (left eigenvector / steady-state of DTMC)
  - Hopfield prototype: ξ_{t+1} = X^T softmax(β X ξ_t)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    apply_rotary_pos_emb,
    repeat_kv,
)


# ---------------------------------------------------------------------------
# 1. TokenRank — Steady-state distribution of DTMC
# ---------------------------------------------------------------------------

def compute_token_rank(
    attn_weights: torch.Tensor,
    num_iterations: int = 20,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Compute TokenRank via power iteration on the transpose of the
    attention transition matrix.

    Args:
        attn_weights: (B, H, S, S) — row-stochastic (post-softmax).
        num_iterations: power-iteration steps.
        epsilon: numerical stability constant.

    Returns:
        token_rank: (B, H, S) — importance score per token per head.
    """
    P_T = attn_weights.transpose(-2, -1)
    S = attn_weights.shape[-1]

    pi = attn_weights.new_ones(*attn_weights.shape[:-1]) / S

    for _ in range(num_iterations):
        pi = torch.einsum("bhij,bhj->bhi", P_T, pi)
        pi = pi / (pi.sum(dim=-1, keepdim=True) + epsilon)

    return pi


# ---------------------------------------------------------------------------
# 2. Chunk-level compression decision
# ---------------------------------------------------------------------------

def identify_chunks(
    token_rank: torch.Tensor,
    chunk_size: int = 8,
    top_k_ratio: float = 0.65,
) -> Tuple[torch.Tensor, int]:
    """
    Partition the sequence into fixed-size chunks and mark which to compress
    based on aggregate TokenRank mass.

    Returns:
        compress_mask: (B, H, num_chunks) bool — True = compress.
        num_chunks: int
    """
    B, H, S = token_rank.shape
    pad = (chunk_size - S % chunk_size) % chunk_size
    if pad > 0:
        token_rank = F.pad(token_rank, (0, pad), value=0.0)

    num_chunks = token_rank.shape[-1] // chunk_size
    chunk_scores = token_rank.view(B, H, num_chunks, chunk_size).sum(dim=-1)

    k = max(1, int(num_chunks * top_k_ratio))
    topk_vals, _ = chunk_scores.topk(k, dim=-1)
    threshold = topk_vals[..., -1:]

    compress_mask = chunk_scores < threshold

    return compress_mask, num_chunks


# ---------------------------------------------------------------------------
# 3. Vectorized Modern Hopfield Prototype
# ---------------------------------------------------------------------------

def hopfield_prototype_batched(
    X: torch.Tensor,
    beta: float = 2.0,
    num_steps: int = 1,
) -> torch.Tensor:
    """
    Batched multi-step Modern Hopfield update.

    ξ_{t+1} = X^T softmax(β X ξ_t)

    Args:
        X: (*, N, D) — stored patterns (last two dims are patterns x features).
        beta: inverse temperature.
        num_steps: number of iterative updates toward the fixed-point attractor.

    Returns:
        prototype: (*, D) — fused prototype per batch element.
    """
    xi = X.mean(dim=-2)

    for _ in range(num_steps):
        logits = beta * torch.einsum("...nd,...d->...n", X, xi)
        weights = F.softmax(logits, dim=-1)
        xi = torch.einsum("...nd,...n->...d", X, weights)

    return xi


# ---------------------------------------------------------------------------
# 4. Vectorized KV compression (window-aware)
# ---------------------------------------------------------------------------

def compress_kv_with_hopfield(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    compress_mask: torch.Tensor,
    chunk_size: int,
    beta: float,
    num_steps: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hopfield prototype merging for selected chunks.

    Compressed chunks  -> 1 prototype token (Hopfield update)
    Preserved  chunks  -> all chunk_size tokens kept verbatim

    Args:
        key_states:    (B, H_kv, S, D) — only the compressible region
        value_states:  (B, H_kv, S, D)
        compress_mask: (B, H_kv, num_chunks) bool
        chunk_size:    tokens per chunk
        beta:          Hopfield inverse temperature
        num_steps:     Hopfield iteration count

    Returns:
        compressed_keys:   (B, H_kv, S', D)
        compressed_values: (B, H_kv, S', D)
    """
    B, H, S, D = key_states.shape
    device = key_states.device
    dtype = key_states.dtype

    # Pad to chunk-aligned length
    pad = (chunk_size - S % chunk_size) % chunk_size
    if pad > 0:
        key_states = F.pad(key_states, (0, 0, 0, pad))
        value_states = F.pad(value_states, (0, 0, 0, pad))

    S_padded = key_states.shape[2]
    num_chunks = S_padded // chunk_size

    # Reshape: (B, H, num_chunks, chunk_size, D)
    k_chunks = key_states.view(B, H, num_chunks, chunk_size, D)
    v_chunks = value_states.view(B, H, num_chunks, chunk_size, D)

    # Compute prototypes for ALL chunks (vectorized): (B, H, num_chunks, D)
    k_protos = hopfield_prototype_batched(k_chunks, beta=beta, num_steps=num_steps)
    v_protos = hopfield_prototype_batched(v_chunks, beta=beta, num_steps=num_steps)

    # Tokens per chunk: compressed -> 1, preserved -> chunk_size
    tokens_per_chunk = torch.where(
        compress_mask,
        torch.ones_like(compress_mask, dtype=torch.long),
        torch.full_like(compress_mask, chunk_size, dtype=torch.long),
    )

    total_tokens = tokens_per_chunk.sum(dim=-1)
    S_out = total_tokens.max().item()

    out_k = torch.zeros(B, H, S_out, D, device=device, dtype=dtype)
    out_v = torch.zeros(B, H, S_out, D, device=device, dtype=dtype)

    write_pos = torch.zeros(B, H, device=device, dtype=torch.long)

    for c in range(num_chunks):
        mask_c = compress_mask[:, :, c]

        # Compressed path: 1 prototype token
        b_comp, h_comp = torch.where(mask_c)
        if b_comp.numel() > 0:
            pos = write_pos[b_comp, h_comp]
            out_k[b_comp, h_comp, pos] = k_protos[b_comp, h_comp, c]
            out_v[b_comp, h_comp, pos] = v_protos[b_comp, h_comp, c]
            write_pos[b_comp, h_comp] += 1

        # Preserved path: chunk_size tokens
        b_keep, h_keep = torch.where(~mask_c)
        if b_keep.numel() > 0:
            for t in range(chunk_size):
                pos = write_pos[b_keep, h_keep] + t
                out_k[b_keep, h_keep, pos] = k_chunks[b_keep, h_keep, c, t]
                out_v[b_keep, h_keep, pos] = v_chunks[b_keep, h_keep, c, t]
            write_pos[b_keep, h_keep] += chunk_size

    return out_k, out_v


# ---------------------------------------------------------------------------
# 5. Main Attention Module
# ---------------------------------------------------------------------------

class HopfieldCompressedAttention(LlamaAttention):
    """
    Drop-in replacement for LlamaAttention with Hopfield-compressed KV cache.

    v3 — Observe Window + Soft Beta:
      - Hybrid observe window: last `window_size` tokens are NEVER compressed,
        only older tokens are chunked and merged via Hopfield prototypes.
        This prevents repetition loops by preserving recent context verbatim.
      - Softer β=2.0 reduces over-attraction to dominant attractor patterns
      - 1-step Hopfield update for speed with minimal quality trade-off
      - One-shot compression: compress once when cache exceeds threshold,
        then append without re-compressing

    Hyperparameters (set via config attributes):
        hopfield_beta          Inverse temperature for Hopfield update     (default: 2.0)
        hopfield_steps         Hopfield iteration count per prototype      (default: 1)
        chunk_size             Tokens per compression chunk                (default: 8)
        top_k_ratio            Fraction of chunks to keep uncompressed     (default: 0.65)
        rank_iterations        Power-iteration steps for TokenRank         (default: 20)
        compress_threshold     Minimum seq length to trigger compression   (default: 32)
        window_size            Recent tokens to keep uncompressed          (default: 32)
    """

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        self.hopfield_beta = getattr(config, "hopfield_beta", 2.0)
        self.hopfield_steps = getattr(config, "hopfield_steps", 1)
        self.chunk_size = getattr(config, "chunk_size", 8)
        self.top_k_ratio = getattr(config, "top_k_ratio", 0.65)
        self.rank_iterations = getattr(config, "rank_iterations", 20)
        self.compress_threshold = getattr(config, "compress_threshold", 32)
        self.window_size = getattr(config, "window_size", 32)

        # One-shot flag: compress once, then append
        self._compressed = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, S_q, _ = hidden_states.shape
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # ── Projections ──────────────────────────────────────────────
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # ── Rotary position embeddings ────────────────────────────────
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin,
            )

        # ── Prepend cached KV ─────────────────────────────────────────
        has_cache = (
            past_key_values is not None
            and self.layer_idx < len(past_key_values.layers)
            and past_key_values.layers[self.layer_idx].is_initialized
        )
        if has_cache:
            cached_layer = past_key_values.layers[self.layer_idx]
            key_states = torch.cat([cached_layer.keys, key_states], dim=2)
            value_states = torch.cat([cached_layer.values, value_states], dim=2)

        total_seq_len = key_states.shape[2]

        # ── GQA expansion for attention computation ───────────────────
        key_states_expanded = repeat_kv(key_states, self.num_key_value_groups)
        value_states_expanded = repeat_kv(value_states, self.num_key_value_groups)

        # ── Scaled dot-product attention (explicit) ───────────────────
        attn_weights = torch.matmul(
            query_states,
            key_states_expanded.transpose(-2, -1),
        ) * self.scaling

        # Causal mask
        if attention_mask is not None:
            causal_len = attn_weights.shape[-1]
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :causal_len]
                attn_weights = attn_weights + (1.0 - mask) * torch.finfo(attn_weights.dtype).min
            elif attention_mask.dim() == 4:
                mask = attention_mask[..., :attn_weights.shape[-2], :causal_len]
                attn_weights = attn_weights + mask
        else:
            S_kv = key_states_expanded.shape[2]
            causal = torch.triu(
                torch.full(
                    (S_q, S_kv), float("-inf"),
                    device=attn_weights.device, dtype=attn_weights.dtype,
                ),
                diagonal=S_kv - S_q + 1,
            )
            attn_weights = attn_weights + causal

        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype,
        )

        if self.training:
            attn_probs = F.dropout(attn_probs, p=self.attention_dropout)

        attn_output = torch.matmul(attn_probs, value_states_expanded)

        # ── KV compression (one-shot, window-aware) ───────────────────
        #
        # Layout after compression:
        #   [ compressed_old_tokens | recent_window_tokens ]
        #         ^                         ^
        #   Hopfield prototypes       verbatim (never touched)
        #
        should_compress = (
            not self._compressed
            and total_seq_len >= self.compress_threshold
        )

        if should_compress:
            # Split: compressible region vs observe window
            window = min(self.window_size, total_seq_len)
            compress_len = total_seq_len - window

            if compress_len >= self.chunk_size:
                # Slice into old (compressible) and recent (protected)
                k_old = key_states[:, :, :compress_len, :]
                v_old = value_states[:, :, :compress_len, :]
                k_recent = key_states[:, :, compress_len:, :]
                v_recent = value_states[:, :, compress_len:, :]

                with torch.no_grad():
                    # TokenRank on the compressible region only
                    kk = torch.matmul(
                        k_old, k_old.transpose(-2, -1),
                    ) * self.scaling
                    causal_kk = torch.triu(
                        torch.full(
                            (compress_len, compress_len), float("-inf"),
                            device=kk.device, dtype=kk.dtype,
                        ),
                        diagonal=1,
                    )
                    P = F.softmax(kk + causal_kk, dim=-1)

                    token_rank = compute_token_rank(
                        P, num_iterations=self.rank_iterations,
                    )
                    compress_mask, _ = identify_chunks(
                        token_rank,
                        chunk_size=self.chunk_size,
                        top_k_ratio=self.top_k_ratio,
                    )

                # Compress old region
                comp_k, comp_v = compress_kv_with_hopfield(
                    k_old, v_old,
                    compress_mask,
                    chunk_size=self.chunk_size,
                    beta=self.hopfield_beta,
                    num_steps=self.hopfield_steps,
                )

                # Concat: compressed_old + recent_window
                compressed_k = torch.cat([comp_k, k_recent], dim=2)
                compressed_v = torch.cat([comp_v, v_recent], dim=2)
            else:
                # Not enough old tokens to form even one chunk — keep as-is
                compressed_k = key_states
                compressed_v = value_states

            self._compressed = True
        else:
            compressed_k = key_states
            compressed_v = value_states

        # ── Update KV cache ───────────────────────────────────────────
        if past_key_values is not None:
            while len(past_key_values.layers) <= self.layer_idx:
                past_key_values.update(
                    torch.zeros(
                        B, key_states.shape[1], 0, self.head_dim,
                        device=key_states.device, dtype=key_states.dtype,
                    ),
                    torch.zeros(
                        B, value_states.shape[1], 0, self.head_dim,
                        device=value_states.device, dtype=value_states.dtype,
                    ),
                    layer_idx=len(past_key_values.layers),
                )
            past_key_values.layers[self.layer_idx].keys = compressed_k
            past_key_values.layers[self.layer_idx].values = compressed_v

        # ── Output projection ─────────────────────────────────────────
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_probs
