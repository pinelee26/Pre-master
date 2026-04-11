"""
Hopfield-Compressed Attention: KV Cache Compression via TokenRank & Modern Hopfield Networks
(v4 — Streaming Hopfield Cache with Periodic Re-compression)

Architecture:
  KV cache is logically split into two regions tracked by _compressed_len:

    [ compressed_kv (prototypes)  |  recent_kv (verbatim)  ]
    ├── 0 .. _compressed_len ────┤├── sliding window ──────┤

  - During decode, new tokens append to recent_kv.
  - When recent_kv exceeds (window_size + compression_interval), the oldest
    compression_interval tokens are popped, Hopfield-compressed into prototypes,
    and appended to compressed_kv.
  - recent_kv shrinks back to ~window_size.
  - This keeps total cache size bounded regardless of generation length.

Key improvements over v3:
  1. Periodic re-compression during decode (not just one-shot at prefill)
  2. Dual-region cache with tracked boundary (_compressed_len)
  3. Bounded memory: cache grows at a fraction of the generation rate
  4. Overflow chunks are ALL compressed (no TokenRank needed for periodic passes)
  5. Initial prefill still uses TokenRank for intelligent first compression

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
    P_T = attn_weights.transpose(-2, -1)
    S = attn_weights.shape[-1]
    pi = attn_weights.new_ones(*attn_weights.shape[:-1]) / S
    for _ in range(num_iterations):
        pi = torch.einsum("bhij,bhj->bhi", P_T, pi)
        pi = pi / (pi.sum(dim=-1, keepdim=True) + epsilon)
    return pi


# ---------------------------------------------------------------------------
# 2. Chunk-level compression decision (used for initial prefill only)
# ---------------------------------------------------------------------------

def identify_chunks(
    token_rank: torch.Tensor,
    chunk_size: int = 8,
    top_k_ratio: float = 0.65,
) -> Tuple[torch.Tensor, int]:
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
    ξ_{t+1} = X^T softmax(β X ξ_t)

    Args:
        X: (*, N, D) — stored patterns.
        beta: inverse temperature.
        num_steps: iteration count.
    Returns:
        prototype: (*, D)
    """
    xi = X.mean(dim=-2)
    for _ in range(num_steps):
        logits = beta * torch.einsum("...nd,...d->...n", X, xi)
        weights = F.softmax(logits, dim=-1)
        xi = torch.einsum("...nd,...n->...d", X, weights)
    return xi


# ---------------------------------------------------------------------------
# 4. KV compression functions
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
    Selective Hopfield compression (used for initial prefill with TokenRank).
    compress_mask determines which chunks to compress vs preserve verbatim.
    """
    B, H, S, D = key_states.shape
    device = key_states.device
    dtype = key_states.dtype

    pad = (chunk_size - S % chunk_size) % chunk_size
    if pad > 0:
        key_states = F.pad(key_states, (0, 0, 0, pad))
        value_states = F.pad(value_states, (0, 0, 0, pad))

    S_padded = key_states.shape[2]
    num_chunks = S_padded // chunk_size

    k_chunks = key_states.view(B, H, num_chunks, chunk_size, D)
    v_chunks = value_states.view(B, H, num_chunks, chunk_size, D)

    k_protos = hopfield_prototype_batched(k_chunks, beta=beta, num_steps=num_steps)
    v_protos = hopfield_prototype_batched(v_chunks, beta=beta, num_steps=num_steps)

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
        b_comp, h_comp = torch.where(mask_c)
        if b_comp.numel() > 0:
            pos = write_pos[b_comp, h_comp]
            out_k[b_comp, h_comp, pos] = k_protos[b_comp, h_comp, c]
            out_v[b_comp, h_comp, pos] = v_protos[b_comp, h_comp, c]
            write_pos[b_comp, h_comp] += 1

        b_keep, h_keep = torch.where(~mask_c)
        if b_keep.numel() > 0:
            for t in range(chunk_size):
                pos = write_pos[b_keep, h_keep] + t
                out_k[b_keep, h_keep, pos] = k_chunks[b_keep, h_keep, c, t]
                out_v[b_keep, h_keep, pos] = v_chunks[b_keep, h_keep, c, t]
            write_pos[b_keep, h_keep] += chunk_size

    return out_k, out_v


def compress_all_chunks(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    chunk_size: int,
    beta: float,
    num_steps: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compress ALL chunks unconditionally (used for periodic re-compression).
    Each chunk of chunk_size tokens → 1 prototype token.

    Args:
        key_states:   (B, H, S, D) — the overflow region to compress
        value_states: (B, H, S, D)
    Returns:
        proto_keys:   (B, H, num_chunks, D) — one prototype per chunk
        proto_values: (B, H, num_chunks, D)
    """
    B, H, S, D = key_states.shape

    # Truncate to chunk-aligned length (discard trailing partial chunk)
    num_chunks = S // chunk_size
    usable_len = num_chunks * chunk_size

    if num_chunks == 0:
        # Not enough tokens for even one chunk — return empty
        return (
            key_states.new_zeros(B, H, 0, D),
            value_states.new_zeros(B, H, 0, D),
        )

    k_chunks = key_states[:, :, :usable_len, :].view(B, H, num_chunks, chunk_size, D)
    v_chunks = value_states[:, :, :usable_len, :].view(B, H, num_chunks, chunk_size, D)

    k_protos = hopfield_prototype_batched(k_chunks, beta=beta, num_steps=num_steps)
    v_protos = hopfield_prototype_batched(v_chunks, beta=beta, num_steps=num_steps)

    return k_protos, v_protos  # (B, H, num_chunks, D)


# ---------------------------------------------------------------------------
# 5. Main Attention Module — Streaming Hopfield Cache
# ---------------------------------------------------------------------------

class HopfieldCompressedAttention(LlamaAttention):
    """
    Drop-in replacement for LlamaAttention with streaming Hopfield KV cache.

    v4 — Periodic Re-compression:
      The KV cache is logically split at _compressed_len:

        [ compressed_kv | recent_kv ]

      During decode, new tokens append to recent_kv. When recent_kv exceeds
      (window_size + compression_interval), the oldest compression_interval
      tokens overflow, get Hopfield-compressed into prototypes, and append
      to compressed_kv. This keeps memory bounded.

    Hyperparameters (set via config attributes):
        hopfield_beta           Inverse temperature                     (default: 2.0)
        hopfield_steps          Hopfield iterations per prototype       (default: 1)
        chunk_size              Tokens per compression chunk            (default: 8)
        top_k_ratio             Chunks to keep in initial compression   (default: 0.65)
        rank_iterations         Power-iteration steps for TokenRank     (default: 20)
        compress_threshold      Min seq length for first compression    (default: 32)
        window_size             Recent tokens always kept verbatim      (default: 32)
        compression_interval    New tokens before periodic recompress   (default: 64)
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
        self.compression_interval = getattr(config, "compression_interval", 64)

        # Streaming state
        self._compressed_len = 0    # boundary: tokens [0:_compressed_len] are prototypes
        self._initialized = False   # has initial compression been done?

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

        # ── GQA expansion for attention ───────────────────────────────
        key_states_expanded = repeat_kv(key_states, self.num_key_value_groups)
        value_states_expanded = repeat_kv(value_states, self.num_key_value_groups)

        # ── Scaled dot-product attention ──────────────────────────────
        attn_weights = torch.matmul(
            query_states,
            key_states_expanded.transpose(-2, -1),
        ) * self.scaling

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

        # ==============================================================
        # KV COMPRESSION — Streaming Hopfield Cache
        # ==============================================================
        #
        # Cache layout:  [ compressed_kv | recent_kv ]
        #                  0.._compressed_len  _compressed_len..total
        #
        # Two compression paths:
        #   A) Initial compression: first time total exceeds threshold
        #      → TokenRank-based selective compression on old tokens
        #   B) Periodic re-compression: recent_kv overflows
        #      → compress ALL overflow chunks unconditionally
        # ==============================================================

        recent_len = total_seq_len - self._compressed_len
        max_recent = self.window_size + self.compression_interval
        did_compress = False

        # ── Path A: Initial compression ───────────────────────────────
        if not self._initialized and total_seq_len >= self.compress_threshold:
            window = min(self.window_size, total_seq_len)
            compress_region_len = total_seq_len - window

            if compress_region_len >= self.chunk_size:
                k_old = key_states[:, :, :compress_region_len, :]
                v_old = value_states[:, :, :compress_region_len, :]
                k_recent = key_states[:, :, compress_region_len:, :]
                v_recent = value_states[:, :, compress_region_len:, :]

                with torch.no_grad():
                    kk = torch.matmul(
                        k_old, k_old.transpose(-2, -1),
                    ) * self.scaling
                    causal_kk = torch.triu(
                        torch.full(
                            (compress_region_len, compress_region_len),
                            float("-inf"),
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

                comp_k, comp_v = compress_kv_with_hopfield(
                    k_old, v_old,
                    compress_mask,
                    chunk_size=self.chunk_size,
                    beta=self.hopfield_beta,
                    num_steps=self.hopfield_steps,
                )

                key_states = torch.cat([comp_k, k_recent], dim=2)
                value_states = torch.cat([comp_v, v_recent], dim=2)
                self._compressed_len = comp_k.shape[2]
                self._initialized = True
                did_compress = True

        # ── Path B: Periodic re-compression ───────────────────────────
        elif self._initialized and recent_len > max_recent:
            # How many tokens overflowed past (window + interval)?
            overflow_count = recent_len - self.window_size
            # Align to chunk_size (only compress full chunks)
            overflow_chunks = overflow_count // self.chunk_size
            overflow_aligned = overflow_chunks * self.chunk_size

            if overflow_aligned >= self.chunk_size:
                # Split cache into three regions:
                #   [existing_compressed | overflow | kept_recent]
                split_point = self._compressed_len + overflow_aligned

                k_existing = key_states[:, :, :self._compressed_len, :]
                v_existing = value_states[:, :, :self._compressed_len, :]

                k_overflow = key_states[:, :, self._compressed_len:split_point, :]
                v_overflow = value_states[:, :, self._compressed_len:split_point, :]

                k_kept = key_states[:, :, split_point:, :]
                v_kept = value_states[:, :, split_point:, :]

                # Compress ALL overflow chunks → prototypes
                with torch.no_grad():
                    k_new_protos, v_new_protos = compress_all_chunks(
                        k_overflow, v_overflow,
                        chunk_size=self.chunk_size,
                        beta=self.hopfield_beta,
                        num_steps=self.hopfield_steps,
                    )

                # Reassemble: [existing_compressed + new_protos | kept_recent]
                new_compressed_k = torch.cat([k_existing, k_new_protos], dim=2)
                new_compressed_v = torch.cat([v_existing, v_new_protos], dim=2)

                key_states = torch.cat([new_compressed_k, k_kept], dim=2)
                value_states = torch.cat([new_compressed_v, v_kept], dim=2)
                self._compressed_len = new_compressed_k.shape[2]
                did_compress = True

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
            past_key_values.layers[self.layer_idx].keys = key_states
            past_key_values.layers[self.layer_idx].values = value_states

        # ── Output projection ─────────────────────────────────────────
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_probs
