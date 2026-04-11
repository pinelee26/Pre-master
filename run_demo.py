"""
Demo script: Hopfield-Compressed Attention with Streaming KV Cache (v4)

Features:
  - Baseline vs compressed comparison
  - Long-context generation (500+ tokens)
  - Real-time KV cache size monitoring during generation
  - Bounded memory verification

Usage:
    python run_demo.py
    python run_demo.py --max_new_tokens 512 --compression_interval 64
    python run_demo.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --long
"""

import argparse
import sys
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
    StoppingCriteria,
    StoppingCriteriaList,
)

from hopfield_compressed_attention import HopfieldCompressedAttention


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def measure_kv_cache_size(cache) -> int:
    """Return total KV cache size in bytes."""
    total = 0
    for layer in cache.layers:
        if layer.is_initialized:
            total += layer.keys.numel() * layer.keys.element_size()
            total += layer.values.numel() * layer.values.element_size()
    return total


def get_kv_cache_tokens(cache) -> int:
    """Return number of tokens stored in KV cache (from first layer)."""
    if len(cache.layers) > 0 and cache.layers[0].is_initialized:
        return cache.layers[0].keys.shape[2]
    return 0


def inject_hopfield_attention(model):
    """Replace all attention layers with HopfieldCompressedAttention."""
    replaced = 0
    for layer_idx, layer in enumerate(model.model.layers):
        old_attn = layer.self_attn
        new_attn = HopfieldCompressedAttention(model.config, layer_idx=layer_idx)
        new_attn.load_state_dict(old_attn.state_dict(), strict=False)
        new_attn = new_attn.to(
            device=next(old_attn.parameters()).device,
            dtype=next(old_attn.parameters()).dtype,
        )
        layer.self_attn = new_attn
        replaced += 1

    cfg = model.config
    print(f"[+] Replaced {replaced} layers with HopfieldCompressedAttention (v4)")
    print(f"    beta={cfg.hopfield_beta}, steps={cfg.hopfield_steps}, "
          f"chunk={cfg.chunk_size}, top_k={cfg.top_k_ratio}, "
          f"window={cfg.window_size}, interval={cfg.compression_interval}")
    return model


# ---------------------------------------------------------------------------
# KV Cache Monitor — logs cache size at regular intervals during generation
# ---------------------------------------------------------------------------

class KVCacheMonitor(StoppingCriteria):
    """
    Not a stopping criterion — abuses the callback interface to log
    KV cache metrics during generation without stopping.
    """

    def __init__(self, cache, log_every: int = 50, label: str = ""):
        self.cache = cache
        self.log_every = log_every
        self.label = label
        self.step = 0
        self.history = []  # [(step, kv_tokens, kv_kb)]

    def __call__(self, input_ids, scores, **kwargs):
        self.step += 1
        if self.step % self.log_every == 0 or self.step == 1:
            kv_tokens = get_kv_cache_tokens(self.cache)
            kv_kb = measure_kv_cache_size(self.cache) / 1024
            self.history.append((self.step, kv_tokens, kv_kb))
            print(
                f"    [{self.label}] step {self.step:>4d}  |  "
                f"KV tokens: {kv_tokens:>5d}  |  "
                f"KV cache: {kv_kb:>8.1f} KB"
            )
        return False  # Never stop


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

LONG_PROMPT = (
    "The development of large language models has fundamentally transformed the "
    "landscape of artificial intelligence research. Starting from the early days "
    "of recurrent neural networks and long short-term memory architectures, the "
    "field has witnessed a paradigm shift with the introduction of the transformer "
    "architecture in 2017. The key innovation of self-attention mechanisms allowed "
    "models to capture long-range dependencies in text without the sequential "
    "processing bottleneck that plagued earlier approaches. This breakthrough led "
    "to a rapid scaling of model parameters, from millions to billions and "
    "eventually trillions, each generation demonstrating emergent capabilities "
    "that were not explicitly programmed. However, this scaling has introduced "
    "significant computational challenges, particularly in managing the key-value "
    "cache during autoregressive generation, where memory consumption grows "
    "linearly with sequence length."
)


def main():
    parser = argparse.ArgumentParser(description="Hopfield Streaming Cache Demo (v4)")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--prompt", type=str, default=LONG_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Tokens to generate (default 512 for long-context test)")
    parser.add_argument("--log_every", type=int, default=50,
                        help="Log KV cache size every N tokens")

    # v4 hyperparameters
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--hopfield_steps", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--top_k_ratio", type=float, default=0.65)
    parser.add_argument("--compress_threshold", type=int, default=32)
    parser.add_argument("--rank_iterations", type=int, default=20)
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--compression_interval", type=int, default=64,
                        help="New tokens accumulated before periodic re-compression")

    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Device: {device}")
    print(f"Loading model: {args.model} ...")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        device_map=device,
        attn_implementation="eager",
    )
    model.eval()

    # Set config
    model.config.hopfield_beta = args.beta
    model.config.hopfield_steps = args.hopfield_steps
    model.config.chunk_size = args.chunk_size
    model.config.top_k_ratio = args.top_k_ratio
    model.config.compress_threshold = args.compress_threshold
    model.config.rank_iterations = args.rank_iterations
    model.config.window_size = args.window_size
    model.config.compression_interval = args.compression_interval

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    prompt_tokens = inputs["input_ids"].shape[1]
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Target generation: {args.max_new_tokens} tokens")

    # ==================================================================
    # BASELINE
    # ==================================================================
    print("\n" + "=" * 80)
    print("  BASELINE (standard LlamaAttention) — KV cache grows linearly")
    print("=" * 80)

    baseline_cache = DynamicCache()
    baseline_monitor = KVCacheMonitor(
        baseline_cache, log_every=args.log_every, label="baseline",
    )

    t0 = time.time()
    with torch.no_grad():
        baseline_out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            past_key_values=baseline_cache,
            stopping_criteria=StoppingCriteriaList([baseline_monitor]),
        )
    baseline_time = time.time() - t0
    baseline_text = tokenizer.decode(baseline_out[0], skip_special_tokens=True)
    baseline_gen = tokenizer.decode(
        baseline_out[0][prompt_tokens:], skip_special_tokens=True,
    )
    baseline_kv = measure_kv_cache_size(baseline_cache)
    baseline_tokens = get_kv_cache_tokens(baseline_cache)

    print(f"\n  Final: {baseline_tokens} KV tokens, "
          f"{baseline_kv / 1024:.1f} KB, {baseline_time:.2f}s")
    print(f"  Generated (first 150 chars): {baseline_gen[:150]}...")

    # ==================================================================
    # COMPRESSED (v4 Streaming)
    # ==================================================================
    model = inject_hopfield_attention(model)

    print("\n" + "=" * 80)
    print("  HOPFIELD STREAMING CACHE (v4) — KV cache should be bounded")
    print("=" * 80)

    compressed_cache = DynamicCache()
    compressed_monitor = KVCacheMonitor(
        compressed_cache, log_every=args.log_every, label="hopfield",
    )

    t0 = time.time()
    with torch.no_grad():
        compressed_out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            past_key_values=compressed_cache,
            stopping_criteria=StoppingCriteriaList([compressed_monitor]),
        )
    compressed_time = time.time() - t0
    compressed_text = tokenizer.decode(compressed_out[0], skip_special_tokens=True)
    compressed_gen = tokenizer.decode(
        compressed_out[0][prompt_tokens:], skip_special_tokens=True,
    )
    compressed_kv = measure_kv_cache_size(compressed_cache)
    compressed_tokens = get_kv_cache_tokens(compressed_cache)

    print(f"\n  Final: {compressed_tokens} KV tokens, "
          f"{compressed_kv / 1024:.1f} KB, {compressed_time:.2f}s")
    print(f"  Generated (first 150 chars): {compressed_gen[:150]}...")

    # ==================================================================
    # COMPARISON & GROWTH ANALYSIS
    # ==================================================================
    print("\n" + "=" * 80)
    print("  COMPARISON")
    print("=" * 80)

    if baseline_kv > 0:
        ratio = compressed_kv / baseline_kv
        print(f"  Final KV cache   : {compressed_kv / 1024:.1f} KB vs "
              f"{baseline_kv / 1024:.1f} KB baseline")
        print(f"  Compression ratio: {ratio:.2%} ({(1 - ratio) * 100:.1f}% saved)")

    print(f"  KV tokens        : {compressed_tokens} vs {baseline_tokens} baseline")
    print(f"  Generation time  : {compressed_time:.2f}s vs {baseline_time:.2f}s")

    if baseline_time > 0:
        speedup = baseline_time / compressed_time
        print(f"  Speedup          : {speedup:.2f}x")

    # Growth analysis
    print(f"\n  {'─' * 60}")
    print(f"  KV CACHE GROWTH TRAJECTORY")
    print(f"  {'─' * 60}")
    print(f"  {'step':>6}  {'baseline KV':>14}  {'compressed KV':>14}  {'ratio':>8}")
    print(f"  {'─' * 60}")

    bl_hist = {s: (t, k) for s, t, k in baseline_monitor.history}
    cp_hist = {s: (t, k) for s, t, k in compressed_monitor.history}

    all_steps = sorted(set(bl_hist.keys()) | set(cp_hist.keys()))
    for step in all_steps:
        bl_tok, bl_kb = bl_hist.get(step, (0, 0))
        cp_tok, cp_kb = cp_hist.get(step, (0, 0))
        r = cp_kb / bl_kb if bl_kb > 0 else 0
        print(
            f"  {step:>6}  {bl_tok:>6} ({bl_kb:>6.1f}KB)  "
            f"{cp_tok:>6} ({cp_kb:>6.1f}KB)  {r:>7.1%}"
        )

    # Bounded check
    if compressed_monitor.history:
        kv_sizes = [kb for _, _, kb in compressed_monitor.history]
        max_kv = max(kv_sizes)
        final_kv = kv_sizes[-1]
        growth_ratio = final_kv / kv_sizes[0] if kv_sizes[0] > 0 else 0

        bl_sizes = [kb for _, _, kb in baseline_monitor.history]
        bl_growth = bl_sizes[-1] / bl_sizes[0] if bl_sizes[0] > 0 else 0

        print(f"\n  Baseline growth factor : {bl_growth:.2f}x "
              f"(first → last measurement)")
        print(f"  Compressed growth factor: {growth_ratio:.2f}x "
              f"(first → last measurement)")

        if growth_ratio < bl_growth * 0.5:
            print(f"  ✓ BOUNDED: compressed cache grows {growth_ratio / bl_growth:.0%} "
                  f"as fast as baseline")
        elif growth_ratio < bl_growth * 0.8:
            print(f"  ◎ PARTIALLY BOUNDED: growth rate reduced to "
                  f"{growth_ratio / bl_growth:.0%} of baseline")
        else:
            print(f"  ✗ NOT BOUNDED: growth rate similar to baseline")


if __name__ == "__main__":
    main()
