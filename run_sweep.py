"""
Hyperparameter Sweep: Grid search for optimal Hopfield KV cache compression settings.

Sweeps over (top_k_ratio, chunk_size, window_size) combinations and reports:
  - KV cache compression ratio
  - Memory saved (%)
  - Speed vs baseline
  - Repetition loop detection (n-gram overlap ratio)
  - Generated text quality (first 80 chars)

Usage:
    python run_sweep.py
    python run_sweep.py --model meta-llama/Llama-3.2-1B
    python run_sweep.py --max_new_tokens 128
"""

import argparse
import copy
import itertools
import re
import time
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from hopfield_compressed_attention import HopfieldCompressedAttention


# ---------------------------------------------------------------------------
# Repetition detection
# ---------------------------------------------------------------------------

def compute_repetition_ratio(text: str, n: int = 4) -> float:
    """
    Measure repetition via n-gram overlap ratio.

    Returns a value in [0, 1]:
      0.0 = all n-grams are unique (no repetition)
      1.0 = every n-gram is repeated (extreme loop)
    """
    words = text.lower().split()
    if len(words) < n:
        return 0.0

    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    total = len(ngrams)
    unique = len(set(ngrams))

    if total == 0:
        return 0.0

    return 1.0 - (unique / total)


def detect_repetition_loop(text: str, threshold: float = 0.3) -> bool:
    """Return True if the text shows significant repetition (loop detected)."""
    return compute_repetition_ratio(text, n=4) > threshold


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


def inject_hopfield_attention(model):
    """Replace all attention layers with HopfieldCompressedAttention."""
    for layer_idx, layer in enumerate(model.model.layers):
        old_attn = layer.self_attn
        new_attn = HopfieldCompressedAttention(model.config, layer_idx=layer_idx)
        new_attn.load_state_dict(old_attn.state_dict(), strict=False)
        new_attn = new_attn.to(
            device=next(old_attn.parameters()).device,
            dtype=next(old_attn.parameters()).dtype,
        )
        layer.self_attn = new_attn
    return model


def reset_compression_state(model):
    """Reset streaming state on all attention layers for a fresh run."""
    for layer in model.model.layers:
        attn = layer.self_attn
        if hasattr(attn, "_initialized"):
            attn._initialized = False
            attn._compressed_len = 0


# ---------------------------------------------------------------------------
# Main sweep
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
    parser = argparse.ArgumentParser(description="Hopfield Compression Hyperparameter Sweep")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--prompt", type=str, default=LONG_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--beta", type=float, default=2.0,
                        help="Fixed beta for all sweep runs")
    parser.add_argument("--hopfield_steps", type=int, default=1,
                        help="Fixed Hopfield steps for all sweep runs")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Auto-detect device
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

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    prompt_tokens = inputs["input_ids"].shape[1]
    print(f"Prompt tokens: {prompt_tokens}")

    # ── Baseline ──────────────────────────────────────────────────
    print("\nRunning baseline ...")
    t0 = time.time()
    with torch.no_grad():
        baseline_cache = DynamicCache()
        baseline_out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            past_key_values=baseline_cache,
        )
    baseline_time = time.time() - t0
    baseline_text = tokenizer.decode(baseline_out[0], skip_special_tokens=True)
    baseline_kv = measure_kv_cache_size(baseline_cache)
    baseline_rep = compute_repetition_ratio(baseline_text)

    # Extract only the generated part (after prompt)
    baseline_gen = tokenizer.decode(
        baseline_out[0][prompt_tokens:], skip_special_tokens=True,
    )

    print(f"Baseline: {baseline_kv / 1024:.0f} KB, {baseline_time:.2f}s, "
          f"rep={baseline_rep:.3f}")
    print(f"  Generated: {baseline_gen[:100]}...")

    # ── Sweep grid ────────────────────────────────────────────────
    top_k_ratios = [0.60, 0.65, 0.70]
    chunk_sizes = [4, 8, 16]
    window_sizes = [16, 32, 64]

    # Fixed params
    beta = args.beta
    steps = args.hopfield_steps

    # Inject once — we'll reset state between runs
    model.config.hopfield_beta = beta
    model.config.hopfield_steps = steps
    model.config.compress_threshold = 32
    model.config.rank_iterations = 20
    model.config.compression_interval = 64
    # Set initial values (will be overwritten per run)
    model.config.chunk_size = 8
    model.config.top_k_ratio = 0.65
    model.config.window_size = 32
    model = inject_hopfield_attention(model)

    combos = list(itertools.product(top_k_ratios, chunk_sizes, window_sizes))
    total_runs = len(combos)

    # ── Results table ─────────────────────────────────────────────
    sep = "-" * 120
    header = (
        f"{'#':>3} | {'top_k':>5} | {'chunk':>5} | {'window':>6} | "
        f"{'KV(KB)':>7} | {'saved%':>6} | {'time(s)':>7} | {'speedup':>7} | "
        f"{'rep_ratio':>9} | {'loop?':>5} | {'generated (first 50 chars)':>50}"
    )

    print(f"\n{'=' * 120}")
    print(f"  SWEEP: beta={beta}, steps={steps}, max_new_tokens={args.max_new_tokens}")
    print(f"  Grid: top_k_ratio={top_k_ratios} x chunk_size={chunk_sizes} x window_size={window_sizes}")
    print(f"  Total combinations: {total_runs}")
    print(f"{'=' * 120}")
    print(header)
    print(sep)

    # Baseline row
    loop_tag = "YES" if detect_repetition_loop(baseline_text) else "no"
    print(
        f"{'BL':>3} | {'--':>5} | {'--':>5} | {'--':>6} | "
        f"{baseline_kv / 1024:>7.0f} | {'0.0%':>6} | {baseline_time:>7.2f} | "
        f"{'1.00x':>7} | {baseline_rep:>9.3f} | {loop_tag:>5} | "
        f"{baseline_gen[:50]}"
    )
    print(sep)

    results = []

    for i, (tkr, cs, ws) in enumerate(combos, 1):
        # Update config for this run
        for layer in model.model.layers:
            attn = layer.self_attn
            attn.top_k_ratio = tkr
            attn.chunk_size = cs
            attn.window_size = ws
            attn._initialized = False
            attn._compressed_len = 0

        t0 = time.time()
        with torch.no_grad():
            cache = DynamicCache()
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                past_key_values=cache,
            )
        elapsed = time.time() - t0

        text = tokenizer.decode(out[0], skip_special_tokens=True)
        gen_text = tokenizer.decode(out[0][prompt_tokens:], skip_special_tokens=True)
        kv_size = measure_kv_cache_size(cache)

        saved = (1 - kv_size / baseline_kv) * 100 if baseline_kv > 0 else 0
        speedup = baseline_time / elapsed if elapsed > 0 else 0
        rep = compute_repetition_ratio(text)
        is_loop = detect_repetition_loop(text)
        loop_tag = "YES" if is_loop else "no"

        print(
            f"{i:>3} | {tkr:>5.2f} | {cs:>5} | {ws:>6} | "
            f"{kv_size / 1024:>7.0f} | {saved:>5.1f}% | {elapsed:>7.2f} | "
            f"{speedup:>6.2f}x | {rep:>9.3f} | {loop_tag:>5} | "
            f"{gen_text[:50]}"
        )

        results.append({
            "top_k_ratio": tkr,
            "chunk_size": cs,
            "window_size": ws,
            "kv_kb": kv_size / 1024,
            "saved_pct": saved,
            "time": elapsed,
            "speedup": speedup,
            "rep_ratio": rep,
            "is_loop": is_loop,
            "text_preview": gen_text[:80],
        })

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 120}")
    print("  SUMMARY")
    print(f"{'=' * 120}")

    # Best: highest memory saving without repetition loop
    good = [r for r in results if not r["is_loop"]]
    if good:
        best = max(good, key=lambda r: r["saved_pct"])
        print(f"\n  [BEST — no loop, max compression]")
        print(f"    top_k_ratio={best['top_k_ratio']}, "
              f"chunk_size={best['chunk_size']}, "
              f"window_size={best['window_size']}")
        print(f"    Memory saved: {best['saved_pct']:.1f}%  |  "
              f"Speed: {best['speedup']:.2f}x  |  "
              f"Rep ratio: {best['rep_ratio']:.3f}")
        print(f"    Preview: {best['text_preview']}")
    else:
        print("\n  [WARNING] All combinations triggered repetition loops!")

    # Best speed
    if good:
        fastest = max(good, key=lambda r: r["speedup"])
        if fastest != best:
            print(f"\n  [FASTEST — no loop]")
            print(f"    top_k_ratio={fastest['top_k_ratio']}, "
                  f"chunk_size={fastest['chunk_size']}, "
                  f"window_size={fastest['window_size']}")
            print(f"    Memory saved: {fastest['saved_pct']:.1f}%  |  "
                  f"Speed: {fastest['speedup']:.2f}x  |  "
                  f"Rep ratio: {fastest['rep_ratio']:.3f}")

    # Loop count
    loop_count = sum(1 for r in results if r["is_loop"])
    print(f"\n  Repetition loops detected: {loop_count}/{len(results)} combinations")
    print(f"  Baseline rep_ratio: {baseline_rep:.3f} (reference)\n")


if __name__ == "__main__":
    main()
