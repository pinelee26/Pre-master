"""
Demo script: Inject HopfieldCompressedAttention into a Llama model and run text generation.

Usage:
    python run_demo.py                           # Use a small Llama model from HF
    python run_demo.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python run_demo.py --prompt "The future of AI is"
    python run_demo.py --beta 2.0 --chunk_size 8 --top_k_ratio 0.65 --window_size 32
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from hopfield_compressed_attention import HopfieldCompressedAttention


def inject_hopfield_attention(model, config):
    """
    Replace every LlamaAttention layer in the model with HopfieldCompressedAttention,
    copying the original weights so the model behaves identically except for KV compression.
    """
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

    print(f"[+] Replaced {replaced} attention layers with HopfieldCompressedAttention")
    print(f"    beta={model.config.hopfield_beta}, steps={model.config.hopfield_steps}, "
          f"chunk_size={model.config.chunk_size}, top_k_ratio={model.config.top_k_ratio}, "
          f"window={model.config.window_size}")
    return model


def measure_kv_cache_size(cache) -> int:
    """Return total KV cache size in bytes."""
    total = 0
    for layer in cache.layers:
        if layer.is_initialized:
            total += layer.keys.numel() * layer.keys.element_size()
            total += layer.values.numel() * layer.values.element_size()
    return total


def main():
    parser = argparse.ArgumentParser(description="Hopfield-Compressed Attention Demo")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="HuggingFace model ID")
    parser.add_argument("--prompt", type=str,
                        default="The key innovation in modern transformer architectures is",
                        help="Input prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="Number of tokens to generate")

    # Hopfield compression hyperparameters (v3 defaults)
    parser.add_argument("--beta", type=float, default=2.0,
                        help="Hopfield inverse temperature (higher = sharper prototype)")
    parser.add_argument("--hopfield_steps", type=int, default=1,
                        help="Hopfield update iterations per prototype")
    parser.add_argument("--chunk_size", type=int, default=8,
                        help="Tokens per chunk for compression")
    parser.add_argument("--top_k_ratio", type=float, default=0.65,
                        help="Fraction of chunks to keep uncompressed")
    parser.add_argument("--compress_threshold", type=int, default=32,
                        help="Min sequence length to trigger compression")
    parser.add_argument("--rank_iterations", type=int, default=20,
                        help="Power-iteration steps for TokenRank")
    parser.add_argument("--window_size", type=int, default=32,
                        help="Recent tokens to keep uncompressed (observe window)")

    # Device
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detected if omitted)")
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

    # Set compression hyperparameters on config
    model.config.hopfield_beta = args.beta
    model.config.hopfield_steps = args.hopfield_steps
    model.config.chunk_size = args.chunk_size
    model.config.top_k_ratio = args.top_k_ratio
    model.config.compress_threshold = args.compress_threshold
    model.config.rank_iterations = args.rank_iterations
    model.config.window_size = args.window_size

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    # ── Baseline ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  BASELINE (standard LlamaAttention)")
    print("=" * 70)

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
    baseline_kv_size = measure_kv_cache_size(baseline_cache)

    print(f"  Prompt : {args.prompt}")
    print(f"  Output : {baseline_text}")
    print(f"  Time   : {baseline_time:.2f}s")
    print(f"  KV size: {baseline_kv_size / 1024:.1f} KB")

    # ── Inject compressed attention ───────────────────────────────
    model = inject_hopfield_attention(model, model.config)

    # ── Compressed generation ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("  HOPFIELD-COMPRESSED ATTENTION (v3)")
    print("=" * 70)

    t0 = time.time()
    with torch.no_grad():
        compressed_cache = DynamicCache()
        compressed_out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            past_key_values=compressed_cache,
        )
    compressed_time = time.time() - t0
    compressed_text = tokenizer.decode(compressed_out[0], skip_special_tokens=True)
    compressed_kv_size = measure_kv_cache_size(compressed_cache)

    print(f"  Prompt : {args.prompt}")
    print(f"  Output : {compressed_text}")
    print(f"  Time   : {compressed_time:.2f}s")
    print(f"  KV size: {compressed_kv_size / 1024:.1f} KB")

    # ── Comparison ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  COMPARISON")
    print("=" * 70)
    if baseline_kv_size > 0:
        ratio = compressed_kv_size / baseline_kv_size
        saving = (1 - ratio) * 100
        print(f"  KV cache ratio  : {ratio:.2%} of baseline")
        print(f"  Memory saved    : {saving:.1f}%")
    print(f"  Speed           : baseline {baseline_time:.2f}s  vs  compressed {compressed_time:.2f}s")
    if baseline_time > 0:
        overhead = (compressed_time - baseline_time) / baseline_time * 100
        print(f"  Speed overhead  : {overhead:+.1f}%")


if __name__ == "__main__":
    main()
