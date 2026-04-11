"""
Needle-In-A-Haystack (NIAH) Benchmark for Hopfield KV Cache Compression

Tests whether the model can retrieve a specific hidden fact ("needle")
embedded at various positions within a long context ("haystack"),
comparing baseline (full KV cache) vs compressed (Hopfield streaming cache).

Produces a 2D heatmap-style table:
    rows    = context length (how long the haystack is)
    columns = needle position (where in the haystack the fact is hidden)
    cells   = retrieval accuracy (1 = found, 0 = missed)

Usage:
    python run_niah.py
    python run_niah.py --model meta-llama/Llama-3.2-1B --max_context 1024
"""

import argparse
import json
import re
import time
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from hopfield_compressed_attention import HopfieldCompressedAttention


# ---------------------------------------------------------------------------
# Haystack construction
# ---------------------------------------------------------------------------

FILLER_PARAGRAPHS = [
    "The history of computing traces back to the earliest mechanical calculators "
    "developed in the 17th century. Charles Babbage conceptualized the Analytical "
    "Engine in 1837, which contained many features found in modern computers. ",

    "Ocean currents play a vital role in regulating the Earth's climate by "
    "distributing heat from the equator toward the poles. The Gulf Stream, for "
    "instance, carries warm water from the Gulf of Mexico across the Atlantic. ",

    "The Renaissance period, spanning roughly from the 14th to the 17th century, "
    "marked a profound cultural transformation in Europe. Artists like Leonardo "
    "da Vinci and Michelangelo pushed the boundaries of human expression. ",

    "Photosynthesis is the process by which green plants convert sunlight into "
    "chemical energy. This remarkable biological mechanism sustains nearly all "
    "life on Earth by producing oxygen and organic compounds from carbon dioxide. ",

    "The development of antibiotics in the 20th century revolutionized medicine. "
    "Alexander Fleming's discovery of penicillin in 1928 opened a new era in "
    "the treatment of bacterial infections that had previously been fatal. ",

    "Plate tectonics theory explains the large-scale motion of Earth's lithosphere. "
    "The movement of tectonic plates causes earthquakes, volcanic eruptions, "
    "and the creation of mountain ranges over millions of years. ",

    "The invention of the printing press by Johannes Gutenberg around 1440 "
    "transformed the spread of knowledge across Europe. Books became more "
    "accessible, literacy rates increased, and ideas could be shared widely. ",

    "Quantum mechanics describes the behavior of matter and energy at the "
    "smallest scales. The uncertainty principle, wave-particle duality, and "
    "quantum entanglement challenge our classical understanding of physics. ",
]

# The "needle" — a distinctive, easily verifiable fact
NEEDLE = (
    "The special secret code for this experiment is: PURPLE ELEPHANT 7492. "
    "Remember this code carefully as you may be asked about it later."
)

QUERY = "What is the special secret code mentioned in the text?"
ANSWER_KEY = "PURPLE ELEPHANT 7492"


def build_haystack(
    tokenizer,
    target_token_count: int,
    needle_position_ratio: float,  # 0.0 = start, 0.5 = middle, 1.0 = end
) -> str:
    """
    Build a haystack of approximately `target_token_count` tokens with the
    needle inserted at `needle_position_ratio` through the text.
    """
    # Build filler text by repeating paragraphs
    filler_chunks = []
    current_tokens = 0
    needle_tokens = len(tokenizer.encode(NEEDLE))

    idx = 0
    while current_tokens < target_token_count - needle_tokens:
        para = FILLER_PARAGRAPHS[idx % len(FILLER_PARAGRAPHS)]
        filler_chunks.append(para)
        current_tokens += len(tokenizer.encode(para))
        idx += 1

    # Insert needle at the specified position
    insert_idx = max(0, int(len(filler_chunks) * needle_position_ratio))
    filler_chunks.insert(insert_idx, NEEDLE)

    return "\n\n".join(filler_chunks)


# ---------------------------------------------------------------------------
# Retrieval check
# ---------------------------------------------------------------------------

def check_retrieval(generated_text: str) -> Tuple[bool, float]:
    """
    Check if the model successfully retrieved the needle.

    Returns:
        (found, confidence)
        found: True if the answer key appears in the generated text
        confidence: rough match score (1.0 = exact, partial for partial match)
    """
    text_lower = generated_text.lower()
    answer_lower = ANSWER_KEY.lower()

    # Exact match
    if answer_lower in text_lower:
        return True, 1.0

    # Partial match: check for key components
    components = ["purple", "elephant", "7492"]
    matches = sum(1 for c in components if c in text_lower)
    confidence = matches / len(components)

    return confidence >= 0.67, confidence  # 2/3 components = pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def measure_kv_cache_size(cache) -> int:
    total = 0
    for layer in cache.layers:
        if layer.is_initialized:
            total += layer.keys.numel() * layer.keys.element_size()
            total += layer.values.numel() * layer.values.element_size()
    return total


def inject_hopfield_attention(model):
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


def restore_original_attention(model, original_state):
    """Restore original attention layers from saved state."""
    for layer_idx, layer in enumerate(model.model.layers):
        layer.self_attn = original_state[layer_idx]


def run_single_test(
    model, tokenizer, device, context_text: str, use_compression: bool,
) -> Tuple[bool, float, int, float]:
    """
    Run a single NIAH test.

    Returns: (found, confidence, kv_tokens, elapsed)
    """
    prompt = f"{context_text}\n\nQuestion: {QUERY}\nAnswer: The special secret code is"

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=4096,
    ).to(device)

    # Reset compression state if using Hopfield attention
    if use_compression:
        for layer in model.model.layers:
            attn = layer.self_attn
            if hasattr(attn, "_initialized"):
                attn._initialized = False
                attn._compressed_len = 0

    cache = DynamicCache()
    t0 = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            past_key_values=cache,
        )
    elapsed = time.time() - t0

    generated = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    found, confidence = check_retrieval(generated)
    kv_tokens = cache.layers[0].keys.shape[2] if cache.layers else 0

    return found, confidence, kv_tokens, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Needle-In-A-Haystack Benchmark")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--max_context", type=int, default=1536,
                        help="Maximum context length in tokens")
    parser.add_argument("--num_lengths", type=int, default=4,
                        help="Number of context length steps to test")
    parser.add_argument("--num_positions", type=int, default=5,
                        help="Number of needle positions to test (start→end)")

    # Compression params
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--top_k_ratio", type=float, default=0.65)
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--compression_interval", type=int, default=64)

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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        device_map=device,
        attn_implementation="eager",
    )
    model.eval()

    # Set compression config
    model.config.hopfield_beta = args.beta
    model.config.hopfield_steps = 1
    model.config.chunk_size = args.chunk_size
    model.config.top_k_ratio = args.top_k_ratio
    model.config.compress_threshold = 32
    model.config.rank_iterations = 20
    model.config.window_size = args.window_size
    model.config.compression_interval = args.compression_interval

    # Test grid
    min_context = 256
    context_lengths = [
        int(min_context + (args.max_context - min_context) * i / (args.num_lengths - 1))
        for i in range(args.num_lengths)
    ]
    needle_positions = [
        i / (args.num_positions - 1)
        for i in range(args.num_positions)
    ]
    position_labels = ["Start", "25%", "Mid", "75%", "End"][:args.num_positions]

    print(f"\nContext lengths: {context_lengths}")
    print(f"Needle positions: {[f'{p:.0%}' for p in needle_positions]}")
    print(f"Needle: \"{ANSWER_KEY}\"")

    # ── Save original attention for baseline runs ─────────────────
    original_attns = [layer.self_attn for layer in model.model.layers]

    # ── Run baseline ──────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("  PHASE 1: BASELINE (full KV cache)")
    print(f"{'=' * 90}")

    baseline_results = {}
    for ctx_len in context_lengths:
        for pos_idx, pos_ratio in enumerate(needle_positions):
            haystack = build_haystack(tokenizer, ctx_len, pos_ratio)
            found, conf, kv_tok, elapsed = run_single_test(
                model, tokenizer, device, haystack, use_compression=False,
            )
            baseline_results[(ctx_len, pos_idx)] = (found, conf, kv_tok, elapsed)
            status = "HIT" if found else "MISS"
            print(f"  ctx={ctx_len:>5}, pos={position_labels[pos_idx]:>5} → "
                  f"{status} (conf={conf:.2f}, KV={kv_tok}, {elapsed:.1f}s)")

    # ── Inject compression and run ────────────────────────────────
    model = inject_hopfield_attention(model)

    print(f"\n{'=' * 90}")
    print("  PHASE 2: HOPFIELD COMPRESSED (streaming cache)")
    print(f"{'=' * 90}")

    compressed_results = {}
    for ctx_len in context_lengths:
        for pos_idx, pos_ratio in enumerate(needle_positions):
            haystack = build_haystack(tokenizer, ctx_len, pos_ratio)
            found, conf, kv_tok, elapsed = run_single_test(
                model, tokenizer, device, haystack, use_compression=True,
            )
            compressed_results[(ctx_len, pos_idx)] = (found, conf, kv_tok, elapsed)
            status = "HIT" if found else "MISS"
            print(f"  ctx={ctx_len:>5}, pos={position_labels[pos_idx]:>5} → "
                  f"{status} (conf={conf:.2f}, KV={kv_tok}, {elapsed:.1f}s)")

    # ── Results heatmap ───────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("  RETRIEVAL ACCURACY HEATMAP")
    print(f"{'=' * 90}")

    # Header
    pos_header = "  ".join(f"{lbl:>7}" for lbl in position_labels)
    print(f"\n  BASELINE (full cache)")
    print(f"  {'ctx_len':>7} | {pos_header}")
    print(f"  {'-' * (10 + 9 * len(position_labels))}")
    for ctx_len in context_lengths:
        cells = []
        for pos_idx in range(len(needle_positions)):
            found, conf, _, _ = baseline_results[(ctx_len, pos_idx)]
            mark = f"{'O':>7}" if found else f"{'X':>7}"
            cells.append(mark)
        print(f"  {ctx_len:>7} | {'  '.join(cells)}")

    baseline_hits = sum(1 for v in baseline_results.values() if v[0])
    baseline_total = len(baseline_results)
    print(f"  Accuracy: {baseline_hits}/{baseline_total} "
          f"({baseline_hits / baseline_total * 100:.1f}%)")

    print(f"\n  COMPRESSED (Hopfield streaming)")
    print(f"  {'ctx_len':>7} | {pos_header}")
    print(f"  {'-' * (10 + 9 * len(position_labels))}")
    for ctx_len in context_lengths:
        cells = []
        for pos_idx in range(len(needle_positions)):
            found, conf, _, _ = compressed_results[(ctx_len, pos_idx)]
            mark = f"{'O':>7}" if found else f"{'X':>7}"
            cells.append(mark)
        print(f"  {ctx_len:>7} | {'  '.join(cells)}")

    compressed_hits = sum(1 for v in compressed_results.values() if v[0])
    compressed_total = len(compressed_results)
    print(f"  Accuracy: {compressed_hits}/{compressed_total} "
          f"({compressed_hits / compressed_total * 100:.1f}%)")

    # ── KV cache comparison ───────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("  KV CACHE SIZE COMPARISON")
    print(f"{'=' * 90}")
    print(f"  {'ctx_len':>7} | {'baseline KV':>12} | {'compressed KV':>14} | {'saved':>7}")
    print(f"  {'-' * 55}")
    for ctx_len in context_lengths:
        bl_kv = baseline_results[(ctx_len, 0)][2]
        cp_kv = compressed_results[(ctx_len, 0)][2]
        saved = (1 - cp_kv / bl_kv) * 100 if bl_kv > 0 else 0
        print(f"  {ctx_len:>7} | {bl_kv:>8} tok | {cp_kv:>10} tok | {saved:>5.1f}%")

    # ── Final summary ─────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 90}")
    print(f"  Baseline accuracy   : {baseline_hits}/{baseline_total} "
          f"({baseline_hits / baseline_total * 100:.1f}%)")
    print(f"  Compressed accuracy : {compressed_hits}/{compressed_total} "
          f"({compressed_hits / compressed_total * 100:.1f}%)")

    if baseline_hits > 0:
        retention = compressed_hits / baseline_hits * 100
        print(f"  Accuracy retention  : {retention:.1f}%")

    # Average KV savings
    avg_bl_kv = sum(v[2] for v in baseline_results.values()) / baseline_total
    avg_cp_kv = sum(v[2] for v in compressed_results.values()) / compressed_total
    print(f"  Avg KV tokens       : {avg_cp_kv:.0f} vs {avg_bl_kv:.0f} baseline "
          f"({(1 - avg_cp_kv / avg_bl_kv) * 100:.1f}% saved)")

    # Average speed
    avg_bl_time = sum(v[3] for v in baseline_results.values()) / baseline_total
    avg_cp_time = sum(v[3] for v in compressed_results.values()) / compressed_total
    if avg_cp_time > 0:
        print(f"  Avg speed           : {avg_bl_time / avg_cp_time:.2f}x")


if __name__ == "__main__":
    main()
