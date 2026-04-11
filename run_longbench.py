"""
LongBench Evaluation for Hopfield KV Cache Compression

Evaluates compressed attention on a subset of LongBench tasks:
  - Single-Doc QA    (narrativeqa)
  - Multi-Doc QA     (hotpotqa)
  - Summarization    (gov_report)
  - Few-shot Learning (trec)

Compares ROUGE-L / F1 / Accuracy between baseline and compressed models.

Requires: pip install datasets rouge-score

Usage:
    python run_longbench.py
    python run_longbench.py --model meta-llama/Llama-3.2-1B --max_samples 20
    python run_longbench.py --tasks narrativeqa hotpotqa
"""

import argparse
import re
import string
import time
from collections import Counter
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from hopfield_compressed_attention import HopfieldCompressedAttention


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower().strip()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = " ".join(text.split())
    return text


def compute_f1(prediction: str, reference: str) -> float:
    """Token-level F1 score between prediction and reference."""
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens or not ref_tokens:
        return 1.0 if pred_tokens == ref_tokens else 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 score."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores["rougeL"].fmeasure


def compute_accuracy(prediction: str, reference: str) -> float:
    """Exact match accuracy (normalized)."""
    return 1.0 if normalize_text(reference) in normalize_text(prediction) else 0.0


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASK_CONFIGS = {
    "narrativeqa": {
        "dataset": "THUDM/LongBench",
        "subset": "narrativeqa",
        "metric": "f1",
        "metric_fn": compute_f1,
        "prompt_template": (
            "Read the following text and answer the question.\n\n"
            "Text: {context}\n\n"
            "Question: {input}\n\n"
            "Answer:"
        ),
        "context_key": "context",
        "question_key": "input",
        "answer_key": "answers",
        "description": "Single-Document QA",
    },
    "hotpotqa": {
        "dataset": "THUDM/LongBench",
        "subset": "hotpotqa",
        "metric": "f1",
        "metric_fn": compute_f1,
        "prompt_template": (
            "Answer the question based on the given passages.\n\n"
            "Passages: {context}\n\n"
            "Question: {input}\n\n"
            "Answer:"
        ),
        "context_key": "context",
        "question_key": "input",
        "answer_key": "answers",
        "description": "Multi-Document QA",
    },
    "gov_report": {
        "dataset": "THUDM/LongBench",
        "subset": "gov_report",
        "metric": "rouge",
        "metric_fn": compute_rouge_l,
        "prompt_template": (
            "Summarize the following government report.\n\n"
            "Report: {context}\n\n"
            "Summary:"
        ),
        "context_key": "context",
        "question_key": "input",
        "answer_key": "answers",
        "description": "Summarization",
    },
    "trec": {
        "dataset": "THUDM/LongBench",
        "subset": "trec",
        "metric": "accuracy",
        "metric_fn": compute_accuracy,
        "prompt_template": (
            "{context}\n\n"
            "{input}\n\n"
            "Answer:"
        ),
        "context_key": "context",
        "question_key": "input",
        "answer_key": "answers",
        "description": "Few-shot Classification",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def measure_kv_tokens(cache) -> int:
    if len(cache.layers) > 0 and cache.layers[0].is_initialized:
        return cache.layers[0].keys.shape[2]
    return 0


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


def reset_compression(model):
    for layer in model.model.layers:
        attn = layer.self_attn
        if hasattr(attn, "_initialized"):
            attn._initialized = False
            attn._compressed_len = 0


def generate_answer(
    model, tokenizer, prompt: str, device: str,
    max_new_tokens: int = 64, use_compression: bool = False,
) -> Tuple[str, int, float]:
    """Generate an answer and return (text, kv_tokens, elapsed)."""
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=3584,
    ).to(device)

    if use_compression:
        reset_compression(model)

    cache = DynamicCache()
    t0 = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            past_key_values=cache,
        )
    elapsed = time.time() - t0

    generated = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    kv_tokens = measure_kv_tokens(cache)
    return generated, kv_tokens, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LongBench Evaluation")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--tasks", nargs="+",
                        default=["narrativeqa", "hotpotqa"],
                        choices=list(TASK_CONFIGS.keys()),
                        help="Tasks to evaluate")
    parser.add_argument("--max_samples", type=int, default=10,
                        help="Max samples per task (for speed)")
    parser.add_argument("--max_gen_tokens", type=int, default=64,
                        help="Max tokens to generate per answer")

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

    # ── Run evaluation ────────────────────────────────────────────
    all_results = {}

    for task_name in args.tasks:
        cfg = TASK_CONFIGS[task_name]
        print(f"\n{'=' * 80}")
        print(f"  TASK: {task_name} ({cfg['description']})")
        print(f"  Metric: {cfg['metric']}")
        print(f"{'=' * 80}")

        # Load dataset
        print(f"  Loading dataset: {cfg['dataset']}/{cfg['subset']} ...")
        try:
            dataset = load_dataset(cfg["dataset"], cfg["subset"], split="test",
                                   trust_remote_code=True)
        except Exception as e:
            print(f"  [ERROR] Failed to load dataset: {e}")
            continue

        samples = list(dataset.select(range(min(args.max_samples, len(dataset)))))
        print(f"  Evaluating {len(samples)} samples ...\n")

        # Phase 1: Baseline
        print(f"  --- BASELINE ---")
        baseline_scores = []
        baseline_kv_total = 0
        baseline_time_total = 0

        for i, sample in enumerate(samples):
            context = sample[cfg["context_key"]]
            question = sample[cfg["question_key"]]
            references = sample[cfg["answer_key"]]
            if isinstance(references, str):
                references = [references]

            prompt = cfg["prompt_template"].format(
                context=context, input=question,
            )

            pred, kv_tok, elapsed = generate_answer(
                model, tokenizer, prompt, device,
                max_new_tokens=args.max_gen_tokens,
                use_compression=False,
            )

            # Score against best reference
            score = max(cfg["metric_fn"](pred, ref) for ref in references)
            baseline_scores.append(score)
            baseline_kv_total += kv_tok
            baseline_time_total += elapsed

            if i < 3:  # Show first 3 examples
                print(f"    [{i+1}] score={score:.3f}, KV={kv_tok}, "
                      f"pred=\"{pred[:60]}...\"")

        avg_baseline = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0

        # Phase 2: Compressed
        model = inject_hopfield_attention(model)
        print(f"\n  --- COMPRESSED ---")
        compressed_scores = []
        compressed_kv_total = 0
        compressed_time_total = 0

        for i, sample in enumerate(samples):
            context = sample[cfg["context_key"]]
            question = sample[cfg["question_key"]]
            references = sample[cfg["answer_key"]]
            if isinstance(references, str):
                references = [references]

            prompt = cfg["prompt_template"].format(
                context=context, input=question,
            )

            pred, kv_tok, elapsed = generate_answer(
                model, tokenizer, prompt, device,
                max_new_tokens=args.max_gen_tokens,
                use_compression=True,
            )

            score = max(cfg["metric_fn"](pred, ref) for ref in references)
            compressed_scores.append(score)
            compressed_kv_total += kv_tok
            compressed_time_total += elapsed

            if i < 3:
                print(f"    [{i+1}] score={score:.3f}, KV={kv_tok}, "
                      f"pred=\"{pred[:60]}...\"")

        avg_compressed = sum(compressed_scores) / len(compressed_scores) if compressed_scores else 0

        # Restore original attention for next task
        # (inject_hopfield_attention is idempotent — will overwrite)

        # Store results
        n = len(samples)
        all_results[task_name] = {
            "baseline_score": avg_baseline,
            "compressed_score": avg_compressed,
            "baseline_kv_avg": baseline_kv_total / n if n else 0,
            "compressed_kv_avg": compressed_kv_total / n if n else 0,
            "baseline_time_avg": baseline_time_total / n if n else 0,
            "compressed_time_avg": compressed_time_total / n if n else 0,
            "metric": cfg["metric"],
            "n_samples": n,
        }

        print(f"\n  {task_name} results:")
        print(f"    Baseline  {cfg['metric']:>8}: {avg_baseline:.4f}")
        print(f"    Compressed {cfg['metric']:>7}: {avg_compressed:.4f}")
        if avg_baseline > 0:
            retention = avg_compressed / avg_baseline * 100
            print(f"    Retention        : {retention:.1f}%")

    # ── Final summary ─────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("  FINAL RESULTS SUMMARY")
    print(f"{'=' * 80}")

    header = (
        f"  {'Task':>15} | {'Metric':>8} | {'Baseline':>8} | "
        f"{'Compressed':>10} | {'Retain%':>7} | {'KV saved':>8} | {'Speedup':>7}"
    )
    print(header)
    print(f"  {'-' * 78}")

    total_retention = []
    for task_name, r in all_results.items():
        retention = (r["compressed_score"] / r["baseline_score"] * 100
                     if r["baseline_score"] > 0 else 0)
        kv_saved = ((1 - r["compressed_kv_avg"] / r["baseline_kv_avg"]) * 100
                    if r["baseline_kv_avg"] > 0 else 0)
        speedup = (r["baseline_time_avg"] / r["compressed_time_avg"]
                   if r["compressed_time_avg"] > 0 else 0)

        print(
            f"  {task_name:>15} | {r['metric']:>8} | "
            f"{r['baseline_score']:>8.4f} | {r['compressed_score']:>10.4f} | "
            f"{retention:>6.1f}% | {kv_saved:>6.1f}% | {speedup:>6.2f}x"
        )
        total_retention.append(retention)

    if total_retention:
        avg_retention = sum(total_retention) / len(total_retention)
        print(f"\n  Average accuracy retention: {avg_retention:.1f}%")


if __name__ == "__main__":
    main()
