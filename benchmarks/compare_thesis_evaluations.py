"""
Compare Thesis Evaluations
==========================

Compares two outputs produced by `thesis_evaluation.py`, usually:

- pre_rag evaluation
- post_rag evaluation

Example:
  python compare_thesis_evaluations.py --before logs/thesis_eval_pre_rag.json --after logs/thesis_eval_post_rag.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = [
    "financial_accuracy_pct",
    "role_appropriateness_pct",
    "personalization_quality_pct",
    "hallucination_rate_pct",
    "citation_correctness_pct",
]

LOWER_IS_BETTER = {"hallucination_rate_pct"}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float | int | None) -> str:
    return "N/A" if value is None else f"{value:.2f}%"


def delta(before: float | int | None, after: float | int | None, metric: str) -> str:
    if before is None or after is None:
        return "N/A"
    raw = after - before
    if metric in LOWER_IS_BETTER:
        improved = raw < 0
    else:
        improved = raw > 0
    marker = "+" if raw >= 0 else ""
    verdict = "improved" if improved else "declined" if raw else "same"
    return f"{marker}{raw:.2f} pp ({verdict})"


def print_table(before: dict, after: dict) -> None:
    before_agg = before.get("aggregate", {})
    after_agg = after.get("aggregate", {})
    before_agg = before_agg.get("model_evaluation_metrics", before_agg)
    after_agg = after_agg.get("model_evaluation_metrics", after_agg)

    print("\nThesis Evaluation Comparison")
    print("=" * 92)
    print(f"{'Metric':35s} {'Before':>14s} {'After':>14s} {'Delta':>24s}")
    print("-" * 92)
    for metric in METRICS:
        b = before_agg.get(metric)
        a = after_agg.get(metric)
        print(f"{metric:35s} {fmt(b):>14s} {fmt(a):>14s} {delta(b, a, metric):>24s}")

    print("\nInterpretation")
    print("-" * 92)
    print("For financial accuracy, role appropriateness, personalization, and citation correctness: higher is better.")
    print("For hallucination rate: lower is better.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pre-RAG and post-RAG thesis evaluations.")
    parser.add_argument("--before", required=True, help="Pre-RAG thesis evaluation JSON.")
    parser.add_argument("--after", required=True, help="Post-RAG thesis evaluation JSON.")
    args = parser.parse_args()

    print_table(load_json(Path(args.before)), load_json(Path(args.after)))


if __name__ == "__main__":
    main()
