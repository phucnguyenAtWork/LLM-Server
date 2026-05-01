"""
Compare Model Benchmark Logs
============================

Builds a thesis-friendly table from multiple `benchmark.py` JSON logs.
Use it for close-model comparisons such as v6 vs v7 vs v8, and for
base/fine-tuned/RAG comparisons.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = [
    ("overall_accuracy_pct", "Overall accuracy"),
    ("financial_accuracy_pct", "Financial accuracy"),
    ("role_appropriateness_pct", "Role appropriateness"),
    ("personalization_quality_pct", "Personalization"),
    ("hallucination_rate_pct", "Hallucination rate"),
    ("citation_correctness_pct", "Citation correctness"),
]


def load_log(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    meta = payload.get("meta", {})
    aggregate = payload.get("aggregate", {})
    model_metrics = aggregate.get("model_evaluation_metrics", aggregate)

    row = {
        "path": str(path),
        "label": meta.get("label") or meta.get("adapter") or path.stem,
        "base_model": meta.get("base_model"),
        "adapter": meta.get("adapter"),
        "phase": meta.get("phase", "unknown"),
        "case_count": meta.get("case_count") or len(payload.get("per_test", [])),
        "overall_accuracy_pct": aggregate.get("overall_accuracy_pct"),
    }
    row.update(model_metrics)
    return row


def fmt(value) -> str:
    return "N/A" if value is None else f"{float(value):.2f}%"


def print_table(rows: list[dict]) -> None:
    print("\nModel Benchmark Comparison")
    print("=" * 124)
    print(
        f"{'Model':28s} {'Phase':10s} {'Cases':>6s} "
        f"{'Overall':>10s} {'Finance':>10s} {'Role':>10s} "
        f"{'Personal':>10s} {'Halluc.':>10s} {'Citation':>10s}"
    )
    print("-" * 124)
    for row in rows:
        print(
            f"{row['label'][:28]:28s} {row['phase'][:10]:10s} {row['case_count']:6d} "
            f"{fmt(row.get('overall_accuracy_pct')):>10s} "
            f"{fmt(row.get('financial_accuracy_pct')):>10s} "
            f"{fmt(row.get('role_appropriateness_pct')):>10s} "
            f"{fmt(row.get('personalization_quality_pct')):>10s} "
            f"{fmt(row.get('hallucination_rate_pct')):>10s} "
            f"{fmt(row.get('citation_correctness_pct')):>10s}"
        )


def write_csv(rows: list[dict], path: Path) -> None:
    headers = [
        "label",
        "phase",
        "case_count",
        "base_model",
        "adapter",
        "overall_accuracy_pct",
        "financial_accuracy_pct",
        "role_appropriateness_pct",
        "personalization_quality_pct",
        "hallucination_rate_pct",
        "citation_correctness_pct",
        "path",
    ]
    lines = [",".join(headers)]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header)
            text = "" if value is None else str(value)
            values.append('"' + text.replace('"', '""') + '"')
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple FINA benchmark JSON files.")
    parser.add_argument("logs", nargs="+", help="Benchmark JSON files.")
    parser.add_argument("--csv", default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    rows = [load_log(Path(log)) for log in args.logs]
    print_table(rows)
    if args.csv:
        output = Path(args.csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_csv(rows, output)
        print(f"\nSaved CSV to {output}")


if __name__ == "__main__":
    main()
