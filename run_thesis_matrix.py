"""
Thesis Matrix Orchestrator (§5.4 + §5.5)
=========================================

Runs the model × retrieval × decoding-mode grid and consolidates results
into a single CSV/JSON the thesis can quote directly.

Headline 4-cell view (one variable isolated at a time):
  B1: untuned base + no RAG       — raw base capability
  B2: untuned base + vector RAG   — RAG gain on the base
  M1: v8 LoRA  + no RAG           — fine-tuning gain alone
  M2: v8 LoRA  + vector RAG       — deployment configuration

Full grid extends to {baseline, v6, v7, v8} × {none, oracle, vector}.

Stability is two separate experiments, never merged:
  4a. deterministic (greedy, same seed)        → reproducibility (boolean per cell)
  4b. sampling     (temperature + N seeds)     → mean ± std per metric

Each cell shells out to benchmark.py and then thesis_evaluation.py and
collects the resulting JSON. Wall-clock budget is dominated by inference;
prefer --cases diverse for fast iteration.

Usage examples:
  python run_thesis_matrix.py --mode deterministic --cells headline --cases diverse
  python run_thesis_matrix.py --mode sampling --seeds 0 1 2 --cases diverse
  python run_thesis_matrix.py --mode deterministic --cells full --cases full
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LOGS_DIR = PROJECT_ROOT / "logs"
PYTHON = sys.executable

HEADLINE_CELLS = [
    {"name": "B1_baseline_no_rag", "adapter": "none", "rag_backend": "none",   "cite_sources": False},
    {"name": "B2_baseline_vector", "adapter": "none", "rag_backend": "vector", "cite_sources": True},
    {"name": "M1_v8_no_rag",       "adapter": "financial_qwen_native_v8", "rag_backend": "none",   "cite_sources": False},
    {"name": "M2_v8_vector",       "adapter": "financial_qwen_native_v8", "rag_backend": "vector", "cite_sources": True},
]

FULL_ADAPTERS = ["none", "financial_qwen_native_v6", "financial_qwen_native_v7", "financial_qwen_native_v8"]
FULL_BACKENDS = [
    {"name": "none",      "rag_backend": "none",      "cite_sources": False},
    {"name": "oracle",    "rag_backend": "synthetic", "cite_sources": True},
    {"name": "vector",    "rag_backend": "vector",    "cite_sources": True},
]


def build_full_cells() -> list[dict]:
    cells = []
    for adapter in FULL_ADAPTERS:
        adapter_label = "baseline" if adapter == "none" else adapter.replace("financial_qwen_native_", "")
        for backend in FULL_BACKENDS:
            cells.append({
                "name": f"{adapter_label}_{backend['name']}",
                "adapter": adapter,
                "rag_backend": backend["rag_backend"],
                "cite_sources": backend["cite_sources"],
            })
    return cells


@dataclass
class CellResult:
    cell: str
    adapter: str
    rag_backend: str
    seed: int | None
    benchmark_log: Path
    thesis_log: Path
    aggregate: dict = field(default_factory=dict)
    pass_rates: dict = field(default_factory=dict)
    role_classification: dict = field(default_factory=dict)
    retrieval_quality: dict | None = None
    json_compliance_pct: float | None = None
    overall_accuracy_pct: float | None = None


def run_benchmark_cell(cell: dict, *, cases: str, mode: str, seed: int, temperature: float, label: str) -> Path:
    cmd = [
        PYTHON, str(PROJECT_ROOT / "benchmark.py"),
        "--adapter", cell["adapter"],
        "--rag-backend", cell["rag_backend"],
        "--cases", cases,
        "--mode", mode,
        "--seed", str(seed),
        "--temperature", str(temperature),
        "--phase", "post_rag" if cell["cite_sources"] else "pre_rag",
        "--label", label,
    ]
    if cell.get("adapter") == "none":
        cmd.extend(["--base-model", "Qwen/Qwen2.5-3B-Instruct"])
    if cell["cite_sources"]:
        cmd.append("--cite-sources")
    print(f"\n>>> {' '.join(cmd)}")
    before = set(LOGS_DIR.glob("benchmark_*.json"))
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"benchmark.py failed for cell {cell['name']}")
    after = set(LOGS_DIR.glob("benchmark_*.json"))
    new = sorted(after - before, key=lambda p: p.stat().st_mtime)
    if not new:
        raise RuntimeError(f"benchmark.py produced no log for cell {cell['name']}")
    return new[-1]


def run_thesis_eval(benchmark_log: Path, phase: str) -> Path:
    output = LOGS_DIR / f"thesis_eval_{benchmark_log.stem}.json"
    cmd = [
        PYTHON, str(PROJECT_ROOT / "thesis_evaluation.py"),
        "--input", str(benchmark_log),
        "--phase", phase,
        "--output", str(output),
    ]
    print(f">>> {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"thesis_evaluation.py failed for {benchmark_log}")
    return output


def load_cell_result(cell: dict, seed: int | None, benchmark_log: Path, thesis_log: Path) -> CellResult:
    bench = json.loads(benchmark_log.read_text(encoding="utf-8"))
    thesis = json.loads(thesis_log.read_text(encoding="utf-8"))
    bench_aggregate = bench.get("aggregate", {})
    return CellResult(
        cell=cell["name"],
        adapter=cell["adapter"],
        rag_backend=cell["rag_backend"],
        seed=seed,
        benchmark_log=benchmark_log,
        thesis_log=thesis_log,
        aggregate=thesis.get("aggregate", {}),
        pass_rates=thesis.get("pass_rates", {}).get("overall", {}),
        role_classification=thesis.get("role_classification", {}),
        retrieval_quality=thesis.get("retrieval_quality"),
        json_compliance_pct=bench_aggregate.get("json_compliance_pct"),
        overall_accuracy_pct=bench_aggregate.get("overall_accuracy_pct"),
    )


def deterministic_reproducibility(cells: list[dict], cases: str) -> list[dict]:
    """4a — run each cell twice, compare benchmark_log responses byte-for-byte."""
    rows = []
    for cell in cells:
        log_a = run_benchmark_cell(cell, cases=cases, mode="deterministic", seed=0, temperature=0.0, label=f"{cell['name']}_repA")
        log_b = run_benchmark_cell(cell, cases=cases, mode="deterministic", seed=0, temperature=0.0, label=f"{cell['name']}_repB")
        responses_a = [r.get("response") for r in json.loads(log_a.read_text(encoding="utf-8")).get("per_test", [])]
        responses_b = [r.get("response") for r in json.loads(log_b.read_text(encoding="utf-8")).get("per_test", [])]
        identical = responses_a == responses_b
        diff_count = sum(1 for a, b in zip(responses_a, responses_b) if a != b)
        rows.append({
            "cell": cell["name"],
            "reproducible": identical,
            "diff_count": diff_count,
            "n": min(len(responses_a), len(responses_b)),
            "log_a": str(log_a),
            "log_b": str(log_b),
        })
        print(f"[reproducibility] {cell['name']:30s} identical={identical} diffs={diff_count}/{len(responses_a)}")
    return rows


def sampling_robustness(cells: list[dict], cases: str, seeds: list[int], temperature: float) -> list[CellResult]:
    """4b — N seeds per cell, report mean ± std per metric (computed downstream)."""
    results: list[CellResult] = []
    for cell in cells:
        phase = "post_rag" if cell["cite_sources"] else "pre_rag"
        for seed in seeds:
            bench_log = run_benchmark_cell(
                cell, cases=cases, mode="sampling", seed=seed,
                temperature=temperature, label=f"{cell['name']}_seed{seed}",
            )
            thesis_log = run_thesis_eval(bench_log, phase)
            results.append(load_cell_result(cell, seed, bench_log, thesis_log))
    return results


def deterministic_grid(cells: list[dict], cases: str) -> list[CellResult]:
    """One deterministic run per cell — used for the headline matrix tables."""
    results: list[CellResult] = []
    for cell in cells:
        phase = "post_rag" if cell["cite_sources"] else "pre_rag"
        bench_log = run_benchmark_cell(
            cell, cases=cases, mode="deterministic", seed=0, temperature=0.0, label=cell["name"],
        )
        thesis_log = run_thesis_eval(bench_log, phase)
        results.append(load_cell_result(cell, None, bench_log, thesis_log))
    return results


METRIC_KEYS = [
    "financial_accuracy_pct",
    "role_appropriateness_pct",
    "personalization_quality_pct",
    "hallucination_rate_pct",
    "citation_correctness_pct",
]


def aggregate_seeds(results: list[CellResult]) -> list[dict]:
    by_cell: dict[str, list[CellResult]] = {}
    for row in results:
        by_cell.setdefault(row.cell, []).append(row)
    rows = []
    for cell_name, group in by_cell.items():
        record = {
            "cell": cell_name,
            "adapter": group[0].adapter,
            "rag_backend": group[0].rag_backend,
            "seeds": [r.seed for r in group],
            "n_seeds": len(group),
            "macro_f1_mean": _mean_std([r.role_classification.get("macro_f1") for r in group])[0],
            "macro_f1_std":  _mean_std([r.role_classification.get("macro_f1") for r in group])[1],
            "json_compliance_pct_mean": _mean_std([r.json_compliance_pct for r in group])[0],
            "overall_accuracy_pct_mean": _mean_std([r.overall_accuracy_pct for r in group])[0],
        }
        for key in METRIC_KEYS:
            mean, std = _mean_std([r.aggregate.get(key) for r in group])
            record[f"{key}_mean"] = mean
            record[f"{key}_std"] = std
        rows.append(record)
    return rows


def _mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    clean = [float(v) for v in values if isinstance(v, (int, float))]
    if not clean:
        return None, None
    if len(clean) == 1:
        return round(clean[0], 4), 0.0
    return round(statistics.mean(clean), 4), round(statistics.stdev(clean), 4)


def write_matrix_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    headers = sorted({key for row in rows for key in row})
    headers = ["cell", "adapter", "rag_backend"] + [h for h in headers if h not in {"cell", "adapter", "rag_backend"}]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in headers})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the thesis evaluation matrix.")
    parser.add_argument("--cells", choices=["headline", "full"], default="headline")
    parser.add_argument("--cases", choices=["full", "diverse"], default="diverse")
    parser.add_argument("--mode", choices=["deterministic", "sampling", "reproducibility"], default="deterministic")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--out", default=None, help="Output CSV path (auto-named if omitted).")
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    cells = HEADLINE_CELLS if args.cells == "headline" else build_full_cells()
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    if args.mode == "reproducibility":
        rows = deterministic_reproducibility(cells, args.cases)
        out_path = Path(args.out) if args.out else LOGS_DIR / f"thesis_matrix_reproducibility_{ts}.csv"
        write_matrix_csv(rows, out_path)
    elif args.mode == "deterministic":
        results = deterministic_grid(cells, args.cases)
        rows = aggregate_seeds(results)
        out_path = Path(args.out) if args.out else LOGS_DIR / f"thesis_matrix_deterministic_{ts}.csv"
        write_matrix_csv(rows, out_path)
    else:
        results = sampling_robustness(cells, args.cases, args.seeds, args.temperature)
        rows = aggregate_seeds(results)
        out_path = Path(args.out) if args.out else LOGS_DIR / f"thesis_matrix_sampling_{ts}.csv"
        write_matrix_csv(rows, out_path)

    print(f"\nMatrix saved to {out_path}")
    print(f"Rows: {len(rows)} | Mode: {args.mode} | Cases: {args.cases} | Cells: {args.cells}")


if __name__ == "__main__":
    main()
