"""
FINA Dataset / Benchmark TC Distribution Audit
===============================================
Loads the generated dataset (hybrid_data.jsonl) and optionally a benchmark
result file to check family, role, tag, and subfamily coverage.

Usage:
  python audit_tc_distribution.py                         # dataset-only mode
  python audit_tc_distribution.py --benchmark logs/benchmark_*.json  # with benchmark results
  python audit_tc_distribution.py --dataset path/to/data.jsonl
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_dataset(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_benchmark_results(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if "results" in data:
        return data["results"]
    if "cases" in data:
        return data["cases"]
    return []


def infer_tc_family(tc_name: str) -> str:
    """Map benchmark TC name to a family tag for comparison."""
    name = tc_name.lower()
    mapping = [
        ("custom split", "custom_split"),
        ("emergency fund", "emergency_fund"),
        ("no category budget", "no_budget_nudge"),
        ("anomaly — no", "no_anomaly_negative"),
        ("anomaly — proactive", "cross_feature_advice"),
        ("log transaction", "log_expense"),
        ("log —", "log_expense"),
        ("tax", "debt_tax"),
        ("debt", "debt_tax"),
        ("budget health", "budget_health"),
        ("saving tips", "saving_tips"),
        ("overspend", "overspend"),
        ("goal", "goal_progress"),
        ("invest", "investment"),
        ("income", "income_advice"),
        ("edge", "edge_case"),
        ("verdict", "verdict"),
        ("forecast", "forecast"),
        ("balance", "balance"),
        ("mom", "mom_comparison"),
        ("month over month", "mom_comparison"),
        ("recurring", "recurring"),
        ("anomaly", "anomaly"),
        ("category budget", "category_budget"),
    ]
    for pattern, family in mapping:
        if pattern in name:
            return family
    return "other"


def print_section(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("=" * 70)


def audit_dataset(rows: list[dict]):
    total = len(rows)
    print(f"\nDataset: {total} examples")

    # Family counts
    family_counter = Counter(r.get("family", "unknown") for r in rows)
    print_section("FAMILY DISTRIBUTION")
    for fam, cnt in family_counter.most_common():
        pct = cnt / total * 100
        print(f"  {fam:30s} {cnt:6d}  ({pct:5.1f}%)")

    # Role by family
    role_by_family = defaultdict(Counter)
    for r in rows:
        role_by_family[r.get("family", "unknown")][r.get("role", "unknown")] += 1

    print_section("ROLE DISTRIBUTION BY FAMILY")
    for fam in sorted(role_by_family):
        roles = role_by_family[fam]
        role_str = "  ".join(f"{role}:{cnt}" for role, cnt in sorted(roles.items()))
        total_fam = sum(roles.values())
        print(f"  {fam:30s}  [{role_str}]  total={total_fam}")

    # Subfamily counts
    sf_counter = Counter(r.get("subfamily", "") for r in rows if r.get("subfamily"))
    if sf_counter:
        print_section("SUBFAMILY DISTRIBUTION")
        for sf, cnt in sf_counter.most_common():
            print(f"  {sf:35s} {cnt:6d}")

    # Tag counts
    tag_counter = Counter()
    for r in rows:
        for t in r.get("tags", []):
            tag_counter[t] += 1

    print_section("TAG DISTRIBUTION")
    for tag, cnt in tag_counter.most_common():
        pct = cnt / total * 100
        print(f"  {tag:35s} {cnt:6d}  ({pct:5.1f}%)")

    # Role distribution per tag (for weak families)
    weak_tags = [
        "custom_split", "emergency_fund", "no_budget_nudge",
        "no_anomaly_negative", "cross_feature_advice", "debt_tax",
        "ratio_generalization", "hard_negative", "followup",
    ]
    print_section("WEAK-FAMILY TAG COVERAGE (with role breakdown)")
    for tag in weak_tags:
        matching = [r for r in rows if tag in r.get("tags", [])]
        cnt = len(matching)
        pct = cnt / total * 100 if total > 0 else 0
        status = "OK" if cnt >= 50 else "LOW" if cnt >= 20 else "CRITICAL" if cnt > 0 else "MISSING"
        role_dist = Counter(r.get("role", "?") for r in matching)
        role_str = "  ".join(f"{role}:{c}" for role, c in sorted(role_dist.items()))
        print(f"  {tag:30s} {cnt:6d} ({pct:5.1f}%)  {status:8s}  [{role_str}]")

    # Action CRUD domination check
    crud_pct = family_counter.get("action_crud", 0) / total * 100 if total > 0 else 0
    print_section("BALANCE CHECKS")
    if crud_pct > 45:
        print(f"  WARNING: action_crud at {crud_pct:.1f}% — still dominating (target 35-40%)")
    elif crud_pct > 40:
        print(f"  OK: action_crud at {crud_pct:.1f}% — slightly above target, acceptable")
    else:
        print(f"  GOOD: action_crud at {crud_pct:.1f}% — within target range")

    # Check for missing weak families
    for tag in weak_tags:
        if tag_counter.get(tag, 0) == 0:
            print(f"  ALERT: tag '{tag}' has ZERO examples — benchmark coverage gap!")

    return tag_counter, family_counter


def audit_benchmark(benchmark_results: list[dict], dataset_tag_counter: Counter):
    """Compare benchmark TC families against dataset coverage."""
    print_section("BENCHMARK TC FAMILY vs DATASET COVERAGE")

    tc_families = Counter()
    tc_details = defaultdict(list)
    for tc in benchmark_results:
        tc_id = tc.get("id", "?")
        tc_name = tc.get("name", "")
        family = infer_tc_family(tc_name)
        tc_families[family] += 1
        tc_details[family].append(tc_id)

    total_tcs = sum(tc_families.values())
    print(f"\n  Total benchmark TCs: {total_tcs}")
    print(f"  {'TC Family':30s} {'TCs':>5s}  {'TC%':>6s}  {'Dataset':>8s}  {'Status'}")
    print(f"  {'-' * 75}")

    for family, tc_count in tc_families.most_common():
        tc_pct = tc_count / total_tcs * 100 if total_tcs > 0 else 0
        ds_count = dataset_tag_counter.get(family, 0)
        # Also check subfamily or family field
        if ds_count == 0:
            # Try common mappings
            alt_keys = {
                "budget_health": "budget_health",
                "saving_tips": "general_advice",
                "overspend": "over_budget",
                "goal_progress": "goal_progress",
                "log_expense": "log_expense",
            }
            ds_count = dataset_tag_counter.get(alt_keys.get(family, ""), 0)

        ratio = ds_count / max(tc_count, 1)
        if ratio >= 10:
            status = "GOOD"
        elif ratio >= 3:
            status = "OK"
        elif ds_count > 0:
            status = "UNDER-COVERED"
        else:
            status = "MISSING"

        tc_ids = ", ".join(tc_details[family][:5])
        if len(tc_details[family]) > 5:
            tc_ids += "..."
        print(f"  {family:30s} {tc_count:5d}  {tc_pct:5.1f}%  {ds_count:8d}  {status:15s}  [{tc_ids}]")


def audit_benchmark_cases_file():
    """If benchmark_cases.py exists, parse TC IDs/names directly for coverage mapping."""
    cases_path = Path(__file__).with_name("benchmark_cases.py")
    if not cases_path.exists():
        return []

    # Import directly if possible
    try:
        from benchmark_cases import TEST_CASES
        return TEST_CASES
    except Exception:
        pass

    # Fallback: regex parse
    text = cases_path.read_text(encoding="utf-8")
    tcs = []
    for match in re.finditer(r'"id":\s*"(TC\d+)".*?"name":\s*"([^"]+)"', text):
        tcs.append({"id": match.group(1), "name": match.group(2)})
    return tcs


def main():
    parser = argparse.ArgumentParser(description="FINA dataset/benchmark distribution audit")
    parser.add_argument("--dataset", type=str, default="hybrid_data.jsonl",
                        help="Path to JSONL dataset (default: hybrid_data.jsonl)")
    parser.add_argument("--benchmark", type=str, default=None,
                        help="Path to benchmark result JSON (optional)")
    args = parser.parse_args()

    ds_path = Path(args.dataset)
    if not ds_path.exists():
        print(f"Dataset not found: {ds_path}")
        print("Run 'python generate_hybrid.py' first.")
        sys.exit(1)

    print(f"Loading dataset: {ds_path}")
    rows = load_dataset(ds_path)
    tag_counter, family_counter = audit_dataset(rows)

    # Try benchmark results if provided
    if args.benchmark:
        bm_path = Path(args.benchmark)
        if bm_path.exists():
            print(f"\nLoading benchmark results: {bm_path}")
            bm_results = load_benchmark_results(bm_path)
            audit_benchmark(bm_results, tag_counter)
        else:
            print(f"\nBenchmark file not found: {bm_path}")

    # Always try to parse benchmark_cases.py for TC coverage
    tcs = audit_benchmark_cases_file()
    if tcs:
        print(f"\nFound {len(tcs)} TCs in benchmark_cases.py")
        audit_benchmark(tcs, tag_counter)

    print("\n" + "=" * 70)
    print("  AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
