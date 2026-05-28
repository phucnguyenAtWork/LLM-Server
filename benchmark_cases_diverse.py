"""
Diverse Generalization Slice — 18 curated cases (§5.3 of the thesis).

6 cases per role × 3 difficulty bands. All IDs are pulled from the existing
benchmark_cases.TEST_CASES so no new ground truth is introduced. The slice is
the headline generalization/stability table; the full 141 stays as the
comprehensive run.

Difficulty bands:
  - easy:  single-step calculation (totals, surplus, simple advice)
  - multi: multi-step (goals, tax with deductions, custom budget splits)
  - hard:  edge cases (deficit, missing budgets, anomalies, very low income)
"""

from __future__ import annotations

from benchmark_cases import TEST_CASES

DIVERSE_IDS_BY_BAND: dict[str, dict[str, list[str]]] = {
    "Student": {
        "easy":  ["TC02", "TC09"],
        "multi": ["TC14", "TC72"],
        "hard":  ["TC03", "TC97"],
    },
    "Worker": {
        "easy":  ["TC04", "TC05"],
        "multi": ["TC74", "TC92"],
        "hard":  ["TC117", "TC202"],
    },
    "Freelancer": {
        "easy":  ["TC07", "TC25"],
        "multi": ["TC23", "TC26"],
        "hard":  ["TC44", "TC319"],
    },
}

DIFFICULTY_BY_ID: dict[str, str] = {
    case_id: band
    for role_bands in DIVERSE_IDS_BY_BAND.values()
    for band, ids in role_bands.items()
    for case_id in ids
}

DIVERSE_IDS: list[str] = list(DIFFICULTY_BY_ID.keys())


def _build_diverse_cases() -> list[dict]:
    by_id = {case["id"]: case for case in TEST_CASES}
    missing = [cid for cid in DIVERSE_IDS if cid not in by_id]
    if missing:
        raise RuntimeError(
            "benchmark_cases_diverse.py references IDs not present in TEST_CASES: "
            + ", ".join(missing)
        )
    return [by_id[cid] for cid in DIVERSE_IDS]


DIVERSE_CASES: list[dict] = _build_diverse_cases()


def summary() -> str:
    lines = [f"Diverse generalization slice: {len(DIVERSE_CASES)} cases"]
    for role, bands in DIVERSE_IDS_BY_BAND.items():
        for band, ids in bands.items():
            lines.append(f"  {role:12s} {band:6s} {ids}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
