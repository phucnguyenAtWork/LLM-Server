"""
Thesis Evaluation Metrics
=========================

Scores benchmark outputs for thesis-facing model improvement metrics:

1. Financial accuracy
2. Role appropriateness
3. Personalization quality
4. Hallucination rate
5. Citation correctness

Use this for both phases:
- pre_rag: current model output before retrieval evidence is added.
- post_rag: model output after retrieval evidence is added.

Existing `benchmark.py` logs are supported. Future RAG benchmark logs may add
`retrieved_sources` per test item:

{
  "id": "TC01",
  "response": "{\"kind\":\"analysis\",\"message\":\"... [S1]\"}",
  "retrieved_sources": [{"id": "S1", "text": "Spent 1.500.000 VND on Food."}]
}
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from benchmark_cases import CALC, TEST_CASES
from fina_schema import parse_model_output

PROJECT_ROOT = Path(__file__).resolve().parent
LOGS_DIR = PROJECT_ROOT / "logs"

# Threshold pass-rate definitions (§5.1 of the thesis).
# Expected labels in this benchmark are always PASS, so this is a pass-rate
# and failure-mode breakdown — NOT a balanced classification confusion matrix.
PASS_THRESHOLDS = {
    "financial_accuracy": ("ge", 0.8),
    "role_appropriateness": ("ge", 0.7),
    "personalization_quality": ("ge", 0.6),
    "hallucination_rate": ("le", 0.1),
}

ROLES = ("Student", "Worker", "Freelancer")

ROLE_TERMS = {
    "Student": {
        "positive": [
            "student",
            "education",
            "tuition",
            "semester",
            "part-time",
            "affordability",
            "debt avoidance",
        ],
        "negative": ["retirement", "business", "invoice", "quarterly tax"],
    },
    "Worker": {
        "positive": [
            "salary",
            "worker",
            "emergency fund",
            "retirement",
            "invest",
            "bhxh",
            "paycheck",
        ],
        "negative": ["tuition", "invoice", "freelance client"],
    },
    "Freelancer": {
        "positive": [
            "freelancer",
            "irregular",
            "variable income",
            "income buffer",
            "tax reserve",
            "invoice",
            "business",
            "quarterly",
        ],
        "negative": ["salary", "semester", "tuition"],
    },
}

KNOWN_CATEGORIES = {
    "Food",
    "Transport",
    "Shopping",
    "Entertainment",
    "Bills",
    "Health",
    "Education",
    "Equipment",
    "Software",
}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    raw = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    parsed = parse_model_output(raw)
    return parsed.message if parsed else raw


def normalize_number_text(text: str) -> str:
    return re.sub(r"[,.]", "", text)


def amount_variants(amount: float | int) -> set[str]:
    n = int(round(float(amount)))
    return {str(n), f"{n:,}", f"{n:,}".replace(",", ".")}


def extract_amounts(text: str) -> list[int]:
    amounts = []
    pattern = r"(?<!\d)(\d{1,3}(?:[,.]\d{3})+|\d{5,})(?!\d)"
    for match in re.finditer(pattern, text):
        try:
            amounts.append(int(normalize_number_text(match.group(1))))
        except ValueError:
            pass
    return amounts


def extract_citations(text: str) -> list[str]:
    return re.findall(r"\[([A-Z]\d+)\]", text)


def case_by_id() -> dict[str, dict[str, Any]]:
    return {case["id"]: case for case in TEST_CASES}


def expected_amounts(case: dict[str, Any]) -> set[int]:
    income = int(case.get("income", 0))
    spending = case.get("spending", {})
    total_spent = int(sum(spending.values()))
    surplus = income - total_spent

    amounts = {income, total_spent, abs(surplus)}
    amounts.update(int(v) for v in spending.values())
    amounts.update(int(round(income * ratio)) for ratio in (0.2, 0.3, 0.5, 0.8))

    for goal in case.get("goals", []) or []:
        target = int(goal.get("target_amount", 0))
        saved = int(goal.get("current_saved", 0))
        amounts.update({target, saved, max(target - saved, 0)})

    for balance in case.get("balances", []) or []:
        amounts.add(int(balance.get("balance", 0)))

    for budget in case.get("category_budgets", []) or []:
        amounts.add(int(budget.get("monthlyLimit", 0)))

    for recurring in case.get("recurring", []) or []:
        amounts.add(int(recurring.get("amount", 0)))

    forecast = case.get("forecast")
    if forecast:
        amounts.add(int(forecast.get("total", 0)))

    return {amount for amount in amounts if amount > 0}


def score_financial_accuracy(log_item: dict[str, Any], case: dict[str, Any]) -> float | None:
    checks = log_item.get("checks")
    if not isinstance(checks, dict):
        return None

    calc_names = [
        name
        for name, check in case.get("checks", {}).items()
        if check.get("cat") == CALC
    ]
    if not calc_names:
        return None

    passed = sum(1 for name in calc_names if checks.get(name) is True)
    return passed / len(calc_names)


def score_role_appropriateness(message: str, role: str) -> float:
    role_config = ROLE_TERMS.get(role, {"positive": [], "negative": []})
    text = message.lower()
    positives = sum(1 for term in role_config["positive"] if term in text)
    negatives = sum(1 for term in role_config["negative"] if term in text)

    # Baseline credit: an answer can be role-appropriate without literally
    # saying "student", "worker", or "freelancer" every time. Penalize clear
    # wrong-role language, and add credit for role-specific wording when present.
    base_score = 0.6
    role_signal = min(positives / 2, 0.4)
    role_name_bonus = 0.15 if role.lower() in text else 0.0
    penalty = min(negatives * 0.3, 0.8)
    return max(0.0, min(1.0, round(base_score + role_signal + role_name_bonus - penalty, 4)))


def score_personalization(message: str, case: dict[str, Any]) -> float:
    text = message.lower()
    spending = case.get("spending", {})
    categories = [cat for cat in spending if cat.lower() in text]
    amounts = expected_amounts(case)
    mentioned_amounts = [
        amount for amount in amounts if any(variant in message for variant in amount_variants(amount))
    ]

    goal_names = [
        str(goal.get("name", "")).lower()
        for goal in case.get("goals", []) or []
        if goal.get("name")
    ]
    mentioned_goals = [goal for goal in goal_names if goal and goal in text]
    extracted = extract_amounts(message)

    numeric_grounding = min(len(extracted) / 2, 1.0)
    category_grounding = min(len(categories) / 2, 1.0)
    exact_amount_grounding = min(len(mentioned_amounts) / 2, 1.0)
    goal_grounding = 1.0 if mentioned_goals else (0.0 if goal_names else 0.75)
    actionable_grounding = (
        1.0
        if any(
            word in text
            for word in [
                "cut",
                "reduce",
                "save",
                "set aside",
                "invest",
                "buffer",
                "limit",
                "transfer",
                "account",
                "check",
                "track",
                "review",
            ]
        )
        else 0.0
    )

    components = [
        category_grounding,
        max(exact_amount_grounding, numeric_grounding),
        goal_grounding,
        actionable_grounding,
    ]
    return round(mean(components), 4)


def score_hallucination(message: str, case: dict[str, Any]) -> dict[str, Any]:
    text = message.lower()
    spending = case.get("spending", {})
    allowed_categories = {cat.lower() for cat in spending}

    mentioned_known_categories = {
        cat.lower()
        for cat in KNOWN_CATEGORIES
        if cat.lower() in text
    }
    unsupported_categories = sorted(mentioned_known_categories - allowed_categories)

    allowed_amounts = expected_amounts(case)
    income = int(case.get("income", 0))
    largest_expected = max([income, *allowed_amounts], default=0)
    unsupported_amounts = []
    extracted_amounts = extract_amounts(message)
    for amount in extracted_amounts:
        if amount in allowed_amounts:
            continue
        # Advice may propose new target amounts. Flag only values that are clearly
        # outside the user's financial scale as an automated hallucination proxy.
        if largest_expected and amount > largest_expected * 2:
            unsupported_amounts.append(amount)

    unsupported = len(unsupported_categories) + len(unsupported_amounts)
    checked = len(mentioned_known_categories) + len(extracted_amounts)
    rate = unsupported / checked if checked else 0.0

    return {
        "rate": round(rate, 4),
        "unsupported_categories": unsupported_categories,
        "unsupported_amounts": unsupported_amounts,
        "unsupported_claims": unsupported,
        "checked_claims": checked,
    }


def score_citation_correctness(
    message: str,
    log_item: dict[str, Any],
    phase: str,
) -> float | None:
    sources = log_item.get("retrieved_sources") or log_item.get("sources") or []
    source_ids = {
        str(source.get("id"))
        for source in sources
        if isinstance(source, dict) and source.get("id")
    }

    if not source_ids:
        return None if phase == "pre_rag" else 0.0

    citations = extract_citations(message)
    if not citations:
        return 0.0

    valid = sum(1 for citation in citations if citation in source_ids)
    return round(valid / len(citations), 4)


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if "per_test" in data:
        return data["per_test"]
    if "results" in data:
        return data["results"]
    if "cases" in data:
        return data["cases"]
    raise ValueError(f"Unsupported benchmark format: {path}")


def aggregate_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    if not values:
        return None
    return round(mean(values) * 100, 2)


# ── §1a. Threshold pass-rates / failure-mode breakdown ─────────────────────

def _passes(metric: str, value: float | None) -> bool | None:
    if value is None:
        return None
    op, threshold = PASS_THRESHOLDS[metric]
    return value >= threshold if op == "ge" else value <= threshold


def _first_failure(row: dict[str, Any]) -> str | None:
    """Return the first metric that tripped its threshold, or 'schema_invalid', or None."""
    if row.get("json_ok") is False:
        return "schema_invalid"
    for metric in PASS_THRESHOLDS:
        verdict = _passes(metric, row.get(metric))
        if verdict is False:
            return metric
    return None


def _failure_explanation(metric: str | None, row: dict[str, Any]) -> str:
    if metric is None:
        return "all thresholds met"
    if metric == "schema_invalid":
        return "model output failed JSON schema parse"
    value = row.get(metric)
    op, threshold = PASS_THRESHOLDS[metric]
    direction = ">=" if op == "ge" else "<="
    return f"{metric}={value} (need {direction} {threshold})"


def compute_pass_rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    overall: dict[str, Any] = {}
    failure_modes: dict[str, int] = defaultdict(int)
    for metric in PASS_THRESHOLDS:
        verdicts = [_passes(metric, row.get(metric)) for row in rows]
        scored = [v for v in verdicts if v is not None]
        overall[metric] = {
            "n": len(scored),
            "pass": sum(1 for v in scored if v),
            "pass_rate_pct": round(sum(1 for v in scored if v) / len(scored) * 100, 2) if scored else None,
            "threshold": PASS_THRESHOLDS[metric],
        }

    json_rows = [row for row in rows if row.get("json_ok") is not None]
    overall["json_schema"] = {
        "n": len(json_rows),
        "pass": sum(1 for row in json_rows if row.get("json_ok")),
        "pass_rate_pct": round(
            sum(1 for row in json_rows if row.get("json_ok")) / len(json_rows) * 100, 2
        ) if json_rows else None,
    }

    for row in rows:
        mode = _first_failure(row)
        if mode is not None:
            failure_modes[mode] += 1

    by_role: dict[str, dict[str, Any]] = {}
    for role in ROLES:
        role_rows = [r for r in rows if r.get("role") == role]
        if not role_rows:
            continue
        by_role[role] = {
            "count": len(role_rows),
            "all_pass": sum(1 for r in role_rows if _first_failure(r) is None),
            "all_pass_pct": round(
                sum(1 for r in role_rows if _first_failure(r) is None) / len(role_rows) * 100, 2
            ),
        }
        for metric in PASS_THRESHOLDS:
            verdicts = [_passes(metric, r.get(metric)) for r in role_rows]
            scored = [v for v in verdicts if v is not None]
            by_role[role][f"{metric}_pass_pct"] = (
                round(sum(1 for v in scored if v) / len(scored) * 100, 2) if scored else None
            )

    return {
        "overall": overall,
        "failure_modes": dict(failure_modes),
        "by_role": by_role,
    }


# ── §1b. True 3×3 role classification (the only legitimate F1 in this thesis) ─

def infer_target_role(message: str) -> str | None:
    """Score the response against each role's vocabulary; return argmax role."""
    text = message.lower()
    if not text.strip():
        return None
    scores: dict[str, float] = {}
    for role in ROLES:
        cfg = ROLE_TERMS[role]
        positives = sum(1 for term in cfg["positive"] if term in text)
        negatives = sum(1 for term in cfg["negative"] if term in text)
        bonus = 1.0 if role.lower() in text else 0.0
        scores[role] = positives + bonus - 0.5 * negatives
    best = max(scores.values())
    if best <= 0:
        return None
    winners = [role for role, score in scores.items() if score == best]
    return winners[0] if len(winners) == 1 else None


def compute_role_classification(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cm: dict[str, dict[str, int]] = {true: {pred: 0 for pred in (*ROLES, "unknown")} for true in ROLES}
    for row in rows:
        true_role = row.get("role")
        if true_role not in ROLES:
            continue
        pred = row.get("predicted_role") or "unknown"
        cm[true_role][pred] = cm[true_role].get(pred, 0) + 1

    per_role: dict[str, dict[str, float | None]] = {}
    f1_values: list[float] = []
    for role in ROLES:
        tp = cm[role][role]
        fp = sum(cm[other][role] for other in ROLES if other != role)
        fn = sum(cm[role][pred] for pred in cm[role] if pred != role)
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        if precision and recall:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0 if (tp + fp + fn) else None
        per_role[role] = {
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "support": tp + fn,
        }
        if f1 is not None:
            f1_values.append(f1)

    total = sum(cm[true][pred] for true in ROLES for pred in cm[true])
    correct = sum(cm[role][role] for role in ROLES)
    accuracy = round(correct / total, 4) if total else None
    macro_f1 = round(mean(f1_values), 4) if f1_values else None

    return {
        "confusion_matrix": cm,
        "per_role": per_role,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "support": total,
    }


# ── §7. Retrieval-quality metrics (Precision@k, Recall@k, MRR) ──────────────

def compute_retrieval_quality(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Score retrieved IDs against oracle source IDs when both are present."""
    scored = []
    for row in rows:
        retrieved = row.get("retrieved_ids") or []
        oracle = set(row.get("oracle_ids") or [])
        if not retrieved or not oracle:
            continue
        retrieved_set = set(retrieved)
        precision = len(retrieved_set & oracle) / len(retrieved_set) if retrieved_set else 0.0
        recall = len(retrieved_set & oracle) / len(oracle) if oracle else 0.0
        rr = 0.0
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in oracle:
                rr = 1.0 / rank
                break
        scored.append({"precision": precision, "recall": recall, "rr": rr})

    if not scored:
        return None
    return {
        "n": len(scored),
        "precision_at_k": round(mean(s["precision"] for s in scored), 4),
        "recall_at_k": round(mean(s["recall"] for s in scored), 4),
        "mrr": round(mean(s["rr"] for s in scored), 4),
    }


# ── §8. Report-ready case analysis CSVs ─────────────────────────────────────

CASE_CSV_COLUMNS = [
    "id", "role", "difficulty", "user_input", "expected_output", "model_output",
    "financial_accuracy", "role_appropriateness", "personalization_quality",
    "hallucination_rate", "citation_correctness", "json_ok",
    "failure_type", "explanation",
]


def _expected_summary(case: dict[str, Any]) -> str:
    parts = []
    income = case.get("income")
    spending = case.get("spending") or {}
    if income:
        parts.append(f"income={income}")
    if spending:
        total = sum(spending.values())
        parts.append(f"total_spent={total}")
        parts.append(f"surplus={income - total}")
    checks = case.get("checks") or {}
    if checks:
        parts.append("checks=" + "|".join(checks.keys()))
    return "; ".join(parts)


def _difficulty_for(case_id: str) -> str:
    """Look up difficulty if a diverse-slice mapping exists; else 'standard'."""
    try:
        from benchmark_cases_diverse import DIFFICULTY_BY_ID
    except ImportError:
        return "standard"
    return DIFFICULTY_BY_ID.get(case_id, "standard")


def export_case_csvs(
    rows: list[dict[str, Any]],
    cases_by_id: dict[str, dict[str, Any]],
    output_dir: Path,
    stem: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    easy_path = output_dir / f"{stem}_easy_success_cases.csv"
    hard_path = output_dir / f"{stem}_hard_success_cases.csv"
    fail_path = output_dir / f"{stem}_failure_cases.csv"

    easy_writer = csv.DictWriter(easy_path.open("w", encoding="utf-8", newline=""), fieldnames=CASE_CSV_COLUMNS)
    hard_writer = csv.DictWriter(hard_path.open("w", encoding="utf-8", newline=""), fieldnames=CASE_CSV_COLUMNS)
    fail_writer = csv.DictWriter(fail_path.open("w", encoding="utf-8", newline=""), fieldnames=CASE_CSV_COLUMNS)
    for writer in (easy_writer, hard_writer, fail_writer):
        writer.writeheader()

    for row in rows:
        case = cases_by_id.get(row["id"], {})
        difficulty = _difficulty_for(row["id"])
        failure = _first_failure(row)
        record = {
            "id": row["id"],
            "role": row.get("role", ""),
            "difficulty": difficulty,
            "user_input": case.get("question", ""),
            "expected_output": _expected_summary(case),
            "model_output": (row.get("response_text") or "")[:1000],
            "financial_accuracy": row.get("financial_accuracy"),
            "role_appropriateness": row.get("role_appropriateness"),
            "personalization_quality": row.get("personalization_quality"),
            "hallucination_rate": row.get("hallucination_rate"),
            "citation_correctness": row.get("citation_correctness"),
            "json_ok": row.get("json_ok"),
            "failure_type": failure or "",
            "explanation": _failure_explanation(failure, row),
        }
        if failure is not None:
            fail_writer.writerow(record)
        elif difficulty == "hard" or row.get("financial_accuracy") == 1.0:
            (hard_writer if difficulty == "hard" else easy_writer).writerow(record)

    return {"easy": easy_path, "hard": hard_path, "failure": fail_path}


def evaluate(path: Path, phase: str) -> dict[str, Any]:
    cases = case_by_id()
    rows = []

    for item in load_benchmark(path):
        case = cases.get(item.get("id"))
        if not case:
            continue

        raw_response = item.get("response", "")
        message = normalize_text(raw_response)
        parsed = parse_model_output(raw_response, log_failure=False) if isinstance(raw_response, str) else None
        json_ok = item.get("json_compliant")
        if json_ok is None:
            json_ok = parsed is not None
        hallucination = score_hallucination(message, case)
        rows.append(
            {
                "id": item.get("id"),
                "name": item.get("name") or case.get("name"),
                "role": item.get("role") or case.get("role"),
                "predicted_role": infer_target_role(message),
                "financial_accuracy": score_financial_accuracy(item, case),
                "role_appropriateness": score_role_appropriateness(message, case.get("role", "")),
                "personalization_quality": score_personalization(message, case),
                "hallucination_rate": hallucination["rate"],
                "citation_correctness": score_citation_correctness(message, item, phase),
                "hallucination_details": hallucination,
                "json_ok": bool(json_ok),
                "response_text": message,
                "retrieved_ids": [
                    str(s.get("id")) for s in (item.get("retrieved_sources") or [])
                    if isinstance(s, dict) and s.get("id")
                ],
                "oracle_ids": item.get("oracle_ids") or [],
            }
        )

    role_breakdown = defaultdict(list)
    for row in rows:
        role_breakdown[row["role"]].append(row)

    aggregate = {
        "financial_accuracy_pct": aggregate_metric(rows, "financial_accuracy"),
        "role_appropriateness_pct": aggregate_metric(rows, "role_appropriateness"),
        "personalization_quality_pct": aggregate_metric(rows, "personalization_quality"),
        "hallucination_rate_pct": aggregate_metric(rows, "hallucination_rate"),
        "citation_correctness_pct": aggregate_metric(rows, "citation_correctness"),
    }

    by_role = {}
    for role, role_rows in sorted(role_breakdown.items()):
        by_role[role] = {
            "count": len(role_rows),
            "financial_accuracy_pct": aggregate_metric(role_rows, "financial_accuracy"),
            "role_appropriateness_pct": aggregate_metric(role_rows, "role_appropriateness"),
            "personalization_quality_pct": aggregate_metric(role_rows, "personalization_quality"),
            "hallucination_rate_pct": aggregate_metric(role_rows, "hallucination_rate"),
            "citation_correctness_pct": aggregate_metric(role_rows, "citation_correctness"),
        }

    pass_rates = compute_pass_rates(rows)
    role_classification = compute_role_classification(rows)
    retrieval_quality = compute_retrieval_quality(rows)

    return {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "input": str(path),
            "case_count": len(rows),
            "notes": {
                "hallucination_rate": (
                    "Automated proxy based on unsupported categories and implausible "
                    "unsupported amounts. Human review is recommended for final thesis reporting."
                ),
                "citation_correctness": (
                    "Null in pre_rag when no retrieved source IDs exist; post_rag expects "
                    "retrieved_sources with IDs such as S1."
                ),
                "pass_rates": (
                    "Threshold pass-rate / failure-mode breakdown. Expected labels are "
                    "always PASS in this benchmark, so this is NOT a balanced classification "
                    "confusion matrix — report it as a pass-rate per metric."
                ),
                "role_classification": (
                    "True 3×3 classification {Student, Worker, Freelancer}. Predicted role "
                    "is argmax of role-vocabulary scores in the model response."
                ),
            },
        },
        "aggregate": aggregate,
        "by_role": by_role,
        "pass_rates": pass_rates,
        "role_classification": role_classification,
        "retrieval_quality": retrieval_quality,
        "per_test": rows,
    }


def print_summary(payload: dict[str, Any]) -> None:
    print("\nThesis Evaluation Summary")
    print("=" * 72)
    print(f"Phase: {payload['meta']['phase']}")
    print(f"Cases: {payload['meta']['case_count']}")
    for key, value in payload["aggregate"].items():
        label = key.replace("_", " ").replace("pct", "%")
        shown = "N/A" if value is None else f"{value:.2f}%"
        print(f"{label:35s} {shown}")

    print("\nBy role")
    print("-" * 72)
    for role, metrics in payload["by_role"].items():
        print(
            f"{role:12s} n={metrics['count']:3d} "
            f"accuracy={metrics['financial_accuracy_pct']} "
            f"role_fit={metrics['role_appropriateness_pct']} "
            f"personalization={metrics['personalization_quality_pct']} "
            f"hallucination={metrics['hallucination_rate_pct']} "
            f"citation={metrics['citation_correctness_pct']}"
        )

    pr = payload.get("pass_rates", {}).get("overall", {})
    if pr:
        print("\nThreshold pass-rates (NOT a classification CM)")
        print("-" * 72)
        for metric, info in pr.items():
            rate = info.get("pass_rate_pct")
            shown = "N/A" if rate is None else f"{rate:.2f}%"
            print(f"{metric:30s} pass={info.get('pass'):>4} / n={info.get('n'):>4}  {shown}")
        modes = payload["pass_rates"].get("failure_modes", {})
        if modes:
            print("Failure modes (first tripped threshold):")
            for mode, count in sorted(modes.items(), key=lambda x: -x[1]):
                print(f"  {mode:30s} {count}")

    rc = payload.get("role_classification") or {}
    if rc.get("support"):
        print("\nRole classification (3×3 CM, macro-F1)")
        print("-" * 72)
        print(f"accuracy={rc['accuracy']}  macro_f1={rc['macro_f1']}  support={rc['support']}")
        header = "true/pred"
        print(f"{header:12s}" + "".join(f"{r:>12s}" for r in (*ROLES, 'unknown')))
        for true_role in ROLES:
            counts = rc["confusion_matrix"][true_role]
            print(f"{true_role:12s}" + "".join(f"{counts.get(p, 0):>12d}" for p in (*ROLES, 'unknown')))
        for role, stats in rc["per_role"].items():
            print(
                f"  {role:12s} P={stats['precision']} R={stats['recall']} "
                f"F1={stats['f1']} (support={stats['support']})"
            )

    rq = payload.get("retrieval_quality")
    if rq:
        print("\nRetrieval quality")
        print("-" * 72)
        print(
            f"n={rq['n']}  P@k={rq['precision_at_k']}  "
            f"R@k={rq['recall_at_k']}  MRR={rq['mrr']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate thesis model metrics before/after RAG.")
    parser.add_argument("--input", required=True, help="Benchmark JSON file to evaluate.")
    parser.add_argument("--phase", choices=["pre_rag", "post_rag"], required=True)
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    input_path = Path(args.input)
    payload = evaluate(input_path, args.phase)

    output_path = Path(args.output) if args.output else LOGS_DIR / (
        f"thesis_eval_{args.phase}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_paths = export_case_csvs(
        payload["per_test"],
        case_by_id(),
        output_path.parent,
        output_path.stem,
    )

    print_summary(payload)
    print(f"\nSaved thesis evaluation to {output_path}")
    for label, path in csv_paths.items():
        print(f"Saved {label}_cases CSV to {path}")


if __name__ == "__main__":
    main()
