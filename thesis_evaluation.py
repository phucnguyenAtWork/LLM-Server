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


def evaluate(path: Path, phase: str) -> dict[str, Any]:
    cases = case_by_id()
    rows = []

    for item in load_benchmark(path):
        case = cases.get(item.get("id"))
        if not case:
            continue

        message = normalize_text(item.get("response", ""))
        hallucination = score_hallucination(message, case)
        rows.append(
            {
                "id": item.get("id"),
                "name": item.get("name") or case.get("name"),
                "role": item.get("role") or case.get("role"),
                "financial_accuracy": score_financial_accuracy(item, case),
                "role_appropriateness": score_role_appropriateness(message, case.get("role", "")),
                "personalization_quality": score_personalization(message, case),
                "hallucination_rate": hallucination["rate"],
                "citation_correctness": score_citation_correctness(message, item, phase),
                "hallucination_details": hallucination,
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
            },
        },
        "aggregate": aggregate,
        "by_role": by_role,
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

    print_summary(payload)
    print(f"\nSaved thesis evaluation to {output_path}")


if __name__ == "__main__":
    main()
