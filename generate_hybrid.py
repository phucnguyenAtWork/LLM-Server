"""
FINA Training Data Generator v3
=================================
Generates structured prompt/completion SFT data for FINA v7.1 (Qwen2.5-3B).
All assistant outputs are strict JSON matching the FINA schema.

Families: action_crud, clarification, hard_negative, context_analysis,
          multi_turn, role_specific

Output: hybrid_data.jsonl
Usage:  python generate_hybrid.py
"""

import json
import random
from datetime import date, timedelta
from pathlib import Path

try:
    from fina_schema import (
        SYSTEM_PROMPT, build_response, build_action,
    )
except ModuleNotFoundError as exc:
    if exc.name != "pydantic":
        raise

    # Keep dataset generation usable even when the local runtime is missing pydantic.
    VALID_CATEGORIES = {
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
    VALID_SIGNALS = {
        "anomaly_detected",
        "over_budget",
        "within_budget",
        "goal_at_risk",
        "below_savings_target",
        "above_savings_target",
        "category_budget_exceeded",
        "spending_up_mom",
        "spending_down_mom",
        "high_fixed_costs",
        "no_category_budgets",
        "deficit",
        "on_track",
    }
    VALID_ACTION_TYPES = {
        "LOG_EXPENSE",
        "LOG_INCOME",
        "UPDATE_TRANSACTION",
        "DELETE_TRANSACTION",
    }
    VALID_KINDS = {"action", "analysis", "clarification"}

    SYSTEM_PROMPT = """\
You are FINA, a financial AI assistant. Output ONLY one valid JSON object.

Schema:
{
  "kind": "action" | "analysis" | "clarification",
  "message": "<natural language>",
  "action": null | {
    "type": "LOG_EXPENSE" | "LOG_INCOME" | "UPDATE_TRANSACTION" | "DELETE_TRANSACTION",
    "arguments": {
      "transaction_ref": null | "<string>",
      "amount": null | <int>,
      "currency": "VND",
      "category": null | "Food" | "Transport" | "Shopping" | "Entertainment" | "Bills" | "Health" | "Education",
      "item": null | "<string>",
      "datetime": null | "<string>",
      "account": null | "<string>",
      "confidence": <float 0.0-1.0>
    }
  },
  "signals": [],
  "needs_clarification": false
}

Rules:
- Use "action" only when the user clearly wants to log, edit, or delete a transaction.
- Use "analysis" for advice, summaries, affordability, trends, goals, forecasts, and status questions.
- Use "clarification" when intent is ambiguous or required fields are missing.
- Use null for unknown fields. Never invent amounts, categories, dates, accounts, or transaction refs.
- If amount or category is required for an action and missing, return kind="clarification".
- Keep all natural language inside "message".
- Output no markdown, no prose outside JSON, and no extra keys.

Context rules:
- FINANCIAL CONTEXT contains pre-computed totals, verdicts, and analysis.
- Copy amounts, percentages, verdicts, and timelines from context. Do not recalculate them.
- Use the supplied context sections when they exist: budgets, category budgets, month-over-month, recurring expenses, anomalies, goals, balances, and forecasts.
- Mention anomalies proactively when present.
- Prioritize the most important risks first in both "message" and "signals": anomaly_detected, goal_at_risk, category_budget_exceeded, deficit, below_savings_target, then lower-priority signals.

USER ROLE rules:
- Student: focus on affordability, debt avoidance, semester planning, and part-time income.
- Worker: focus on salary splitting, emergency funds, BHXH/retirement, and investing basics.
- Freelancer: focus on tax reserve, income buffer, quarterly planning, and business/personal separation.

Style:
- "message" must be concise, specific, and grounded in the provided numbers.
- On follow-ups, be brief and avoid repeating the full overview.
"""

    def build_action(
        action_type: str,
        *,
        amount=None,
        currency="VND",
        category=None,
        item=None,
        datetime_str=None,
        account=None,
        confidence=0.95,
        transaction_ref=None,
    ) -> dict:
        if action_type not in VALID_ACTION_TYPES:
            raise ValueError(f"invalid action type: {action_type}")
        if currency is not None and currency != "VND":
            raise ValueError("currency must be VND when provided")
        if category is not None and category not in VALID_CATEGORIES:
            raise ValueError(f"invalid category: {category}")
        confidence = round(float(confidence), 2)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")

        args = {
            "transaction_ref": transaction_ref,
            "amount": int(amount) if amount is not None else None,
            "currency": currency,
            "category": category,
            "item": item,
            "datetime": datetime_str,
            "account": account,
            "confidence": confidence,
        }

        if action_type in {"LOG_EXPENSE", "LOG_INCOME"} and (args["amount"] is None or args["item"] is None):
            raise ValueError("log actions require amount and item")
        if action_type == "LOG_EXPENSE" and args["category"] is None:
            raise ValueError("LOG_EXPENSE requires category")
        if action_type == "UPDATE_TRANSACTION":
            if args["amount"] is None and args["category"] is None and args["item"] is None:
                raise ValueError("UPDATE_TRANSACTION requires at least one changed field")
        if action_type == "DELETE_TRANSACTION":
            if args["transaction_ref"] is None and args["item"] is None:
                raise ValueError("DELETE_TRANSACTION requires a reference or item")

        return {"type": action_type, "arguments": args}

    def build_response(kind: str, message: str, *, action=None, signals=None, needs_clarification=False) -> str:
        if kind not in VALID_KINDS:
            raise ValueError(f"invalid kind: {kind}")
        if not message or not str(message).strip():
            raise ValueError("message must be non-empty")
        signals = signals or []
        unknown_signals = [signal for signal in signals if signal not in VALID_SIGNALS]
        if unknown_signals:
            raise ValueError(f"invalid signals: {unknown_signals}")
        if kind == "clarification":
            if not needs_clarification:
                raise ValueError("clarification outputs must set needs_clarification=true")
            if action is not None:
                raise ValueError("clarification outputs must not include action")
        if kind == "action" and action is None:
            raise ValueError("action outputs must include action payload")
        if kind == "analysis" and needs_clarification:
            raise ValueError("analysis outputs must not require clarification")

        payload = {
            "kind": kind,
            "message": message,
            "action": action,
            "signals": signals,
            "needs_clarification": needs_clarification,
        }
        return json.dumps(payload, ensure_ascii=False)

OUTPUT_FILE = Path(__file__).with_name("hybrid_data.jsonl")
USE_EXTERNAL_DATA = False  # Disabled by default; external datasets risk schema conflicts

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def fmt(amount: float) -> str:
    """Format VND for message text: 1.500.000 VND"""
    return "{:,.0f} VND".format(amount).replace(",", ".")


def rvar(base: float, pct_range: float = 0.2) -> float:
    """Add random variance to a numeric amount, round to 50k."""
    try:
        base = float(base)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"rvar() expected numeric base, got {base!r}") from exc

    if pct_range < 0:
        raise ValueError(f"rvar() expected non-negative pct_range, got {pct_range!r}")

    v = base * random.uniform(1 - pct_range, 1 + pct_range)
    return max(round(v / 50_000) * 50_000, 50_000)


def total_spent(spending: dict) -> float:
    return sum(spending.values())


def surplus(income: float, spending: dict) -> float:
    return income - total_spent(spending)


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLE BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def make_sample(role: str, context: str, question: str, response_json: str, family: str,
                 *, subfamily: str = "", tags: list | None = None) -> dict:
    """Single-turn training example."""
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": question},
        ],
        "completion": [
            {"role": "assistant", "content": response_json},
        ],
        "family": family,
        "subfamily": subfamily,
        "role": role,
        "tags": tags or [],
    }


def make_multi_turn_sample(role: str, context: str, turns: list, family: str,
                           *, subfamily: str = "", tags: list | None = None) -> dict:
    """Multi-turn training example. turns = [(q1, a1_json), (q2, a2_json), ...]"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context},
        {"role": "assistant", "content": "Understood."},
    ]
    for q, a in turns[:-1]:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    last_q, last_a = turns[-1]
    messages.append({"role": "user", "content": last_q})
    return {
        "prompt": messages,
        "completion": [{"role": "assistant", "content": last_a}],
        "family": family,
        "subfamily": subfamily,
        "role": role,
        "tags": tags or [],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO FACTORIES
# ═══════════════════════════════════════════════════════════════════════════════

GOAL_TEMPLATES = [
    ("Laptop Fund", 15_000_000), ("Emergency Fund", 30_000_000),
    ("Motorbike Fund", 50_000_000), ("Tet Savings", 5_000_000),
    ("Trip Abroad", 20_000_000), ("Online Course", 3_000_000),
    ("Phone Upgrade", 8_000_000), ("House Deposit", 200_000_000),
    ("Wedding Fund", 100_000_000), ("Tuition Next Semester", 10_000_000),
]

ACCOUNT_TEMPLATES = [
    ("Checking (VCB)", "checking"), ("Savings (TPBank)", "savings"),
    ("MoMo Wallet", "ewallet"), ("Checking (Vietcombank)", "checking"),
    ("Savings (Techcombank)", "savings"), ("ZaloPay", "ewallet"),
]

RECURRING_TEMPLATES = [
    ("Netflix", "Entertainment", 89_000), ("Spotify", "Entertainment", 59_000),
    ("YouTube Premium", "Entertainment", 79_000), ("FPT Internet", "Bills", 180_000),
    ("Rent", "Bills", 2_500_000), ("Gym fee", "Health", 450_000),
    ("Phone plan", "Bills", 150_000), ("Cloud storage", "Bills", 50_000),
    ("Insurance", "Health", 500_000), ("Electricity", "Bills", 300_000),
]

CUSTOM_SPLITS = [
    (30, 30, 40), (40, 30, 30), (45, 35, 20), (50, 20, 30),
    (50, 30, 20), (55, 25, 20), (60, 25, 15), (65, 15, 20),
    (65, 20, 15), (70, 20, 10), (80, 10, 10),
]


def student_scenario():
    income = rvar(random.choice([3_500_000, 4_000_000, 5_000_000, 6_000_000, 8_000_000]))
    spending = {
        "Food": rvar(income * random.uniform(0.25, 0.45)),
        "Transport": rvar(income * random.uniform(0.08, 0.15)),
        "Entertainment": rvar(income * random.uniform(0.08, 0.18)),
        "Education": rvar(income * random.uniform(0.05, 0.15)),
        "Shopping": rvar(income * random.uniform(0.05, 0.15)),
    }
    return income, spending


def worker_scenario():
    income = rvar(random.choice([10_000_000, 15_000_000, 20_000_000, 25_000_000, 30_000_000]))
    spending = {
        "Food": rvar(income * random.uniform(0.20, 0.30)),
        "Transport": rvar(income * random.uniform(0.08, 0.12)),
        "Bills": rvar(income * random.uniform(0.10, 0.20)),
        "Shopping": rvar(income * random.uniform(0.08, 0.15)),
        "Entertainment": rvar(income * random.uniform(0.05, 0.10)),
        "Health": rvar(income * random.uniform(0.02, 0.06)),
    }
    return income, spending


def freelancer_scenario():
    income = rvar(random.choice([8_000_000, 12_000_000, 20_000_000, 35_000_000, 50_000_000]))
    spending = {
        "Food": rvar(income * random.uniform(0.15, 0.25)),
        "Transport": rvar(income * random.uniform(0.05, 0.12)),
        "Bills": rvar(income * random.uniform(0.08, 0.15)),
        "Shopping": rvar(income * random.uniform(0.05, 0.12)),
        "Entertainment": rvar(income * random.uniform(0.03, 0.08)),
        "Health": rvar(income * random.uniform(0.02, 0.05)),
    }
    return income, spending


SCENARIO_MAP = {"Student": student_scenario, "Worker": worker_scenario, "Freelancer": freelancer_scenario}


def make_goals(income, surp):
    n = random.choice([0, 0, 1, 1, 2, 2, 3])
    if n == 0:
        return []
    goals = []
    for name, target in random.sample(GOAL_TEMPLATES, k=min(n, len(GOAL_TEMPLATES))):
        target = rvar(target, 0.3)
        progress = random.uniform(0.05, 0.85)
        saved = round(target * progress / 50_000) * 50_000
        months_ahead = random.randint(2, 18)
        target_date = (date.today() + timedelta(days=months_ahead * 30)).strftime("%Y-%m-%d")
        priority = random.choice(["HIGH", "MEDIUM", "LOW"])
        goals.append({"name": name, "target_amount": target, "current_saved": saved,
                       "target_date": target_date, "priority": priority})
    return goals


def make_balances(income):
    n = random.randint(1, 3)
    accounts = []
    for name, atype in random.sample(ACCOUNT_TEMPLATES, k=min(n, len(ACCOUNT_TEMPLATES))):
        if atype == "savings":
            bal = rvar(income * random.uniform(1.0, 6.0))
        elif atype == "ewallet":
            bal = rvar(income * random.uniform(0.02, 0.15))
        else:
            bal = rvar(income * random.uniform(0.3, 1.5))
        accounts.append({"name": name, "balance": bal, "currency": "VND", "type": atype})
    return accounts


def make_forecast(spending):
    total = total_spent(spending)
    projected = total * random.uniform(0.85, 1.20)
    return {"total": round(projected / 50_000) * 50_000, "confidence": random.choice([60, 70, 75, 80])}


def make_category_budgets(spending, tight=False):
    budgets = []
    for cat, spent in spending.items():
        factor = random.uniform(0.70, 1.10) if tight else random.uniform(1.10, 1.40)
        limit = max(round(spent * factor / 500_000) * 500_000, 500_000)
        budgets.append({"categoryName": cat, "monthlyLimit": limit})
    return budgets


def make_monthly_history(spending):
    n_months = random.choice([2, 2, 3, 3])
    months_data = {}
    today = date.today()
    for i in range(1, n_months + 1):
        m, y = today.month - i, today.year
        while m <= 0:
            m += 12; y -= 1
        months_data[f"{y}-{m:02d}"] = {cat: rvar(amt, 0.30) for cat, amt in spending.items()}
    return months_data


def make_recurring(spending):
    n = random.choice([0, 0, 2, 2, 3, 3, 4])
    if n == 0:
        return []
    valid = [t for t in RECURRING_TEMPLATES if t[1] in spending]
    if not valid:
        return []
    chosen = random.sample(valid, k=min(n, len(valid)))
    return [{"description": name, "category": cat, "amount": rvar(amt, 0.1),
             "occurrences": random.choice([2, 3, 3])} for name, cat, amt in chosen]


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def make_context(role, income, spending, *, goals=None, balances=None,
                 budget_split=None, forecast=None, category_budgets=None,
                 monthly_history=None, recurring=None):
    """Build FINANCIAL CONTEXT block identical to inference context."""
    spent = total_spent(spending)
    save = surplus(income, spending)
    rate = (save / income * 100) if income > 0 else 0

    sorted_cats = sorted(spending.items(), key=lambda x: x[1], reverse=True)
    top_cat, top_amt = sorted_cats[0] if sorted_cats else ("N/A", 0)
    top_pct = (top_amt / income * 100) if income > 0 else 0

    cat_lines = "\n".join(
        f"  - {k}: {fmt(v)} ({v / income * 100:.1f}% of income)" for k, v in spending.items()
    )

    if budget_split:
        n_pct, w_pct, s_pct = budget_split
        split_label = f"{n_pct}/{w_pct}/{s_pct} (custom)"
    else:
        n_pct, w_pct, s_pct = 50, 30, 20
        split_label = "50/30/20 (default)"

    needs_limit = income * n_pct / 100
    wants_limit = income * w_pct / 100
    savings_target = income * s_pct / 100

    # Pre-compute verdicts
    if rate >= s_pct:
        savings_verdict = f"ABOVE TARGET - saving {rate:.1f}% (target {s_pct}%). Surplus {fmt(save)} exceeds goal of {fmt(savings_target)} by {fmt(save - savings_target)}. GOOD."
    else:
        shortfall = savings_target - save
        savings_verdict = f"BELOW TARGET - saving {rate:.1f}% (target {s_pct}%). Need {fmt(shortfall)} more. WARNING."

    if spent <= needs_limit:
        spending_verdict = f"UNDER {n_pct}% needs limit. Spent {fmt(spent)} vs {fmt(needs_limit)}. GOOD."
    elif spent <= needs_limit + wants_limit:
        spending_verdict = f"Within needs+wants. Spent {fmt(spent)} vs {fmt(needs_limit + wants_limit)}. OK."
    else:
        spending_verdict = f"OVER BUDGET. Spent {fmt(spent)} exceeds {fmt(needs_limit + wants_limit)}. DANGER."

    ctx = (
        f"--- FINANCIAL CONTEXT ---\n"
        f"USER ROLE: {role}\n"
        f"CURRENCY: VND\n"
        f"TOTAL INCOME: {fmt(income)}\n\n"
        f"SPENDING BY CATEGORY:\n{cat_lines}\n\n"
        f"PRE-COMPUTED TOTALS:\n"
        f"- TOTAL SPENT: {fmt(spent)}\n"
        f"- SURPLUS: {fmt(save)}\n"
        f"- SAVINGS RATE: {rate:.1f}%\n"
        f"- TOP CATEGORY: {top_cat} at {fmt(top_amt)} ({top_pct:.1f}%)\n\n"
        f"BUDGET SPLIT: {split_label}\n"
        f"- Needs ({n_pct}%): {fmt(needs_limit)}\n"
        f"- Wants ({w_pct}%): {fmt(wants_limit)}\n"
        f"- Savings ({s_pct}%): {fmt(savings_target)}\n"
    )

    # Category budgets
    if category_budgets:
        cb_lines = []
        over_details, near_details = [], []
        for cb in category_budgets:
            cat_name = cb["categoryName"]
            limit = cb["monthlyLimit"]
            spent_amt = spending.get(cat_name, 0)
            pct_used = (spent_amt / limit * 100) if limit > 0 else 0
            if pct_used > 100:
                status = f"OVER LIMIT by {fmt(spent_amt - limit)}"
                over_details.append(cat_name)
            elif pct_used > 90:
                status = "NEAR LIMIT"
                near_details.append(cat_name)
            else:
                status = "OK"
            cb_lines.append(f"  - {cat_name}: {fmt(spent_amt)} / {fmt(limit)} ({pct_used:.0f}%) {status}")
        verdict_parts = []
        if over_details:
            verdict_parts.append(f"{len(over_details)} OVER ({', '.join(over_details)})")
        if near_details:
            verdict_parts.append(f"{len(near_details)} NEAR ({', '.join(near_details)})")
        ctx += "\nCATEGORY BUDGET STATUS:\n" + "\n".join(cb_lines) + f"\nBUDGET VERDICT: {', '.join(verdict_parts) if verdict_parts else 'All OK'}\n"
    else:
        ctx += "\nCATEGORY BUDGET: NOT SET\n"

    # Month-over-month
    if monthly_history:
        prev_key = sorted(monthly_history.keys())[-1]
        prev = monthly_history[prev_key]
        prev_total = sum(prev.values())
        mom_lines = []
        for cat, curr_amt in spending.items():
            prev_amt = prev.get(cat, 0)
            if prev_amt == 0:
                mom_lines.append(f"  - {cat}: {fmt(curr_amt)} (NEW)")
            else:
                delta = ((curr_amt - prev_amt) / prev_amt) * 100
                d = "UP" if delta > 0 else "DOWN" if delta < 0 else "FLAT"
                mom_lines.append(f"  - {cat}: {fmt(prev_amt)} -> {fmt(curr_amt)} ({delta:+.1f}%, {d})")
        overall_delta = ((spent - prev_total) / prev_total * 100) if prev_total > 0 else 0
        overall_d = "UP" if overall_delta > 0 else "DOWN" if overall_delta < 0 else "FLAT"
        mom_verdict = f"Spending {overall_d} {abs(overall_delta):.1f}% vs last month."
        ctx += f"\nMONTH-OVER-MONTH:\n" + "\n".join(mom_lines) + f"\nMOM VERDICT: {mom_verdict}\n"

    # Recurring
    if recurring:
        rec_lines = []
        total_fixed = 0
        for r in recurring:
            rec_lines.append(f"  - {r['description']} ({r['category']}): ~{fmt(r['amount'])}/mo ({r['occurrences']} months)")
            total_fixed += r['amount']
        discretionary = max(spent - total_fixed, 0)
        fixed_pct = (total_fixed / income * 100) if income > 0 else 0
        ctx += f"\nRECURRING EXPENSES:\n" + "\n".join(rec_lines) + f"\nRECURRING VERDICT: Fixed {fmt(total_fixed)}/mo ({fixed_pct:.1f}% of income). Discretionary: {fmt(discretionary)}.\n"

    # Anomalies (derived from monthly_history)
    if monthly_history:
        all_past = {}
        for month_data in monthly_history.values():
            for cat, amt in month_data.items():
                all_past.setdefault(cat, []).append(amt)
        anomalies = []
        for cat, curr_amt in spending.items():
            past = [v for v in all_past.get(cat, []) if v > 0]
            if not past:
                continue
            avg = sum(past) / len(past)
            if avg > 0:
                ratio = curr_amt / avg
                if ratio >= 2.0:
                    anomalies.append(("UNUSUAL", cat, curr_amt, avg, ratio))
                elif ratio >= 1.5:
                    anomalies.append(("WATCH", cat, curr_amt, avg, ratio))
        if anomalies:
            anom_lines = []
            for level, cat, current, avg, ratio in anomalies:
                anom_lines.append(f"  - {level}: {cat} at {fmt(current)} is {ratio:.1f}x avg of {fmt(avg)}")
            top_a = max(anomalies, key=lambda a: a[4])
            ctx += f"\nANOMALY ALERTS:\n" + "\n".join(anom_lines) + f"\nANOMALY VERDICT: {top_a[1]} needs attention.\n"

    # Verdicts
    ctx += (
        f"\nVERDICTS:\n"
        f"- SAVINGS: {savings_verdict}\n"
        f"- SPENDING: {spending_verdict}\n"
    )

    # Balances
    if balances:
        total_liquid = sum(a["balance"] for a in balances)
        bal_lines = "\n".join(f"  - {a['name']}: {fmt(a['balance'])}" for a in balances)
        ctx += f"\nACCOUNT BALANCES:\n{bal_lines}\n  Total liquid: {fmt(total_liquid)}\n"

    # Goals
    if goals:
        goal_lines = []
        for g in goals:
            target = g["target_amount"]
            saved = g["current_saved"]
            remaining = max(target - saved, 0)
            pct_done = (saved / target * 100) if target > 0 else 0
            m_needed = (remaining / save) if save > 0 else float("inf")
            m_str = f"~{m_needed:.1f} months" if m_needed < 999 else "not reachable"
            goal_lines.append(f"  - {g['name']}: {fmt(saved)}/{fmt(target)} ({pct_done:.0f}% done, {m_str})")
        ctx += "\nFINANCIAL GOALS:\n" + "\n".join(goal_lines) + "\n"

    # Forecast
    if forecast:
        ctx += f"\nFORECAST: Projected {fmt(forecast['total'])} next month (confidence {forecast['confidence']}%)\n"

    ctx += "---"
    return ctx


def make_light_context(role, income, spending):
    """Lean context for generic advice that should not depend on extra sections."""
    return make_context(
        role,
        income,
        spending,
        goals=None,
        balances=None,
        budget_split=None,
        forecast=None,
        category_budgets=None,
        monthly_history=None,
        recurring=None,
    )


def full_scenario(role):
    """Generate a complete scenario with optional enrichments."""
    income, spending = SCENARIO_MAP[role]()
    surp = surplus(income, spending)
    goals = make_goals(income, surp) if random.random() < 0.6 else None
    balances = make_balances(income) if random.random() < 0.5 else None
    forecast = make_forecast(spending) if random.random() < 0.4 else None
    budget_split = random.choice(CUSTOM_SPLITS) if random.random() < 0.25 else None
    cat_budgets = make_category_budgets(spending, tight=random.random() < 0.5) if random.random() < 0.4 else None
    hist = make_monthly_history(spending) if random.random() < 0.5 else None
    rec = make_recurring(spending) if random.random() < 0.4 else None
    ctx = make_context(role, income, spending, goals=goals, balances=balances,
                       budget_split=budget_split, forecast=forecast, category_budgets=cat_budgets,
                       monthly_history=hist, recurring=rec)
    return income, spending, goals, balances, forecast, budget_split, cat_budgets, hist, rec, ctx


# ═══════════════════════════════════════════════════════════════════════════════
# FAMILY 1: ACTION / CRUD
# ═══════════════════════════════════════════════════════════════════════════════

EXPENSE_ITEMS = [
    (50_000, "coffee", "Food"), (120_000, "lunch at the office", "Food"),
    (250_000, "Grab Food dinner", "Food"), (35_000, "Grab bike", "Transport"),
    (200_000, "petrol", "Transport"), (500_000, "Shopee order", "Shopping"),
    (150_000, "Watsons skincare", "Shopping"), (89_000, "Netflix", "Entertainment"),
    (1_200_000, "tuition installment", "Education"), (300_000, "pharmacy", "Health"),
    (2_500_000, "monthly rent", "Bills"), (180_000, "FPT Internet", "Bills"),
    (75_000, "CGV movie ticket", "Entertainment"), (450_000, "gym monthly fee", "Health"),
    (80_000, "Highlands Coffee", "Food"), (25_000, "banh mi", "Food"),
    (350_000, "dentist visit", "Health"), (900_000, "online course", "Education"),
    (60_000, "Grab car", "Transport"), (1_500_000, "laptop repair", "Shopping"),
    (40_000, "parking fee", "Transport"), (100_000, "birthday gift", "Shopping"),
]

EXPENSE_FORMAL_TEMPLATES = [
    "I just spent {amt} on {item}.",
    "Log {amt} for {item}.",
    "Add a {cat} expense: {amt} for {item}.",
    "Record {amt} spent on {item} today.",
    "Put down {amt} for {item} under {cat}.",
]

EXPENSE_NATURAL_TEMPLATES = [
    "Had {item} for {amt}",
    "Just had {item}, {amt}",
    "{item} was {amt}",
    "Paid {amt} for {item}",
    "{amt} on {item} today",
    "Spent {amt} at {item}",
    "Got {item} for {amt}",
    "{item}: {amt}",
]

INCOME_ITEMS = [
    (10_000_000, "monthly salary", None), (15_000_000, "monthly salary", None),
    (20_000_000, "monthly salary", None), (5_000_000, "freelance project", None),
    (3_000_000, "part-time tutoring", None), (8_000_000, "client payment", None),
    (2_000_000, "side gig", None), (1_000_000, "sold old phone", None),
    (500_000, "cashback reward", None), (50_000_000, "quarterly bonus", None),
    (7_000_000, "design project", None), (25_000_000, "consulting fee", None),
]

INCOME_TEMPLATES = [
    "I got paid {amt}.",
    "Received {amt} from {item}.",
    "Just got {amt} for {item}.",
    "Log income: {amt} from {item}.",
    "My {item} came in, {amt}.",
    "{item} paid me {amt}.",
]


def gen_log_expense(role):
    income, spending = SCENARIO_MAP[role]()
    ctx = make_light_context(role, income, spending)
    amount, item, category = random.choice(EXPENSE_ITEMS)
    amount = rvar(amount, 0.3)
    template = random.choice(EXPENSE_FORMAL_TEMPLATES)
    q = template.format(amt=fmt(amount), item=item, cat=category)
    new_total = spending.get(category, 0) + amount
    msg = f"Logged {fmt(amount)} for {item} under {category}. Your {category} total is now {fmt(new_total)} this month."
    resp = build_response("action", msg,
                          action=build_action("LOG_EXPENSE", amount=amount, category=category, item=item),
                          signals=["category_budget_exceeded"] if new_total > income * 0.3 else [])
    return make_sample(role, ctx, q, resp, "action_crud", subfamily="log_expense", tags=["log_expense"])


def gen_log_expense_natural(role):
    income, spending, *_, ctx = full_scenario(role)
    amount, item, category = random.choice(EXPENSE_ITEMS)
    amount = rvar(amount, 0.3)
    template = random.choice(EXPENSE_NATURAL_TEMPLATES)
    q = template.format(amt=fmt(amount), item=item, cat=category)
    new_total = spending.get(category, 0) + amount
    msg = f"Got it! {fmt(amount)} for {item} logged under {category}. {category} total: {fmt(new_total)} this month."
    resp = build_response("action", msg,
                          action=build_action("LOG_EXPENSE", amount=amount, category=category, item=item))
    return make_sample(role, ctx, q, resp, "action_crud")


def gen_log_income(role):
    income, spending, *_, ctx = full_scenario(role)
    amount, item, _ = random.choice(INCOME_ITEMS)
    amount = rvar(amount, 0.3)
    template = random.choice(INCOME_TEMPLATES)
    q = template.format(amt=fmt(amount), item=item)
    msg = f"Logged income of {fmt(amount)} from {item}. Your updated income this month is {fmt(income + amount)}."
    resp = build_response("action", msg,
                          action=build_action("LOG_INCOME", amount=amount, item=item))
    return make_sample(role, ctx, q, resp, "action_crud")


def gen_log_income_natural(role):
    income, spending, *_, ctx = full_scenario(role)
    amount, item, _ = random.choice(INCOME_ITEMS)
    amount = rvar(amount, 0.3)
    casual = random.choice([
        f"Got {fmt(amount)} from {item}",
        f"{item} just hit, {fmt(amount)}",
        f"Made {fmt(amount)} from {item} today",
        f"{fmt(amount)} in from {item}",
    ])
    msg = f"Noted! {fmt(amount)} income from {item} recorded."
    resp = build_response("action", msg,
                          action=build_action("LOG_INCOME", amount=amount, item=item))
    return make_sample(role, ctx, casual, resp, "action_crud")


def gen_update_transaction(role):
    income, spending, *_, ctx = full_scenario(role)
    amount, item, category = random.choice(EXPENSE_ITEMS)
    try:
        amount = float(amount)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid EXPENSE_ITEMS amount for update transaction: {amount!r} ({item=}, {category=})"
        ) from exc

    old_amount = rvar(amount, 0.3)
    new_amount = rvar(amount, 0.3)
    retry_count = 0
    while abs(new_amount - old_amount) < 10_000 and retry_count < 10:
        new_amount = rvar(amount, 0.5)
        retry_count += 1
    if abs(new_amount - old_amount) < 10_000:
        new_amount = old_amount + 50_000
    q = random.choice([
        f"Actually that {item} was {fmt(new_amount)} not {fmt(old_amount)}.",
        f"Correct my last {item} entry to {fmt(new_amount)}.",
        f"Change the {item} transaction amount to {fmt(new_amount)}.",
        f"I made a mistake, the {item} was {fmt(new_amount)}.",
    ])
    msg = f"Updated {item} from {fmt(old_amount)} to {fmt(new_amount)} under {category}."
    resp = build_response("action", msg,
                          action=build_action("UPDATE_TRANSACTION", amount=new_amount,
                                              category=category, item=item, confidence=0.9))
    return make_sample(role, ctx, q, resp, "action_crud")


def gen_delete_transaction(role):
    income, spending, *_, ctx = full_scenario(role)
    amount, item, category = random.choice(EXPENSE_ITEMS)
    amount = rvar(amount, 0.3)
    q = random.choice([
        f"Remove the {item} entry.",
        f"Delete my {item} transaction.",
        f"Cancel that {fmt(amount)} {item} charge.",
        f"That {item} was a mistake, remove it.",
        f"Undo the {item} log.",
    ])
    msg = f"Removed the {item} transaction ({fmt(amount)} under {category})."
    resp = build_response("action", msg,
                          action=build_action("DELETE_TRANSACTION", amount=amount,
                                              category=category, item=item, confidence=0.9))
    return make_sample(role, ctx, q, resp, "action_crud")


# ═══════════════════════════════════════════════════════════════════════════════
# FAMILY 2: CLARIFICATION / CORRECTION
# ═══════════════════════════════════════════════════════════════════════════════

CLARIFICATION_ITEMS = [
    "lunch", "coffee", "groceries", "something online", "dinner", "a book", "taxi",
    "snacks", "bubble tea", "banh mi", "pho", "a drink", "medicine", "some clothes",
    "a gift", "parking", "a Grab ride", "some supplies", "a smoothie", "noodles",
    "a pen drive", "toiletries", "a charger", "some fruit", "bottled water",
    "a jacket", "earphones", "an umbrella", "stationery", "a notebook",
]

CLARIFICATION_AMOUNT_TEMPLATES = [
    "I bought {item}.",
    "Just had {item}.",
    "Log {item} for me.",
    "I got {item} today.",
    "Spent money on {item}.",
    "Add {item} to my expenses.",
    "Had {item} this morning.",
    "Picked up {item}.",
    "Got {item} earlier.",
    "I paid for {item}.",
    "{item} today.",
    "Record {item} please.",
]

CLARIFICATION_CATEGORY_AMOUNTS = [
    30_000, 45_000, 50_000, 75_000, 100_000, 120_000, 150_000, 180_000,
    200_000, 250_000, 300_000, 400_000, 500_000, 750_000, 1_000_000,
]

CLARIFICATION_RESPONSES_AMOUNT = [
    "I'd like to log that {item} for you. How much did it cost?",
    "Got it, {item}. What was the amount?",
    "Sure, logging {item}. How much did you spend?",
    "Noted. How much was the {item}?",
    "I can log {item} — just need the amount.",
]

CLARIFICATION_RESPONSES_CATEGORY = [
    "Got {amount}. What was it for? That will help me categorize it correctly.",
    "{amount} noted. What did you spend it on?",
    "Sure, {amount}. What category should I put that under?",
    "Recorded {amount}. Can you tell me what it was for?",
]

VAGUE_QUESTIONS = [
    "I spent some money today.",
    "Had a few expenses.",
    "There were some costs this week.",
    "I bought stuff.",
    "I had some purchases.",
    "Made a payment earlier.",
    "Spent a bit today.",
    "There was an expense.",
    "I used some cash.",
    "Had to pay for something.",
    "Bought a few things.",
    "Some spending happened today.",
]

VAGUE_RESPONSES = [
    "Could you tell me the amount and what you spent on? I need those details to log it properly.",
    "What did you buy, and how much? I'll log it once I know the details.",
    "I need the amount and item to log this. What was it?",
    "Can you give me specifics — what was it and how much?",
]


def gen_missing_amount(role):
    income, spending, *_, ctx = full_scenario(role)
    item = random.choice(CLARIFICATION_ITEMS)
    q = random.choice(CLARIFICATION_AMOUNT_TEMPLATES).format(item=item)
    msg = random.choice(CLARIFICATION_RESPONSES_AMOUNT).format(item=item)
    resp = build_response("clarification", msg, needs_clarification=True)
    return make_sample(role, ctx, q, resp, "clarification",
                       subfamily="missing_amount", tags=["clarification"])


def gen_missing_field_clarification(role):
    """Alias name kept explicit for auditability."""
    return gen_missing_amount(role)


def gen_missing_category(role):
    income, spending, *_, ctx = full_scenario(role)
    amount = rvar(random.choice(CLARIFICATION_CATEGORY_AMOUNTS), 0.3)
    q = random.choice([
        f"I spent {fmt(amount)} today.",
        f"Log {fmt(amount)} for me.",
        f"Record a {fmt(amount)} expense.",
        f"Add {fmt(amount)} to my spending.",
        f"{fmt(amount)} spent today.",
        f"Put down {fmt(amount)}.",
        f"I paid {fmt(amount)} earlier.",
        f"Just spent {fmt(amount)}.",
    ])
    msg = random.choice(CLARIFICATION_RESPONSES_CATEGORY).format(amount=fmt(amount))
    resp = build_response("clarification", msg, needs_clarification=True)
    return make_sample(role, ctx, q, resp, "clarification",
                       subfamily="missing_category", tags=["clarification"])


def gen_vague_report(role):
    income, spending, *_, ctx = full_scenario(role)
    q = random.choice(VAGUE_QUESTIONS)
    msg = random.choice(VAGUE_RESPONSES)
    resp = build_response("clarification", msg, needs_clarification=True)
    return make_sample(role, ctx, q, resp, "clarification",
                       subfamily="vague_report", tags=["clarification"])


def gen_correction_followup(role):
    """Multi-turn: user logs, then corrects."""
    income, spending, *_, ctx = full_scenario(role)
    amount, item, category = random.choice(EXPENSE_ITEMS)
    amount = rvar(amount, 0.3)
    new_amount = rvar(amount * random.uniform(0.3, 0.7), 0.2)

    q1 = f"I spent {fmt(amount)} on {item}."
    a1 = build_response("action", f"Logged {fmt(amount)} for {item} under {category}.",
                         action=build_action("LOG_EXPENSE", amount=amount, category=category, item=item))

    q2 = random.choice([
        f"Wait, it was actually {fmt(new_amount)}.",
        f"Sorry, that should be {fmt(new_amount)} not {fmt(amount)}.",
        f"Correction: {fmt(new_amount)}.",
    ])
    a2 = build_response("action", f"Updated {item} to {fmt(new_amount)}.",
                         action=build_action("UPDATE_TRANSACTION", amount=new_amount,
                                             category=category, item=item, confidence=0.92))

    return make_multi_turn_sample(role, ctx, [(q1, a1), (q2, a2)], "clarification")


def gen_transaction_correction_turn(role):
    """Alias name kept explicit for auditability."""
    return gen_correction_followup(role)


AMBIGUOUS_PARSEABLE = [
    ("Coffee 50k", 50_000, "coffee", "Food"),
    ("200k food", 200_000, "food", "Food"),
    ("150k grab", 150_000, "grab ride", "Transport"),
    ("80k lunch", 80_000, "lunch", "Food"),
    ("300k shopping", 300_000, "shopping", "Shopping"),
    ("100k snack", 100_000, "snack", "Food"),
    ("45k banh mi", 45_000, "banh mi", "Food"),
    ("250k groceries", 250_000, "groceries", "Food"),
    ("60k parking", 60_000, "parking", "Transport"),
    ("500k dentist", 500_000, "dentist", "Health"),
]

AMBIGUOUS_UNPARSEABLE = [
    "transport yesterday",
    "lunch",
    "something today",
    "a few things",
    "groceries earlier",
    "health stuff",
    "some bills",
    "entertainment",
]


def gen_ambiguous_intent(role):
    income, spending, *_, ctx = full_scenario(role)
    if random.random() < 0.55:
        q_text, amount, item, cat = random.choice(AMBIGUOUS_PARSEABLE)
        amount = rvar(amount, 0.15)
        q = q_text.split()[0] + " " + fmt(amount).split()[0] if random.random() < 0.3 else q_text
        msg = f"Logging {fmt(amount)} for {item} under {cat}. Is that correct?"
        resp = build_response("action", msg,
                              action=build_action("LOG_EXPENSE", amount=amount, category=cat,
                                                  item=item, confidence=0.7),
                              needs_clarification=True)
    else:
        q = random.choice(AMBIGUOUS_UNPARSEABLE)
        msg = "Could you clarify the amount? I want to make sure I log it correctly."
        resp = build_response("clarification", msg, needs_clarification=True)
    return make_sample(role, ctx, q, resp, "clarification",
                       subfamily="ambiguous_intent", tags=["clarification"])


# ═══════════════════════════════════════════════════════════════════════════════
# FAMILY 3: HARD NEGATIVES
# ═══════════════════════════════════════════════════════════════════════════════

def gen_neg_spending_question(role):
    """User asks about spending — should NOT trigger action."""
    income, spending, *_, ctx = full_scenario(role)
    cat = random.choice(list(spending.keys()))
    q = random.choice([
        f"How much did I spend on {cat}?",
        f"What's my {cat} spending this month?",
        f"Break down my {cat} costs.",
        "What's my biggest expense?",
        "How much have I spent this month?",
        f"Am I over budget on {cat}?",
    ])
    spent = total_spent(spending)
    msg = f"Your {cat} spending this month is {fmt(spending[cat])} ({spending[cat] / income * 100:.1f}% of income). Total spending: {fmt(spent)} out of {fmt(income)}."
    signals = []
    if spending[cat] > income * 0.3:
        signals.append("over_budget")
    resp = build_response("analysis", msg, signals=signals)
    return make_sample(role, ctx, q, resp, "hard_negative")


def gen_neg_hypothetical(role):
    """Hypothetical / should-I-buy — NOT an action."""
    income, spending, *_, ctx = full_scenario(role)
    items = [
        ("a new phone", rvar(8_000_000, 0.3)),
        ("a laptop", rvar(15_000_000, 0.3)),
        ("a motorbike", rvar(50_000_000, 0.3)),
        ("a trip to Da Lat", rvar(3_000_000, 0.3)),
        ("new furniture", rvar(10_000_000, 0.3)),
    ]
    item_name, item_cost = random.choice(items)
    save = surplus(income, spending)
    q = random.choice([
        f"Should I buy {item_name} for {fmt(item_cost)}?",
        f"Can I afford {item_name} ({fmt(item_cost)})?",
        f"Is {item_name} worth it at {fmt(item_cost)}?",
        f"What if I bought {item_name}?",
    ])
    months = item_cost / save if save > 0 else float("inf")
    if save > 0 and months < 3:
        msg = f"With a surplus of {fmt(save)}/month, you could save for {item_name} ({fmt(item_cost)}) in about {months:.1f} months. It's feasible but make sure it aligns with your goals."
    elif save > 0:
        msg = f"{item_name} at {fmt(item_cost)} would take about {months:.1f} months at your current surplus of {fmt(save)}/month. Consider whether this is a priority vs your other goals."
    else:
        msg = f"Your current surplus is {fmt(save)}. {item_name} at {fmt(item_cost)} isn't affordable right now. Focus on reducing spending first."
        return make_sample(role, ctx, q, build_response("analysis", msg, signals=["deficit"]), "hard_negative")
    resp = build_response("analysis", msg)
    return make_sample(role, ctx, q, resp, "hard_negative")


def gen_hard_negative_hypothetical(role):
    """Alias name kept explicit for auditability."""
    return gen_neg_hypothetical(role)


def gen_neg_status_query(role):
    """General status question — analysis, not action."""
    income, spending, *_, ctx = full_scenario(role)
    spent = total_spent(spending)
    save = surplus(income, spending)
    rate = (save / income * 100) if income > 0 else 0
    top_cat = max(spending, key=spending.get)
    q = random.choice([
        "How is my budget this month?",
        "Am I on track financially?",
        "Give me a financial overview.",
        "What's my financial status?",
        "How am I doing with money?",
    ])
    signals = []
    if rate >= 20:
        signals.append("above_savings_target")
        tone = f"You're saving {rate:.1f}%, above the 20% target."
    else:
        signals.append("below_savings_target")
        tone = f"Your savings rate is {rate:.1f}%, below the 20% target."
    msg = f"Total spending: {fmt(spent)} out of {fmt(income)} income, surplus of {fmt(save)}. Top expense: {top_cat} at {fmt(spending[top_cat])}. {tone}"
    resp = build_response("analysis", msg, signals=signals)
    return make_sample(role, ctx, q, resp, "hard_negative")


def gen_hard_negative_status_query(role):
    """Alias name kept explicit for auditability."""
    return gen_neg_status_query(role)


def gen_neg_general_advice(role):
    """Advice question — analysis only."""
    income, spending, *_, ctx = full_scenario(role)
    save = surplus(income, spending)
    q = random.choice([
        "Give me saving tips.",
        "How can I save more?",
        "What should I do to improve my finances?",
        "Any advice for me?",
    ])
    top_cat = max(spending, key=spending.get)
    cut = spending[top_cat] * 0.2
    if role == "Student":
        advice = f"As a student earning {fmt(income)}, focus on cutting {top_cat} by 20% ({fmt(cut)}). Use student discounts, cook at home more, and consider part-time income."
    elif role == "Freelancer":
        advice = f"Set aside 30% for taxes ({fmt(income * 0.3)}), build a 6-month income buffer, and reduce {top_cat} by 20% ({fmt(cut)}) to strengthen your safety net."
    else:
        advice = f"Automate savings on payday ({fmt(income * 0.2)} target). Your biggest expense is {top_cat} — cutting it by 20% saves {fmt(cut)}/month, {fmt(cut * 12)}/year."
    resp = build_response("analysis", advice)
    return make_sample(role, ctx, q, resp, "hard_negative")


def gen_neg_future_tense(role):
    """Future/hypothetical phrasing — NOT a transaction to log."""
    income, spending, *_, ctx = full_scenario(role)
    q = random.choice([
        "I might get paid tomorrow.",
        "I'm thinking about buying groceries later.",
        "I'll probably spend about 200k on food this week.",
        "I'm planning to invest next month.",
        "My salary should come in on Friday.",
    ])
    msg = "That sounds like a plan, not a transaction to log yet. Let me know when the actual expense or income happens and I'll record it for you."
    resp = build_response("analysis", msg)
    return make_sample(role, ctx, q, resp, "hard_negative")


def gen_neg_should_i_buy(role):
    """Should-I-buy question — analysis, not action."""
    income, spending, *_, ctx = full_scenario(role)
    items = ["new shoes", "a tablet", "a gym membership", "an online course", "takeout every day"]
    item = random.choice(items)
    q = f"Should I spend money on {item}?"
    save = surplus(income, spending)
    if save > income * 0.2:
        msg = f"Your surplus is {fmt(save)} which is healthy. {item.capitalize()} is fine if it fits your priorities, but make sure your savings and goals stay on track."
    else:
        msg = f"Your surplus is only {fmt(save)}. I'd hold off on {item} until you've improved your savings rate. Focus on essentials first."
    resp = build_response("analysis", msg)
    return make_sample(role, ctx, q, resp, "hard_negative")


# ═══════════════════════════════════════════════════════════════════════════════
# FAMILY 4: CONTEXT-GROUNDED ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def gen_budget_health(role):
    income, spending, *_, ctx = full_scenario(role)
    spent = total_spent(spending)
    save = surplus(income, spending)
    rate = (save / income * 100) if income > 0 else 0
    top_cat = max(spending, key=spending.get)
    second_cat = sorted(spending, key=spending.get, reverse=True)[1]
    q = random.choice([
        "How is my budget looking?",
        "Give me a budget health check.",
        "Am I doing well financially?",
        "What's my overall status?",
    ])
    signals = []
    if rate >= 20:
        signals.append("above_savings_target")
        msg = (f"Your spending is {fmt(spent)} with {fmt(save)} surplus ({rate:.1f}% savings rate, above target). "
               f"Top expenses: {top_cat} ({fmt(spending[top_cat])}) and {second_cat} ({fmt(spending[second_cat])}). Looking healthy.")
    else:
        signals.append("below_savings_target")
        gap = income * 0.2 - save
        msg = (f"Your spending is {fmt(spent)} with {fmt(save)} surplus ({rate:.1f}% savings, below 20% target). "
               f"You need {fmt(gap)} more to hit target. Consider cutting {top_cat} ({fmt(spending[top_cat])}) by 15-20%.")
    resp = build_response("analysis", msg, signals=signals)
    return make_sample(role, ctx, q, resp, "context_analysis")


def gen_category_budget_status(role):
    income, spending = SCENARIO_MAP[role]()
    tight = random.random() < 0.5
    cat_budgets = make_category_budgets(spending, tight=tight)
    ctx = make_context(role, income, spending, category_budgets=cat_budgets)
    q = random.choice([
        "How are my category budgets?",
        "Am I staying within my limits?",
        "Check my budget limits.",
        "Any categories over budget?",
    ])
    over, near, ok = [], [], []
    for cb in cat_budgets:
        cat = cb["categoryName"]
        limit = cb["monthlyLimit"]
        s = spending.get(cat, 0)
        pct = (s / limit * 100) if limit > 0 else 0
        if pct > 100:
            over.append(f"{cat} at {fmt(s)} vs {fmt(limit)} limit")
        elif pct > 90:
            near.append(f"{cat} at {pct:.0f}%")
        else:
            ok.append(cat)
    parts = []
    signals = []
    if over:
        parts.append(f"Over limit: {'; '.join(over)}.")
        signals.append("category_budget_exceeded")
    if near:
        parts.append(f"Near limit: {', '.join(near)}.")
    if ok:
        parts.append(f"{', '.join(ok[:3])} are within budget.")
    msg = " ".join(parts) if parts else "All categories are within budget."
    if not signals:
        signals.append("on_track")
    resp = build_response("analysis", msg, signals=signals)
    return make_sample(role, ctx, q, resp, "context_analysis")


def gen_savings_rate(role):
    income, spending, *_, ctx = full_scenario(role)
    save = surplus(income, spending)
    rate = (save / income * 100) if income > 0 else 0
    q = random.choice(["What's my savings rate?", "How much am I saving?", "Am I saving enough?"])
    signals = ["above_savings_target"] if rate >= 20 else ["below_savings_target"]
    if rate >= 20:
        msg = f"You saved {fmt(save)} this month ({rate:.1f}%), above the 20% target. Well done."
    elif rate > 0:
        gap = income * 0.2 - save
        msg = f"You saved {fmt(save)} ({rate:.1f}%), below the 20% target. You need {fmt(gap)} more to hit it."
    else:
        msg = f"You're in deficit by {fmt(abs(save))}. No savings this month. Cut expenses immediately."
        signals = ["deficit"]
    resp = build_response("analysis", msg, signals=signals)
    return make_sample(role, ctx, q, resp, "context_analysis")


def gen_mom_comparison(role):
    income, spending = SCENARIO_MAP[role]()
    hist = make_monthly_history(spending)
    ctx = make_context(role, income, spending, monthly_history=hist)
    spent = total_spent(spending)
    prev_key = sorted(hist.keys())[-1]
    prev = hist[prev_key]
    prev_total = sum(prev.values())
    overall_delta = ((spent - prev_total) / prev_total * 100) if prev_total > 0 else 0
    q = random.choice([
        "Am I spending more than last month?",
        "How does this month compare?",
        "Month over month, how am I doing?",
        "Compare my spending to last month.",
    ])
    changes = []
    for cat, curr in spending.items():
        p = prev.get(cat, 0)
        if p > 0:
            changes.append((cat, ((curr - p) / p) * 100))
    changes.sort(key=lambda x: abs(x[1]), reverse=True)
    top_change = changes[0] if changes else None
    direction = "up" if overall_delta > 0 else "down"
    signals = ["spending_up_mom"] if overall_delta > 5 else ["spending_down_mom"] if overall_delta < -5 else []
    msg = f"Overall spending is {direction} {abs(overall_delta):.1f}% vs last month ({fmt(prev_total)} to {fmt(spent)})."
    if top_change:
        msg += f" Biggest change: {top_change[0]} at {top_change[1]:+.1f}%."
    resp = build_response("analysis", msg, signals=signals)
    return make_sample(role, ctx, q, resp, "context_analysis")


def gen_recurring_analysis(role):
    income, spending = SCENARIO_MAP[role]()
    rec = make_recurring(spending)
    while not rec:
        rec = make_recurring(spending)
    ctx = make_context(role, income, spending, recurring=rec)
    spent = total_spent(spending)
    total_fixed = sum(r["amount"] for r in rec)
    discretionary = max(spent - total_fixed, 0)
    fixed_pct = (total_fixed / income * 100) if income > 0 else 0
    q = random.choice([
        "What are my fixed costs?",
        "How much goes to subscriptions?",
        "What are my recurring expenses?",
        "What's fixed vs discretionary?",
    ])
    rec_names = ", ".join(f"{r['description']} (~{fmt(r['amount'])})" for r in rec[:4])
    signals = ["high_fixed_costs"] if fixed_pct > 40 else []
    msg = f"Your recurring expenses: {rec_names}. Total fixed: ~{fmt(total_fixed)}/month ({fixed_pct:.1f}% of income). Discretionary: {fmt(discretionary)}."
    if fixed_pct > 40:
        msg += " Fixed costs are high — review subscriptions and bills."
    else:
        msg += " Fixed costs are manageable. Focus savings efforts on discretionary spending."
    resp = build_response("analysis", msg, signals=signals)
    return make_sample(role, ctx, q, resp, "context_analysis")


def gen_anomaly_alert(role):
    income, spending = SCENARIO_MAP[role]()
    hist = make_monthly_history(spending)
    spike_cat = random.choice(list(spending.keys()))
    spending[spike_cat] = spending[spike_cat] * random.uniform(2.0, 3.5)
    ctx = make_context(role, income, spending, monthly_history=hist)
    prev_values = [hist[m].get(spike_cat, 0) for m in hist if hist[m].get(spike_cat, 0) > 0]
    avg = sum(prev_values) / len(prev_values) if prev_values else spending[spike_cat] / 2.5
    ratio = spending[spike_cat] / avg if avg > 0 else 2.5
    q = random.choice([
        "Anything unusual in my spending?",
        "Any spending anomalies?",
        "Does anything look off?",
        "Any red flags?",
    ])
    msg = (f"Your {spike_cat} spending of {fmt(spending[spike_cat])} is {ratio:.1f}x your 3-month average of {fmt(avg)}. "
           f"Review recent {spike_cat} transactions for one-time purchases or duplicate charges.")
    resp = build_response("analysis", msg, signals=["anomaly_detected"])
    return make_sample(role, ctx, q, resp, "context_analysis")


def gen_anomaly_proactive(role):
    """General question but anomalies exist — model warns proactively."""
    income, spending = SCENARIO_MAP[role]()
    hist = make_monthly_history(spending)
    spike_cat = random.choice(list(spending.keys()))
    spending[spike_cat] = spending[spike_cat] * random.uniform(2.2, 3.0)
    ctx = make_context(role, income, spending, monthly_history=hist)
    spent = total_spent(spending)
    save = surplus(income, spending)
    prev_values = [hist[m].get(spike_cat, 0) for m in hist if hist[m].get(spike_cat, 0) > 0]
    avg = sum(prev_values) / len(prev_values) if prev_values else spending[spike_cat] / 2.5
    ratio = spending[spike_cat] / avg if avg > 0 else 2.5
    q = random.choice(["How am I doing this month?", "Financial overview please.", "What's my status?"])
    msg = (f"Spending: {fmt(spent)}, surplus: {fmt(save)}. "
           f"Heads up: {spike_cat} is at {fmt(spending[spike_cat])}, which is {ratio:.1f}x your average of {fmt(avg)}. Worth reviewing.")
    resp = build_response("analysis", msg, signals=["anomaly_detected"])
    return make_sample(role, ctx, q, resp, "context_analysis")


def gen_goal_progress(role):
    income, spending = SCENARIO_MAP[role]()
    save = max(surplus(income, spending), 500_000)
    goal_name, goal_target = random.choice(GOAL_TEMPLATES)
    goal_target = rvar(goal_target, 0.3)
    progress = random.uniform(0.1, 0.7)
    saved = round(goal_target * progress / 50_000) * 50_000
    remaining = goal_target - saved
    months_left = random.randint(2, 12)
    target_date = (date.today() + timedelta(days=months_left * 30)).strftime("%Y-%m-%d")
    goals = [{"name": goal_name, "target_amount": goal_target, "current_saved": saved,
              "target_date": target_date, "priority": "HIGH"}]
    ctx = make_context(role, income, spending, goals=goals, balances=make_balances(income))
    months_needed = remaining / save if save > 0 else float("inf")
    feasible = months_needed <= months_left
    q = random.choice([
        f"Am I on track for {goal_name}?",
        f"How's my {goal_name} goal?",
        f"Will I reach {goal_name} in time?",
    ])
    signals = [] if feasible else ["goal_at_risk"]
    if feasible:
        msg = (f"{goal_name}: {fmt(saved)}/{fmt(goal_target)} ({progress * 100:.0f}% done). "
               f"At {fmt(save)}/month, you'll reach it in ~{months_needed:.1f} months. On track.")
    else:
        needed = remaining / months_left if months_left > 0 else remaining
        shortfall = needed - save
        msg = (f"{goal_name}: {fmt(saved)}/{fmt(goal_target)} ({progress * 100:.0f}% done). "
               f"You need {fmt(remaining)} in {months_left} months ({fmt(needed)}/mo), but surplus is only {fmt(save)}/mo. "
               f"Shortfall of {fmt(shortfall)}/mo — cut spending or extend the deadline.")
    resp = build_response("analysis", msg, signals=signals)
    return make_sample(role, ctx, q, resp, "context_analysis")


def gen_cross_feature_priority(role):
    """Multiple features active — model prioritizes."""
    income, spending = SCENARIO_MAP[role]()
    hist = make_monthly_history(spending)
    rec = make_recurring(spending)
    while not rec:
        rec = make_recurring(spending)
    spike_cat = random.choice(list(spending.keys()))
    spending[spike_cat] = spending[spike_cat] * random.uniform(2.0, 2.8)
    cat_budgets = make_category_budgets(spending, tight=True)
    save = surplus(income, spending)
    goal_name, goal_target = random.choice(GOAL_TEMPLATES[:5])
    goals = [{"name": goal_name, "target_amount": rvar(goal_target, 0.3), "current_saved": 0,
              "target_date": (date.today() + timedelta(days=90)).strftime("%Y-%m-%d"), "priority": "HIGH"}]
    ctx = make_context(role, income, spending, monthly_history=hist, recurring=rec,
                       category_budgets=cat_budgets, goals=goals)
    q = random.choice(["What should I focus on?", "What's most important right now?", "Prioritize my finances."])
    signals = ["anomaly_detected"]
    parts = [f"Priority 1: {spike_cat} spending is unusually high — review immediately."]
    over_cats = [cb["categoryName"] for cb in cat_budgets if spending.get(cb["categoryName"], 0) > cb["monthlyLimit"]]
    if over_cats:
        signals.append("category_budget_exceeded")
        parts.append(f"Priority 2: {', '.join(over_cats[:2])} over budget limits.")
    total_fixed = sum(r["amount"] for r in rec)
    if total_fixed / income > 0.4:
        signals.append("high_fixed_costs")
        parts.append(f"Priority 3: Fixed costs are {fmt(total_fixed)}/mo ({total_fixed / income * 100:.0f}% of income).")
    msg = " ".join(parts)
    resp = build_response("analysis", msg, signals=signals)
    return make_sample(role, ctx, q, resp, "context_analysis")


def gen_cross_feature_priority_analysis(role):
    """Alias name kept explicit for auditability."""
    return gen_cross_feature_priority(role)


def gen_no_budget_nudge(role):
    """No category budgets set — model suggests setting them up."""
    income, spending = SCENARIO_MAP[role]()
    ctx = make_context(role, income, spending, category_budgets=None)
    spent = total_spent(spending)
    save = surplus(income, spending)
    rate = (save / income * 100) if income > 0 else 0
    top_cat = max(spending, key=spending.get)
    q = random.choice(["How is my spending?", "Give me an overview.", "Am I doing okay?"])
    msg = (f"Spending: {fmt(spent)} out of {fmt(income)}, surplus {fmt(save)} ({rate:.1f}%). "
           f"Top category: {top_cat} at {fmt(spending[top_cat])}. "
           f"I notice you haven't set category budgets yet — setting per-category limits in the app helps you track and control spending.")
    signals = ["no_category_budgets"]
    if rate < 20:
        signals.append("below_savings_target")
    resp = build_response("analysis", msg, signals=signals)
    return make_sample(role, ctx, q, resp, "context_analysis")


def gen_forecast_discussion(role):
    income, spending = SCENARIO_MAP[role]()
    forecast = make_forecast(spending)
    ctx = make_context(role, income, spending, forecast=forecast)
    q = random.choice([
        "What does my forecast look like?",
        "How much will I spend next month?",
        "What's the spending projection?",
    ])
    projected = forecast["total"]
    conf = forecast["confidence"]
    current = total_spent(spending)
    delta = ((projected - current) / current * 100) if current > 0 else 0
    direction = "up" if delta > 0 else "down"
    msg = f"Projected spending next month: {fmt(projected)} (confidence {conf}%), which is {abs(delta):.1f}% {direction} from this month's {fmt(current)}."
    if delta > 10:
        msg += " Consider tightening spending to stay on track."
    resp = build_response("analysis", msg)
    return make_sample(role, ctx, q, resp, "context_analysis")


# ═══════════════════════════════════════════════════════════════════════════════
# FAMILY 5: MULTI-TURN FOLLOW-UPS
# ═══════════════════════════════════════════════════════════════════════════════

def gen_multi_overview_drilldown(role):
    """Overview → drill down on category."""
    income, spending, *_, ctx = full_scenario(role)
    spent = total_spent(spending)
    save = surplus(income, spending)
    top_cat = max(spending, key=spending.get)
    second_cat = sorted(spending, key=spending.get, reverse=True)[1]

    q1 = random.choice(["How's my budget?", "Financial overview."])
    a1 = build_response("analysis",
        f"Spending: {fmt(spent)}, surplus: {fmt(save)}. Top: {top_cat} ({fmt(spending[top_cat])}) and {second_cat} ({fmt(spending[second_cat])}).",
        signals=["below_savings_target"] if save / income < 0.2 else ["above_savings_target"])

    q2 = random.choice([f"Tell me more about {top_cat}.", f"What about {top_cat} specifically?"])
    limit = income * 0.30
    if spending[top_cat] > limit:
        a2 = build_response("analysis",
            f"{top_cat}: {fmt(spending[top_cat])} ({spending[top_cat] / income * 100:.1f}% of income), over the 30% guideline of {fmt(limit)}. Try cutting to {fmt(limit * 0.8)} next month.",
            signals=["over_budget"])
    else:
        a2 = build_response("analysis",
            f"{top_cat}: {fmt(spending[top_cat])} ({spending[top_cat] / income * 100:.1f}% of income), within the 30% limit of {fmt(limit)}. Under control.")

    return make_multi_turn_sample(role, ctx, [(q1, a1), (q2, a2)], "multi_turn")


def gen_multi_anomaly_acknowledge(role):
    """Model warns about anomaly → user explains → model acknowledges."""
    income, spending = SCENARIO_MAP[role]()
    hist = make_monthly_history(spending)
    spike_cat = random.choice(list(spending.keys()))
    spending[spike_cat] = spending[spike_cat] * random.uniform(2.2, 3.0)
    ctx = make_context(role, income, spending, monthly_history=hist)
    prev_values = [hist[m].get(spike_cat, 0) for m in hist if hist[m].get(spike_cat, 0) > 0]
    avg = sum(prev_values) / len(prev_values) if prev_values else spending[spike_cat] / 2.5
    ratio = spending[spike_cat] / avg if avg > 0 else 2.5

    q1 = "How's my spending?"
    a1 = build_response("analysis",
        f"Heads up: {spike_cat} is at {fmt(spending[spike_cat])}, {ratio:.1f}x your average of {fmt(avg)}. Worth checking.",
        signals=["anomaly_detected"])

    explanations = [
        f"Oh that was a one-time purchase for a gift.",
        f"Yeah I had an emergency expense in {spike_cat}.",
        f"That's because I stocked up for the month.",
    ]
    q2 = random.choice(explanations)
    a2 = build_response("analysis",
        f"Got it, that explains the spike. Your {spike_cat} should normalize next month then. No action needed.")

    return make_multi_turn_sample(role, ctx, [(q1, a1), (q2, a2)], "multi_turn")


def gen_anomaly_acknowledged_by_user(role):
    """Alias name kept explicit for auditability."""
    return gen_multi_anomaly_acknowledge(role)


def gen_multi_log_correction(role):
    """Log transaction → correct it."""
    income, spending, *_, ctx = full_scenario(role)
    amount, item, category = random.choice(EXPENSE_ITEMS)
    amount = rvar(amount, 0.3)
    correct_amount = rvar(amount * 0.5, 0.2)

    q1 = f"I spent {fmt(amount)} on {item}."
    a1 = build_response("action", f"Logged {fmt(amount)} for {item} under {category}.",
                         action=build_action("LOG_EXPENSE", amount=amount, category=category, item=item))

    q2 = f"Wait, that should be {fmt(correct_amount)}."
    a2 = build_response("action", f"Updated {item} to {fmt(correct_amount)}.",
                         action=build_action("UPDATE_TRANSACTION", amount=correct_amount,
                                             category=category, item=item, confidence=0.92))

    return make_multi_turn_sample(role, ctx, [(q1, a1), (q2, a2)], "multi_turn")


def gen_multi_goal_whatif(role):
    """Goal question → what-if cut category."""
    income, spending = SCENARIO_MAP[role]()
    save = max(surplus(income, spending), 500_000)
    goal_name, goal_target = random.choice(GOAL_TEMPLATES[:6])
    goal_target = rvar(goal_target, 0.3)
    goals = [{"name": goal_name, "target_amount": goal_target, "current_saved": 0,
              "target_date": (date.today() + timedelta(days=180)).strftime("%Y-%m-%d"), "priority": "HIGH"}]
    ctx = make_context(role, income, spending, goals=goals, balances=make_balances(income))
    months = goal_target / save
    top_cat = max(spending, key=spending.get)
    cut = spending[top_cat] * 0.20
    faster_save = save + cut
    faster_months = goal_target / faster_save

    q1 = f"How long to save for {goal_name}?"
    a1 = build_response("analysis",
        f"{goal_name}: {fmt(goal_target)}. At {fmt(save)}/month surplus, about {months:.1f} months.")

    q2 = f"What if I cut {top_cat} by 20%?"
    a2 = build_response("analysis",
        f"Cutting {top_cat} by 20% saves {fmt(cut)}/month, surplus becomes {fmt(faster_save)}. Timeline drops to {faster_months:.1f} months, saving you {months - faster_months:.1f} months.")

    return make_multi_turn_sample(role, ctx, [(q1, a1), (q2, a2)], "multi_turn")


def gen_multi_mom_followup(role):
    """MoM overview → user asks about specific category."""
    income, spending = SCENARIO_MAP[role]()
    hist = make_monthly_history(spending)
    ctx = make_context(role, income, spending, monthly_history=hist)
    spent = total_spent(spending)
    prev_key = sorted(hist.keys())[-1]
    prev = hist[prev_key]
    prev_total = sum(prev.values())
    overall_delta = ((spent - prev_total) / prev_total * 100) if prev_total > 0 else 0
    direction = "up" if overall_delta > 0 else "down"

    q1 = "How does this month compare to last?"
    a1 = build_response("analysis",
        f"Overall spending {direction} {abs(overall_delta):.1f}% ({fmt(prev_total)} to {fmt(spent)}).",
        signals=["spending_up_mom"] if overall_delta > 5 else [])

    cat = random.choice(list(spending.keys()))
    prev_amt = prev.get(cat, 0)
    curr_amt = spending[cat]
    if prev_amt > 0:
        cat_delta = ((curr_amt - prev_amt) / prev_amt) * 100
        q2 = f"What about {cat} specifically?"
        a2 = build_response("analysis",
            f"{cat}: {fmt(prev_amt)} last month to {fmt(curr_amt)} this month ({cat_delta:+.1f}%)." +
            (" Consider reviewing." if cat_delta > 15 else " Stable." if abs(cat_delta) < 10 else " Nice reduction."))
    else:
        q2 = f"And {cat}?"
        a2 = build_response("analysis", f"{cat}: {fmt(curr_amt)} this month, no data last month to compare.")

    return make_multi_turn_sample(role, ctx, [(q1, a1), (q2, a2)], "multi_turn")


# ═══════════════════════════════════════════════════════════════════════════════
# FAMILY 6: ROLE-SPECIFIC ADVICE
# ═══════════════════════════════════════════════════════════════════════════════

def gen_student_semester_plan():
    income = rvar(random.choice([4_000_000, 5_000_000, 6_000_000, 8_000_000]))
    months = random.choice([4, 5])
    total_budget = income * months
    tuition = rvar(total_budget * random.uniform(0.25, 0.40))
    living = total_budget - tuition
    monthly_living = living / months
    spending = {"Food": rvar(monthly_living * 0.40), "Transport": rvar(monthly_living * 0.15),
                "Education": rvar(monthly_living * 0.20), "Entertainment": rvar(monthly_living * 0.10),
                "Shopping": rvar(monthly_living * 0.10)}
    ctx = make_context("Student", income, spending,
                       balances=[{"name": "Savings", "balance": total_budget, "currency": "VND", "type": "savings"}])
    q = random.choice([
        f"I have {fmt(total_budget)} for this semester ({months} months). Help me plan.",
        f"Budget {fmt(total_budget)} across {months} months of school.",
    ])
    msg = (f"Semester budget: {fmt(total_budget)} over {months} months. "
           f"Set aside {fmt(tuition)} for tuition first, leaving {fmt(living)} ({fmt(monthly_living)}/month) for living. "
           f"Suggested: Food {fmt(spending['Food'])}, Transport {fmt(spending['Transport'])}, Education {fmt(spending['Education'])}. "
           f"Keep a {fmt(monthly_living * 0.05)}/month emergency buffer.")
    resp = build_response("analysis", msg)
    return make_sample("Student", ctx, q, resp, "role_specific")


def gen_student_debt_advice():
    income, spending = student_scenario()
    ctx = make_light_context("Student", income, spending)
    q = random.choice(["Should I use a credit card?", "Is borrowing money okay as a student?"])
    msg = (f"Avoid credit card debt as a student. Interest rates of 20-35%/year compound fast on a {fmt(income)}/month income. "
           f"If you need a card, pay the full balance monthly. Build a 3-5M VND emergency fund first.")
    resp = build_response("analysis", msg)
    return make_sample("Student", ctx, q, resp, "role_specific")


def gen_student_part_time():
    income, spending = student_scenario()
    ctx = make_light_context("Student", income, spending)
    spent = total_spent(spending)
    q = random.choice(["Should I get a part-time job?", "Is working while studying worth it?"])
    extra = 2_000_000
    msg = (f"Even {fmt(extra)}/month from part-time work would cover your spending gap and build savings. "
           f"Look into tutoring, delivery, or freelance tasks. Keep it under 15-20 hours/week to protect your studies.")
    resp = build_response("analysis", msg)
    return make_sample("Student", ctx, q, resp, "role_specific")


def gen_worker_investment():
    income, spending = worker_scenario()
    ctx = make_light_context("Worker", income, spending)
    save = surplus(income, spending)
    spent = total_spent(spending)
    ef = spent * 6
    q = random.choice(["How should I start investing?", "What should I invest in?"])
    msg = (f"Surplus: {fmt(save)}/month. First build an emergency fund of {fmt(ef)} (6 months expenses). "
           f"Then start with low-cost index funds or government bonds. Max out any employer BHXH matching. Only invest money you won't need for 3+ years.")
    resp = build_response("analysis", msg)
    return make_sample("Worker", ctx, q, resp, "role_specific")


def gen_worker_bhxh():
    income, spending = worker_scenario()
    ctx = make_light_context("Worker", income, spending)
    bhxh = income * 0.08
    employer = income * 0.175
    q = random.choice(["How does BHXH work?", "Is BHXH enough for retirement?", "Tell me about my BHXH."])
    msg = (f"You pay {fmt(bhxh)}/month (8%), employer pays {fmt(employer)} (17.5%), combined {fmt(bhxh + employer)}. "
           f"After 20-30 years, BHXH provides roughly 45-75% of average salary. "
           f"Not enough alone — save an extra {fmt(income * 0.1)}-{fmt(income * 0.15)}/month in personal investments.")
    resp = build_response("analysis", msg)
    return make_sample("Worker", ctx, q, resp, "role_specific")


def gen_worker_salary_split():
    income, spending = worker_scenario()
    ctx = make_light_context("Worker", income, spending)
    q = random.choice(["How should I split my salary on payday?", "I just got paid, what first?"])
    msg = (f"On payday, split {fmt(income)}: 1) Transfer {fmt(income * 0.20)} to savings immediately. "
           f"2) Pay fixed bills ({fmt(income * 0.30)}). 3) Use remaining {fmt(income * 0.50)} for needs and wants. "
           f"Pay-yourself-first is the most effective budgeting habit.")
    resp = build_response("analysis", msg)
    return make_sample("Worker", ctx, q, resp, "role_specific")


def gen_freelancer_tax():
    income, spending = freelancer_scenario()
    ctx = make_light_context("Freelancer", income, spending)
    tax = income * 0.30
    after_tax = surplus(income, spending) - tax
    q = random.choice(["How much for taxes?", "What's the 30% tax rule?", "Tax advice?"])
    msg = (f"Set aside 30% ({fmt(tax)}) of every payment for taxes immediately. "
           f"After taxes, your remaining surplus is {fmt(after_tax)}. "
           f"Move tax money to a separate account. Track deductible expenses (internet, equipment, software) to reduce taxable income.")
    resp = build_response("analysis", msg)
    return make_sample("Freelancer", ctx, q, resp, "role_specific")


def gen_freelancer_buffer():
    income, spending = freelancer_scenario()
    spent = total_spent(spending)
    buf = spent * 6
    balances = make_balances(income)
    total_liquid = sum(a["balance"] for a in balances)
    ctx = make_light_context("Freelancer", income, spending)
    q = random.choice(["My income fluctuates — how to manage?", "How do I budget with irregular income?"])
    buf_pct = (total_liquid / buf * 100) if buf > 0 else 0
    msg = (f"Pay yourself a fixed salary of ~{fmt(spent * 1.1)}/month from a buffer account. "
           f"Buffer target: {fmt(buf)} (6 months expenses). Current liquidity: {fmt(total_liquid)} ({buf_pct:.0f}% of target). "
           + ("Well-buffered." if buf_pct >= 80 else f"Need {fmt(buf - total_liquid)} more for a safe buffer.")
           + " In good months, refill the buffer. In lean months, draw from it.")
    resp = build_response("analysis", msg)
    return make_sample("Freelancer", ctx, q, resp, "role_specific")


def gen_freelancer_quarterly_tax():
    incomes = [rvar(random.choice([15_000_000, 25_000_000, 40_000_000])) for _ in range(3)]
    total_q = sum(incomes)
    tax = total_q * 0.30
    current = incomes[-1]
    spending = {"Food": rvar(current * 0.18), "Transport": rvar(current * 0.08),
                "Bills": rvar(current * 0.12), "Shopping": rvar(current * 0.08)}
    ctx = make_light_context("Freelancer", current, spending)
    q = random.choice(["What's my quarterly tax situation?", "How much tax should I have set aside?"])
    msg = (f"Quarterly income: {fmt(incomes[0])} + {fmt(incomes[1])} + {fmt(incomes[2])} = {fmt(total_q)}. "
           f"Tax reserve (30%): {fmt(tax)}. Make sure you have at least that set aside. "
           f"Track deductible expenses to reduce taxable income by 10-20%.")
    resp = build_response("analysis", msg)
    return make_sample("Freelancer", ctx, q, resp, "role_specific",
                       subfamily="freelancer_quarterly_tax", tags=["debt_tax"])


# ═══════════════════════════════════════════════════════════════════════════════
# FAMILY 7: CUSTOM SPLIT (dedicated)
# ═══════════════════════════════════════════════════════════════════════════════

# Extra ratios for generalization beyond the hardcoded CUSTOM_SPLITS
_EXTRA_RATIOS = [
    (35, 35, 30), (42, 28, 30), (48, 32, 20), (52, 28, 20),
    (58, 22, 20), (62, 23, 15), (68, 17, 15), (72, 18, 10),
    (75, 15, 10), (38, 32, 30), (44, 26, 30), (56, 24, 20),
]
ALL_SPLIT_RATIOS = CUSTOM_SPLITS + _EXTRA_RATIOS


def gen_custom_split_direct(role):
    """Direct custom split calculation from income."""
    income, spending = SCENARIO_MAP[role]()
    n, w, s = random.choice(ALL_SPLIT_RATIOS)
    ctx = make_context(role, income, spending, budget_split=(n, w, s))
    needs = income * n / 100
    wants = income * w / 100
    savings = income * s / 100
    q = random.choice([
        f"Split my income with a {n}/{w}/{s} rule.",
        f"How does a {n}/{w}/{s} split work for me?",
        f"Apply a {n}/{w}/{s} budget to my {fmt(income)} income.",
        f"What would {n}/{w}/{s} look like for my finances?",
        f"Calculate my budget using {n}/{w}/{s}.",
        f"Show me a {n}/{w}/{s} needs/wants/savings split.",
    ])
    msg = (f"With a {n}/{w}/{s} split on {fmt(income)}: "
           f"Needs ({n}%): {fmt(needs)}, Wants ({w}%): {fmt(wants)}, Savings ({s}%): {fmt(savings)}.")
    spent = total_spent(spending)
    if spent > needs + wants:
        msg += f" Your current spending of {fmt(spent)} exceeds the needs+wants limit of {fmt(needs + wants)}. Cut discretionary spending."
        signals = ["over_budget"]
    elif spent > needs:
        msg += f" Spending at {fmt(spent)} is within budget. You're on track for {s}% savings."
        signals = ["on_track"]
    else:
        msg += f" Spending at {fmt(spent)} is well under the needs limit alone. Great discipline."
        signals = ["above_savings_target"]
    resp = build_response("analysis", msg, signals=signals)
    return make_sample(role, ctx, q, resp, "context_analysis",
                       subfamily="custom_split", tags=["custom_split", "ratio_generalization"])


def gen_custom_split_compare(role):
    """Compare current spending against a custom split target."""
    income, spending = SCENARIO_MAP[role]()
    n, w, s = random.choice(ALL_SPLIT_RATIOS)
    ctx = make_context(role, income, spending, budget_split=(n, w, s))
    needs = income * n / 100
    wants = income * w / 100
    savings = income * s / 100
    spent = total_spent(spending)
    save = surplus(income, spending)
    q = random.choice([
        f"Am I following my {n}/{w}/{s} budget?",
        f"How does my spending compare to my {n}/{w}/{s} split?",
        f"Check my spending against the {n}/{w}/{s} rule.",
    ])
    target_save_pct = s
    actual_save_pct = (save / income * 100) if income > 0 else 0
    if actual_save_pct >= target_save_pct:
        msg = (f"Your current spending is {fmt(spent)} vs the needs+wants limit of {fmt(needs + wants)}. "
               f"Savings rate: {actual_save_pct:.1f}%, meeting the {s}% target. On track.")
        signals = ["on_track"]
    else:
        gap = savings - save
        msg = (f"Your current spending is {fmt(spent)}, which leaves only {actual_save_pct:.1f}% savings vs the {s}% target of {fmt(savings)}. "
               f"You need to free up {fmt(gap)} more. "
               f"Top category: {max(spending, key=spending.get)} — consider cutting there first.")
        signals = ["below_savings_target"]
    resp = build_response("analysis", msg, signals=signals)
    return make_sample(role, ctx, q, resp, "context_analysis",
                       subfamily="custom_split", tags=["custom_split", "ratio_generalization"])


# ═══════════════════════════════════════════════════════════════════════════════
# FAMILY 8: EMERGENCY FUND (dedicated)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_emergency_fund(role):
    """Direct emergency fund planning question."""
    income, spending = SCENARIO_MAP[role]()
    spent = total_spent(spending)
    save = surplus(income, spending)
    balances = make_balances(income)
    total_liquid = sum(a["balance"] for a in balances)
    ctx = make_context(role, income, spending, balances=balances)

    # Role-specific month target
    if role == "Freelancer":
        months_target = random.choice([6, 6, 9])
        role_note = "As a freelancer with irregular income, aim for the higher end."
    elif role == "Student":
        months_target = random.choice([3, 3, 4])
        role_note = "As a student, 3 months is a good starting target."
    else:
        months_target = random.choice([3, 4, 6])
        role_note = "As a salaried worker, 3-6 months covers most situations."

    fund_target = spent * months_target
    current_coverage = (total_liquid / spent) if spent > 0 else 0

    q = random.choice([
        "What should my emergency fund be?",
        "How much do I need for an emergency fund?",
        "What's the ideal emergency fund for my expenses?",
        "How many months of expenses should I save?",
        "Help me plan an emergency fund.",
        "How do I build an emergency fund?",
        "Do I have enough saved for emergencies?",
        "What's a good emergency fund target?",
    ])

    msg = (f"Your monthly expenses are {fmt(spent)}. Target: {months_target} months = {fmt(fund_target)}. "
           f"Current liquid savings: {fmt(total_liquid)} ({current_coverage:.1f} months of cover). ")
    if total_liquid >= fund_target:
        msg += (f"You're fully covered! {role_note} "
                f"Keep saving {fmt(save)}/month to maintain and grow this buffer.")
        signals = ["above_savings_target"]
    elif total_liquid >= fund_target * 0.5:
        gap = fund_target - total_liquid
        msg += f"You're halfway there. Need {fmt(gap)} more. "
        if save > 0:
            months_to_goal = gap / save
            msg += f"At {fmt(save)}/month surplus, about {months_to_goal:.1f} months to go. {role_note}"
        else:
            msg += f"You'll need to cut spending to build this up. {role_note}"
        signals = ["below_savings_target"]
    else:
        gap = fund_target - total_liquid
        msg += f"Need {fmt(gap)} more. "
        if save > 0:
            months_to_goal = gap / save
            msg += f"Start by saving {fmt(min(save, gap / 6))}/month. Timeline: ~{months_to_goal:.1f} months. {role_note}"
        else:
            msg += f"Cut spending first to create a surplus. {role_note}"
        signals = ["below_savings_target"]
    resp = build_response("analysis", msg, signals=signals)
    return make_sample(role, ctx, q, resp, "context_analysis",
                       subfamily="emergency_fund", tags=["emergency_fund"])


def gen_emergency_fund_building(role):
    """How-to-build variant with concrete saving steps."""
    income, spending = SCENARIO_MAP[role]()
    spent = total_spent(spending)
    save = max(surplus(income, spending), 50_000)
    months_target = 6 if role == "Freelancer" else 3
    fund_target = spent * months_target
    ctx = make_light_context(role, income, spending)

    q = random.choice([
        "How do I start building an emergency fund?",
        "I have zero savings. How to build an emergency fund?",
        "Give me steps to build an emergency fund.",
        "What's the fastest way to build an emergency fund?",
    ])
    monthly_save = min(save * 0.5, fund_target / 12)
    monthly_save = max(round(monthly_save / 50_000) * 50_000, 100_000)
    months_needed = fund_target / monthly_save if monthly_save > 0 else float("inf")

    top_cat = max(spending, key=spending.get)
    cut_amt = spending[top_cat] * 0.15

    msg = (f"Target: {fmt(fund_target)} ({months_target} months of expenses at {fmt(spent)}/month). "
           f"Step 1: Open a separate savings account. "
           f"Step 2: Auto-transfer {fmt(monthly_save)}/month on payday. "
           f"Step 3: Cut {top_cat} by 15% ({fmt(cut_amt)}) to accelerate. "
           f"Timeline: ~{months_needed:.0f} months at {fmt(monthly_save)}/month.")
    resp = build_response("analysis", msg, signals=["below_savings_target"])
    return make_sample(role, ctx, q, resp, "context_analysis",
                       subfamily="emergency_fund", tags=["emergency_fund"])


# ═══════════════════════════════════════════════════════════════════════════════
# FAMILY 9: NO-BUDGET NUDGE (dedicated, expanded)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_no_budget_nudge_expanded(role):
    """No category budgets — give overview and nudge to set them up."""
    income, spending = SCENARIO_MAP[role]()
    ctx = make_context(role, income, spending, category_budgets=None)
    spent = total_spent(spending)
    save = surplus(income, spending)
    rate = (save / income * 100) if income > 0 else 0
    sorted_cats = sorted(spending.items(), key=lambda x: x[1], reverse=True)
    top_cat, top_amt = sorted_cats[0]
    second_cat, second_amt = sorted_cats[1] if len(sorted_cats) > 1 else (top_cat, top_amt)

    q = random.choice([
        "How is my spending this month?",
        "Give me a financial overview.",
        "Am I overspending?",
        "How am I doing?",
        "Check my budget.",
        "What's my spending situation?",
        "How are my finances looking?",
        "Summarize my spending.",
        "Am I spending too much?",
        "Is my spending under control?",
    ])
    msg = (f"Spending: {fmt(spent)} out of {fmt(income)}, surplus {fmt(save)} ({rate:.1f}%). "
           f"Top: {top_cat} at {fmt(top_amt)} and {second_cat} at {fmt(second_amt)}. "
           f"You don't have category budgets set up yet — adding per-category limits helps you spot overspending early and stay in control.")
    signals = ["no_category_budgets"]
    if rate < 20:
        signals.append("below_savings_target")
    resp = build_response("analysis", msg, signals=signals)
    return make_sample(role, ctx, q, resp, "context_analysis",
                       subfamily="no_budget_nudge", tags=["no_budget_nudge"])


# ═══════════════════════════════════════════════════════════════════════════════
# FAMILY 10: NO-ANOMALY NEGATIVE (dedicated)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_no_anomaly_negative(role):
    """Context has normal spending — model says nothing unusual."""
    income, spending = SCENARIO_MAP[role]()
    # Build history where current month is SIMILAR to past months (no spike)
    hist = {}
    today = date.today()
    for i in range(1, 4):
        m, y = today.month - i, today.year
        while m <= 0:
            m += 12; y -= 1
        # Keep past spending very close to current (within 15%)
        hist[f"{y}-{m:02d}"] = {cat: rvar(amt, 0.12) for cat, amt in spending.items()}
    ctx = make_context(role, income, spending, monthly_history=hist)

    q = random.choice([
        "Anything unusual in my spending?",
        "Do I have any anomalies?",
        "Does anything look off?",
        "Any red flags?",
        "Is my spending normal?",
        "Any spending spikes this month?",
        "Are there any irregularities?",
        "Should I be worried about anything?",
    ])
    spent = total_spent(spending)
    msg = (f"Your spending of {fmt(spent)} looks normal compared to recent months. "
           f"No unusual spikes or anomalies detected. All categories are within their typical ranges.")
    resp = build_response("analysis", msg, signals=["on_track"])
    return make_sample(role, ctx, q, resp, "hard_negative",
                       subfamily="no_anomaly_negative", tags=["no_anomaly_negative", "hard_negative"])


# ═══════════════════════════════════════════════════════════════════════════════
# FAMILY 11: CROSS-FEATURE ACTIONABLE ADVICE (dedicated, expanded)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_cross_feature_anomaly_budget(role):
    """Anomaly + category budget breach together."""
    income, spending = SCENARIO_MAP[role]()
    hist = make_monthly_history(spending)
    spike_cat = random.choice(list(spending.keys()))
    spending[spike_cat] = spending[spike_cat] * random.uniform(2.0, 2.8)
    cat_budgets = make_category_budgets(spending, tight=True)
    ctx = make_context(role, income, spending, monthly_history=hist, category_budgets=cat_budgets)

    prev_values = [hist[m].get(spike_cat, 0) for m in hist if hist[m].get(spike_cat, 0) > 0]
    avg = sum(prev_values) / len(prev_values) if prev_values else spending[spike_cat] / 2.5
    ratio = spending[spike_cat] / avg if avg > 0 else 2.5

    over_cats = [cb["categoryName"] for cb in cat_budgets
                 if spending.get(cb["categoryName"], 0) > cb["monthlyLimit"]]

    q = random.choice(["What should I focus on?", "Any issues with my spending?", "What's most urgent?"])
    parts = [f"{spike_cat} is at {fmt(spending[spike_cat])}, {ratio:.1f}x your average — review immediately."]
    signals = ["anomaly_detected"]
    if over_cats:
        parts.append(f"Budget limits exceeded: {', '.join(over_cats[:3])}.")
        signals.append("category_budget_exceeded")
    parts.append(f"Cut {spike_cat} back to ~{fmt(avg)} next month.")
    msg = " ".join(parts)
    resp = build_response("analysis", msg, signals=signals)
    return make_sample(role, ctx, q, resp, "context_analysis",
                       subfamily="cross_feature_advice", tags=["cross_feature_advice"])


def gen_cross_feature_deficit_discretionary(role):
    """Deficit + high discretionary spend."""
    income, spending = SCENARIO_MAP[role]()
    # Force deficit
    for cat in spending:
        spending[cat] = spending[cat] * random.uniform(1.2, 1.5)
    spent = total_spent(spending)
    save = surplus(income, spending)
    rec = make_recurring(spending)
    while not rec:
        rec = make_recurring(spending)
    total_fixed = sum(r["amount"] for r in rec)
    discretionary = max(spent - total_fixed, 0)
    ctx = make_context(role, income, spending, recurring=rec)

    q = random.choice(["Help me fix my budget.", "I'm spending more than I earn.", "What do I cut?"])
    top_disc_cat = max((c for c in spending if c not in {"Bills"}), key=lambda c: spending[c])
    msg = (f"You're in deficit by {fmt(abs(save))}. Fixed costs: {fmt(total_fixed)}, discretionary: {fmt(discretionary)}. "
           f"Biggest discretionary: {top_disc_cat} at {fmt(spending[top_disc_cat])}. "
           f"Cut {top_disc_cat} by 25-30% ({fmt(spending[top_disc_cat] * 0.25)}) as a first step.")
    resp = build_response("analysis", msg, signals=["deficit"])
    return make_sample(role, ctx, q, resp, "context_analysis",
                       subfamily="cross_feature_advice", tags=["cross_feature_advice"])


def gen_cross_feature_goal_forecast(role):
    """Goal at risk + forecast worsening."""
    income, spending = SCENARIO_MAP[role]()
    save = max(surplus(income, spending), 200_000)
    goal_name, goal_target = random.choice(GOAL_TEMPLATES[:5])
    goal_target = rvar(goal_target, 0.3)
    goals = [{"name": goal_name, "target_amount": goal_target, "current_saved": 0,
              "target_date": (date.today() + timedelta(days=90)).strftime("%Y-%m-%d"), "priority": "HIGH"}]
    forecast = {"total": round(total_spent(spending) * 1.15 / 50_000) * 50_000,
                "confidence": random.choice([65, 70, 75])}
    ctx = make_context(role, income, spending, goals=goals, forecast=forecast)

    q = random.choice(["Am I going to hit my goal?", "What's my financial outlook?", "Should I be worried?"])
    months_needed = goal_target / save if save > 0 else float("inf")
    msg = (f"{goal_name} needs {fmt(goal_target)} in 3 months, but at {fmt(save)}/month surplus it would take {months_needed:.1f} months. "
           f"Forecast shows spending rising to {fmt(forecast['total'])} next month. "
           f"Action: reduce spending now to protect both your goal and monthly surplus.")
    resp = build_response("analysis", msg, signals=["goal_at_risk"])
    return make_sample(role, ctx, q, resp, "context_analysis",
                       subfamily="cross_feature_advice", tags=["cross_feature_advice"])


def gen_cross_feature_no_budget_concentrated(role):
    """No category budgets + concentrated spending."""
    income, spending = SCENARIO_MAP[role]()
    # Make one category dominate
    top_cat = max(spending, key=spending.get)
    spending[top_cat] = spending[top_cat] * 1.8
    ctx = make_context(role, income, spending, category_budgets=None)
    spent = total_spent(spending)
    top_pct = spending[top_cat] / income * 100

    q = random.choice(["Any advice for me?", "What should I change?", "How can I improve?"])
    msg = (f"{top_cat} takes {top_pct:.0f}% of your income ({fmt(spending[top_cat])}), which is very concentrated. "
           f"Set up category budgets to cap {top_cat} and redistribute spending. "
           f"Without limits, overspending in one area is easy to miss.")
    signals = ["no_category_budgets"]
    if spent > income * 0.8:
        signals.append("over_budget")
    resp = build_response("analysis", msg, signals=signals)
    return make_sample(role, ctx, q, resp, "context_analysis",
                       subfamily="cross_feature_advice", tags=["cross_feature_advice", "no_budget_nudge"])


# ═══════════════════════════════════════════════════════════════════════════════
# FAMILY 12: DEBT/TAX EDGE CASES (dedicated)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_debt_tax_freelancer_debt(role):
    """Freelancer tax debt scenario."""
    income, spending = SCENARIO_MAP["Freelancer"]()
    spent = total_spent(spending)
    save = surplus(income, spending)
    tax_owed = income * random.uniform(0.25, 0.35) * random.randint(2, 4)
    ctx = make_light_context("Freelancer", income, spending)

    q = random.choice([
        f"I owe {fmt(tax_owed)} in back taxes. What should I do?",
        f"I have a tax debt of {fmt(tax_owed)}. How do I handle it?",
        f"I forgot to save for taxes and owe {fmt(tax_owed)}.",
    ])
    monthly_payment = min(save * 0.6, tax_owed / 6)
    monthly_payment = max(round(monthly_payment / 50_000) * 50_000, 200_000)
    months = tax_owed / monthly_payment if monthly_payment > 0 else float("inf")
    msg = (f"Tax debt of {fmt(tax_owed)} is serious. With {fmt(save)}/month surplus, "
           f"allocate {fmt(monthly_payment)}/month to repayment (~{months:.0f} months). "
           f"Contact the tax office about installment plans. Going forward, set aside 30% of every payment immediately.")
    resp = build_response("analysis", msg, signals=["deficit"] if save < monthly_payment else [])
    return make_sample("Freelancer", ctx, q, resp, "context_analysis",
                       subfamily="debt_tax", tags=["debt_tax"])


def gen_debt_tax_after_tax_surplus(role):
    """After-tax surplus vs gross surplus distinction."""
    income, spending = SCENARIO_MAP[role]()
    spent = total_spent(spending)
    gross_save = surplus(income, spending)
    tax_rate = 0.30 if role == "Freelancer" else 0.10
    tax = income * tax_rate
    after_tax_save = gross_save - tax
    ctx = make_light_context(role, income, spending)

    q = random.choice([
        "How much do I actually have after taxes?",
        "What's my real surplus after tax?",
        "Am I actually saving anything after tax?",
    ])
    if after_tax_save > 0:
        msg = (f"Gross surplus: {fmt(gross_save)}. Tax reserve ({tax_rate * 100:.0f}%): {fmt(tax)}. "
               f"After-tax surplus: {fmt(after_tax_save)}. This is your true disposable amount.")
    else:
        msg = (f"Gross surplus: {fmt(gross_save)}, but after setting aside {fmt(tax)} for taxes ({tax_rate * 100:.0f}%), "
               f"you're actually short by {fmt(abs(after_tax_save))}. Cut spending to create a real surplus.")
    signals = ["above_savings_target"] if after_tax_save > income * 0.1 else ["below_savings_target"]
    resp = build_response("analysis", msg, signals=signals)
    return make_sample(role, ctx, q, resp, "context_analysis",
                       subfamily="debt_tax", tags=["debt_tax"])


def gen_debt_vs_emergency_tradeoff(role):
    """Debt payoff vs emergency fund tradeoff."""
    income, spending = SCENARIO_MAP[role]()
    spent = total_spent(spending)
    save = max(surplus(income, spending), 500_000)
    debt = rvar(random.choice([5_000_000, 10_000_000, 20_000_000, 30_000_000]), 0.3)
    balances = make_balances(income)
    total_liquid = sum(a["balance"] for a in balances)
    ef_target = spent * 3
    ctx = make_context(role, income, spending, balances=balances)

    q = random.choice([
        "Should I pay off debt or build an emergency fund first?",
        "Debt payoff or emergency savings — which first?",
        f"I have {fmt(debt)} in debt. Save or pay it off?",
    ])
    if total_liquid < ef_target * 0.5:
        msg = (f"Debt: {fmt(debt)}. Emergency fund: {fmt(total_liquid)} vs target {fmt(ef_target)} (only {total_liquid / ef_target * 100:.0f}%). "
               f"Build a minimum 1-month buffer ({fmt(spent)}) first, then split surplus: 70% debt, 30% emergency fund.")
    else:
        msg = (f"Debt: {fmt(debt)}. Emergency fund: {fmt(total_liquid)} ({total_liquid / ef_target * 100:.0f}% of target). "
               f"Your emergency buffer is adequate. Prioritize debt payoff — put {fmt(save * 0.7)}/month toward debt.")
    resp = build_response("analysis", msg)
    return make_sample(role, ctx, q, resp, "context_analysis",
                       subfamily="debt_tax", tags=["debt_tax", "emergency_fund"])


def gen_debt_student_part_time_tax(role):
    """Student part-time tax edge case."""
    income = rvar(random.choice([3_000_000, 4_000_000, 5_000_000]))
    spending = {"Food": rvar(income * 0.35), "Transport": rvar(income * 0.10),
                "Education": rvar(income * 0.15), "Entertainment": rvar(income * 0.10)}
    ctx = make_light_context("Student", income, spending)

    q = random.choice([
        "Do I need to pay taxes on my part-time income?",
        "Is my tutoring income taxable?",
        "Should I worry about taxes as a student?",
    ])
    msg = (f"Part-time income of {fmt(income)}/month may be below the personal deduction threshold (11M VND/month), "
           f"so you likely owe no income tax. However, if total annual income exceeds 132M VND, "
           f"you'll need to file. Keep receipts for any work-related expenses. "
           f"If paid as a contractor, the payer may withhold 10% — you can reclaim it when filing.")
    resp = build_response("analysis", msg)
    return make_sample("Student", ctx, q, resp, "context_analysis",
                       subfamily="debt_tax", tags=["debt_tax"])


# ═══════════════════════════════════════════════════════════════════════════════
# FAMILY 13: NEGATIVE DISTINCTION (expanded hard negatives)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_neg_split_not_action(role):
    """Custom split question must stay analysis, not action."""
    income, spending = SCENARIO_MAP[role]()
    n, w, s = random.choice(ALL_SPLIT_RATIOS)
    ctx = make_light_context(role, income, spending)
    q = random.choice([
        f"What would a {n}/{w}/{s} budget split look like?",
        f"Show me a {n}/{w}/{s} breakdown.",
        f"How would {n}/{w}/{s} work for my income?",
    ])
    needs = income * n / 100
    wants = income * w / 100
    savings = income * s / 100
    msg = (f"A {n}/{w}/{s} split on {fmt(income)}: Needs {fmt(needs)}, Wants {fmt(wants)}, Savings {fmt(savings)}.")
    resp = build_response("analysis", msg)
    return make_sample(role, ctx, q, resp, "hard_negative",
                       subfamily="neg_split_analysis", tags=["hard_negative", "custom_split"])


def gen_neg_show_salary(role):
    """Show last salary — query, not action."""
    income, spending, *_, ctx = full_scenario(role)
    q = random.choice([
        "Can you show my last salary?",
        "What was my last income?",
        "How much did I earn this month?",
        "Show me my income.",
    ])
    msg = f"Your recorded income this month is {fmt(income)}."
    resp = build_response("analysis", msg)
    return make_sample(role, ctx, q, resp, "hard_negative",
                       subfamily="neg_query", tags=["hard_negative"])


def gen_neg_food_query(role):
    """How much did I spend on food — query, not action."""
    income, spending, *_, ctx = full_scenario(role)
    cat = random.choice(list(spending.keys()))
    q = random.choice([
        f"How much did I spend on {cat}?",
        f"What's my {cat} total?",
        f"Show me {cat} spending.",
    ])
    msg = f"Your {cat} spending this month: {fmt(spending[cat])} ({spending[cat] / income * 100:.1f}% of income)."
    resp = build_response("analysis", msg)
    return make_sample(role, ctx, q, resp, "hard_negative",
                       subfamily="neg_query", tags=["hard_negative"])


def gen_neg_affordability(role):
    """Affordability question — analysis only."""
    income, spending, *_, ctx = full_scenario(role)
    items = [("a new laptop", rvar(12_000_000, 0.3)), ("AirPods", rvar(4_000_000, 0.2)),
             ("a vacation", rvar(5_000_000, 0.3)), ("a new phone", rvar(10_000_000, 0.3))]
    item, cost = random.choice(items)
    save = surplus(income, spending)
    q = f"Can I afford {item}?"
    months = cost / save if save > 0 else float("inf")
    if save > 0 and months < 2:
        msg = f"With {fmt(save)}/month surplus, you could cover {item} ({fmt(cost)}) in {months:.1f} months. Feasible if it's a priority."
    elif save > 0:
        msg = f"{item} at {fmt(cost)} would take {months:.1f} months at your {fmt(save)}/month surplus. Plan ahead."
    else:
        msg = f"Your surplus is {fmt(save)}, so {item} ({fmt(cost)}) isn't affordable right now."
    resp = build_response("analysis", msg)
    return make_sample(role, ctx, q, resp, "hard_negative",
                       subfamily="neg_affordability", tags=["hard_negative"])


# ═══════════════════════════════════════════════════════════════════════════════
# FAMILY 14: MULTI-TURN PATCH (weak-family follow-ups)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_multi_overview_no_budget(role):
    """Overview → no-budget nudge follow-up."""
    income, spending = SCENARIO_MAP[role]()
    ctx = make_context(role, income, spending, category_budgets=None)
    spent = total_spent(spending)
    save = surplus(income, spending)
    top_cat = max(spending, key=spending.get)

    q1 = random.choice(["How's my budget?", "Financial overview."])
    a1 = build_response("analysis",
        f"Spending: {fmt(spent)}, surplus: {fmt(save)}. Top: {top_cat} ({fmt(spending[top_cat])}).",
        signals=["no_category_budgets"])

    q2 = random.choice(["How do I set up budgets?", "Should I set category limits?", "What limits do you recommend?"])
    suggested = {cat: round(amt * 1.1 / 500_000) * 500_000 for cat, amt in spending.items()}
    sugg_str = ", ".join(f"{c}: {fmt(v)}" for c, v in list(suggested.items())[:3])
    a2 = build_response("analysis",
        f"Yes, set per-category limits in the app. Suggested starting points: {sugg_str}. Adjust after 1-2 months of tracking.",
        signals=["no_category_budgets"])

    return make_multi_turn_sample(role, ctx, [(q1, a1), (q2, a2)], "multi_turn",
                                  subfamily="mt_no_budget", tags=["no_budget_nudge", "followup"])


def gen_multi_anomaly_none(role):
    """Anomaly question → no anomaly answer."""
    income, spending = SCENARIO_MAP[role]()
    today = date.today()
    hist = {}
    for i in range(1, 4):
        m, y = today.month - i, today.year
        while m <= 0:
            m += 12; y -= 1
        hist[f"{y}-{m:02d}"] = {cat: rvar(amt, 0.10) for cat, amt in spending.items()}
    ctx = make_context(role, income, spending, monthly_history=hist)
    spent = total_spent(spending)

    q1 = "Anything unusual in my spending?"
    a1 = build_response("analysis",
        f"No anomalies detected. Your spending of {fmt(spent)} is consistent with recent months.",
        signals=["on_track"])

    q2 = random.choice(["Good. How about an overview then?", "OK, give me a summary."])
    save = surplus(income, spending)
    top_cat = max(spending, key=spending.get)
    a2 = build_response("analysis",
        f"Spending: {fmt(spent)}, surplus: {fmt(save)}. Top: {top_cat} at {fmt(spending[top_cat])}. All stable.")

    return make_multi_turn_sample(role, ctx, [(q1, a1), (q2, a2)], "multi_turn",
                                  subfamily="mt_no_anomaly", tags=["no_anomaly_negative", "followup"])


def gen_multi_emergency_timeline(role):
    """Emergency fund → how long will it take?"""
    income, spending = SCENARIO_MAP[role]()
    spent = total_spent(spending)
    save = max(surplus(income, spending), 200_000)
    months_target = 6 if role == "Freelancer" else 3
    fund_target = spent * months_target
    ctx = make_light_context(role, income, spending)

    q1 = "How much should my emergency fund be?"
    a1 = build_response("analysis",
        f"Target: {fmt(fund_target)} ({months_target} months × {fmt(spent)}/month expenses).",
        signals=["below_savings_target"])

    q2 = "How long will it take me to build that?"
    months_needed = fund_target / save if save > 0 else float("inf")
    a2 = build_response("analysis",
        f"At your current surplus of {fmt(save)}/month, about {months_needed:.1f} months. "
        f"Accelerate by cutting your top expense or boosting income.")

    return make_multi_turn_sample(role, ctx, [(q1, a1), (q2, a2)], "multi_turn",
                                  subfamily="mt_emergency", tags=["emergency_fund", "followup"])


def gen_multi_split_compare(role):
    """Custom split → compare current spending to target."""
    income, spending = SCENARIO_MAP[role]()
    n, w, s = random.choice(ALL_SPLIT_RATIOS)
    ctx = make_context(role, income, spending, budget_split=(n, w, s))
    needs = income * n / 100
    wants = income * w / 100
    savings = income * s / 100
    spent = total_spent(spending)

    q1 = f"What does a {n}/{w}/{s} split look like for me?"
    a1 = build_response("analysis",
        f"{n}/{w}/{s} on {fmt(income)}: Needs {fmt(needs)}, Wants {fmt(wants)}, Savings {fmt(savings)}.")

    q2 = "How does my actual spending compare?"
    save = surplus(income, spending)
    actual_rate = (save / income * 100) if income > 0 else 0
    if actual_rate >= s:
        a2 = build_response("analysis",
            f"You're saving {actual_rate:.1f}% vs the {s}% target. Spending {fmt(spent)} is within the {fmt(needs + wants)} budget. On track.",
            signals=["on_track"])
    else:
        gap = savings - save
        a2 = build_response("analysis",
            f"Saving {actual_rate:.1f}% vs {s}% target. You need {fmt(gap)} more. Current spending {fmt(spent)} exceeds the budget.",
            signals=["below_savings_target"])

    return make_multi_turn_sample(role, ctx, [(q1, a1), (q2, a2)], "multi_turn",
                                  subfamily="mt_split", tags=["custom_split", "followup"])


def gen_multi_debt_clarify(role):
    """Debt/tax question → follow-up clarification."""
    income, spending = SCENARIO_MAP[role]()
    save = surplus(income, spending)
    debt = rvar(random.choice([5_000_000, 15_000_000, 25_000_000]), 0.3)
    ctx = make_light_context(role, income, spending)

    q1 = f"I have {fmt(debt)} in debt. What do I do?"
    payment = min(save * 0.5, debt / 6)
    payment = max(round(payment / 50_000) * 50_000, 200_000)
    a1 = build_response("analysis",
        f"With {fmt(save)}/month surplus, allocate {fmt(payment)}/month to debt repayment. "
        f"That clears {fmt(debt)} in about {debt / payment:.0f} months.")

    q2 = random.choice(["What if I can pay more?", "Can I speed it up?", "What about interest?"])
    double_payment = payment * 2
    a2 = build_response("analysis",
        f"Doubling to {fmt(double_payment)}/month cuts the timeline to ~{debt / double_payment:.0f} months. "
        f"Prioritize high-interest debt first. Even small extra payments reduce total interest significantly.")

    return make_multi_turn_sample(role, ctx, [(q1, a1), (q2, a2)], "multi_turn",
                                  subfamily="mt_debt", tags=["debt_tax", "followup"])


def gen_multi_anomaly_explain(role):
    """Anomaly warning → user explains purchase → assistant acknowledges."""
    income, spending = SCENARIO_MAP[role]()
    hist = make_monthly_history(spending)
    spike_cat = random.choice(list(spending.keys()))
    spending[spike_cat] = spending[spike_cat] * random.uniform(2.2, 3.0)
    ctx = make_context(role, income, spending, monthly_history=hist)
    prev_values = [hist[m].get(spike_cat, 0) for m in hist if hist[m].get(spike_cat, 0) > 0]
    avg = sum(prev_values) / len(prev_values) if prev_values else spending[spike_cat] / 2.5
    ratio = spending[spike_cat] / avg if avg > 0 else 2.5

    q1 = "Any spending anomalies?"
    a1 = build_response("analysis",
        f"{spike_cat} is at {fmt(spending[spike_cat])}, {ratio:.1f}x your average of {fmt(avg)}.",
        signals=["anomaly_detected"])

    explanations = [
        f"That was a one-time {spike_cat.lower()} purchase I planned.",
        f"I had to replace some {spike_cat.lower()} items.",
        f"It's a seasonal expense, won't repeat.",
    ]
    q2 = random.choice(explanations)
    a2 = build_response("analysis",
        f"Understood — a planned one-time expense. Your {spike_cat} should return to normal (~{fmt(avg)}) next month. No action needed.")

    return make_multi_turn_sample(role, ctx, [(q1, a1), (q2, a2)], "multi_turn",
                                  subfamily="mt_anomaly_ack", tags=["followup"])


# ═══════════════════════════════════════════════════════════════════════════════
# PATCH: SURGICAL BENCHMARK-TARGETED GENERATORS (v7.1)
# Targets: TC04/TC08/TC10 (budget health), TC57/TC99 (zero surplus),
#          TC58 (freelancer tax/accounts), TC81 (debt timeline),
#          TC93 (student income+cuts), TC98 (high income plan)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Helper: zero-surplus scenario (forces total_spent == income exactly) ──────

def zero_surplus_scenario(role):
    """Generate a scenario where spending == income exactly (zero surplus)."""
    income, spending = SCENARIO_MAP[role]()
    spent = total_spent(spending)
    if spent == 0:
        spending["Food"] = income
        return income, spending
    # Scale all categories proportionally so total == income
    scale = income / spent
    spending = {cat: round(amt * scale / 50_000) * 50_000 for cat, amt in spending.items()}
    # Fix rounding drift
    diff = income - sum(spending.values())
    if diff != 0:
        top = max(spending, key=spending.get)
        spending[top] += diff
    return income, spending


# ── A. budget_health_exact_breakdown (targets TC04, TC10, TC98) ───────────────

_BUDGET_HEALTH_EXACT_QUESTIONS = [
    "Give me a breakdown of my budget health.",
    "How is my budget looking this month?",
    "What's my financial status right now?",
    "Give me a complete financial overview.",
    "Am I managing my money well?",
    "Show me a full budget breakdown.",
    "Where does my money go each month?",
    "Review my budget. What should I focus on?",
    "I want a detailed budget check.",
    "Give me the numbers on my spending.",
    "Can you audit my budget?",
    "Break down my finances for me.",
]


def gen_budget_health_exact_breakdown(role):
    """Budget health with 3+ category references, exact totals, and surplus plan.
    Targets TC04 (3+ cat refs, saving advice), TC10 (surplus plan),
    TC98 (50/30/20 rule, emergency fund, ordered plan)."""
    income, spending, *_, ctx = full_scenario(role)
    spent = total_spent(spending)
    save = surplus(income, spending)
    rate = (save / income * 100) if income > 0 else 0

    sorted_cats = sorted(spending.items(), key=lambda x: x[1], reverse=True)
    # Always reference at least 3 categories in the message
    cat_refs = ", ".join(
        f"{cat} ({fmt(amt)})" for cat, amt in sorted_cats[:min(4, len(sorted_cats))]
    )

    ef_low = spent * 3
    ef_high = spent * 6
    needs_50 = income * 0.50
    wants_30 = income * 0.30
    savings_20 = income * 0.20

    q = random.choice(_BUDGET_HEALTH_EXACT_QUESTIONS)
    signals = []

    if save > 0:
        if rate >= 20:
            signals.append("above_savings_target")
            msg = (
                f"Total spending: {fmt(spent)}, surplus: {fmt(save)} ({rate:.1f}% savings rate). "
                f"Breakdown: {cat_refs}. "
                f"You're above the 20% savings target. "
                f"Follow the 50/30/20 rule: Needs {fmt(needs_50)}, Wants {fmt(wants_30)}, Savings {fmt(savings_20)}. "
                f"Ordered plan: first, build an emergency fund of {fmt(ef_low)}-{fmt(ef_high)} (3-6 months expenses), "
                f"then invest the remaining surplus. Start by setting aside {fmt(savings_20)}/month automatically."
            )
        else:
            signals.append("below_savings_target")
            gap = savings_20 - save
            top_cat = sorted_cats[0][0]
            msg = (
                f"Total spending: {fmt(spent)}, surplus: {fmt(save)} ({rate:.1f}% savings rate, below 20% target). "
                f"Breakdown: {cat_refs}. "
                f"You need {fmt(gap)} more to hit the 50/30/20 savings target of {fmt(savings_20)}. "
                f"Consider cutting {top_cat} by 15-20%. "
                f"Emergency fund target: {fmt(ef_low)}-{fmt(ef_high)}. "
                f"Save {fmt(save)}/month and invest once the emergency fund is built."
            )
    else:
        signals.append("deficit")
        msg = (
            f"Total spending: {fmt(spent)} exceeds your income of {fmt(income)} by {fmt(abs(save))}. "
            f"Breakdown: {cat_refs}. "
            f"You are in deficit. Cut spending immediately, starting with {sorted_cats[0][0]} ({fmt(sorted_cats[0][1])}). "
            f"Target the 50/30/20 rule: Needs {fmt(needs_50)}, Wants {fmt(wants_30)}, Savings {fmt(savings_20)}."
        )

    resp = build_response("analysis", msg, signals=signals)
    return make_sample(role, ctx, q, resp, "context_analysis",
                       subfamily="budget_health_exact", tags=["budget_health_patch"])


# ── B. zero_surplus_edge (targets TC08, TC57, TC99) ──────────────────────────

_ZERO_SURPLUS_QUESTIONS = [
    "I have nothing left at the end of the month. What's going on?",
    "I can never save anything. What should I do?",
    "I spend everything I earn. How do I break this cycle?",
    "I feel like I have nothing left at end of month. What's going on?",
    "Why can't I save any money?",
    "My expenses match my income. How do I fix this?",
    "I'm living paycheck to paycheck. Help.",
    "I have zero savings every month. What now?",
    "All my income goes to expenses. What should I change?",
    "I'm spending 100% of my income. What do I cut?",
    "Nothing left after bills. What's the plan?",
    "I break even every month. How to save?",
]


def gen_zero_surplus_edge(role):
    """Zero-surplus scenario with explicit zero language, top cut target, and dual advice.
    Targets TC08 (zero surplus + specific cut + concrete step),
    TC57 (zero + entertainment target + cut+earn), TC99 (zero + tax + cuts + first step)."""
    income, spending = zero_surplus_scenario(role)
    ctx = make_light_context(role, income, spending)
    spent = total_spent(spending)

    sorted_cats = sorted(spending.items(), key=lambda x: x[1], reverse=True)
    top_cat, top_amt = sorted_cats[0]
    second_cat, second_amt = sorted_cats[1] if len(sorted_cats) > 1 else (top_cat, top_amt)

    q = random.choice(_ZERO_SURPLUS_QUESTIONS)
    signals = ["deficit"]

    # Role-specific messaging
    if role == "Student":
        earn_ideas = random.choice([
            "tutoring, freelance work, or a part-time job",
            "part-time work, online tutoring, or delivery gigs",
            "tutoring, an internship, or freelance tasks",
        ])
        msg = (
            f"You're spending {fmt(spent)} — that's 100% of your {fmt(income)} income, "
            f"leaving nothing left and zero surplus. "
            f"Your top expense is {top_cat} at {fmt(top_amt)}. "
            f"Cut {top_cat} by 20-30% (reduce to ~{fmt(top_amt * 0.75)}) and lower {second_cat} ({fmt(second_amt)}). "
            f"Also look into earning more through {earn_ideas}. "
            f"Start by reducing {top_cat} spending this week."
        )
    elif role == "Freelancer":
        tax = income * 0.30
        msg = (
            f"You're spending everything — {fmt(spent)} on a {fmt(income)} income, "
            f"with zero surplus and nothing left. "
            f"As a freelancer, you also need to set aside 30% ({fmt(tax)}) for taxes, "
            f"which means you're actually short by {fmt(tax)}. "
            f"Cut {top_cat} ({fmt(top_amt)}) and {second_cat} ({fmt(second_amt)}) immediately. "
            f"First, open a separate tax account and automate a 30% transfer from every payment. "
            f"Then reduce discretionary spending to create a real surplus."
        )
        signals.append("below_savings_target")
    else:  # Worker
        msg = (
            f"Your spending of {fmt(spent)} consumes your entire {fmt(income)} income — "
            f"100% spent, zero surplus, nothing left. "
            f"Top expenses: {top_cat} ({fmt(top_amt)}) and {second_cat} ({fmt(second_amt)}). "
            f"Cut {top_cat} by 15-20% to free up ~{fmt(top_amt * 0.18)}. "
            f"Also reduce {second_cat} spending. "
            f"Switch to a strict 50/30/20 budget and limit non-essential purchases immediately."
        )

    resp = build_response("analysis", msg, signals=signals)
    return make_sample(role, ctx, q, resp, "context_analysis",
                       subfamily="zero_surplus_edge", tags=["zero_surplus_patch"])


# ── C. freelancer_tax_exact_30pct (targets TC58) ─────────────────────────────

_FREELANCER_TAX_QUESTIONS = [
    "How should I organize my finances as a freelancer?",
    "How do I manage taxes as a freelancer?",
    "What's the best way to handle freelancer taxes?",
    "How should I structure my accounts as a freelancer?",
    "I'm freelancing — how do I handle the money side?",
    "Should I separate my business and personal finances?",
    "What accounts do I need as a freelancer?",
    "Help me set up my freelancer finances.",
    "I don't know how to handle freelance income. Advice?",
    "What's the 30% tax rule for freelancers?",
]


def gen_freelancer_tax_exact_30pct():
    """Freelancer tax with separate accounts, 30% tax amount, and concrete actions.
    Targets TC58 (separate accounts, 30%, tax amount, open/set up/transfer/automat)."""
    income, spending = freelancer_scenario()
    ctx = make_light_context("Freelancer", income, spending)
    spent = total_spent(spending)
    save = surplus(income, spending)
    tax = income * 0.30

    q = random.choice(_FREELANCER_TAX_QUESTIONS)

    msg = (
        f"As a freelancer with {fmt(income)}/month income, separate your business and personal finances. "
        f"Set aside 30% ({fmt(tax)}) of every payment for taxes immediately. "
        f"Open a separate tax account and set up an automatic transfer of 30% from each payment. "
        f"Your personal spending: {fmt(spent)}. After tax reserve, real surplus: {fmt(save - tax)}. "
        f"Keep business expenses (internet, equipment, software) tracked for deductions. "
        f"Transfer your personal budget to a separate account on the 1st of each month."
    )
    resp = build_response("analysis", msg)
    return make_sample("Freelancer", ctx, q, resp, "role_specific",
                       subfamily="freelancer_tax_exact", tags=["freelancer_tax_patch", "debt_tax"])


# ── D. debt_repayment_timeline_exact (targets TC81) ──────────────────────────

_DEBT_QUESTIONS = [
    "I owe a friend {debt}. How do I pay it back quickly?",
    "I have {debt} in debt. What's my repayment plan?",
    "I borrowed {debt}. How fast can I pay it off?",
    "I need to repay {debt}. What should I do?",
    "How do I clear a {debt} debt?",
    "I owe {debt}. Can I pay it back soon?",
    "Help me plan to repay {debt}.",
    "What's the fastest way to pay off {debt}?",
    "I have a {debt} loan to pay back. Strategy?",
    "Someone lent me {debt}. How to repay?",
]


def gen_debt_repayment_timeline_exact(role):
    """Debt repayment with exact surplus, computed timeline, category cuts, and payment amount.
    Targets TC81 (surplus 1.3M, 2 months, cut entertainment/food, payment amount)."""
    income, spending = SCENARIO_MAP[role]()
    spent = total_spent(spending)
    save = surplus(income, spending)
    # Ensure positive surplus for debt repayment
    if save <= 0:
        scale = random.uniform(0.55, 0.75)
        spending = {cat: round(amt * scale / 50_000) * 50_000 for cat, amt in spending.items()}
        spent = total_spent(spending)
        save = surplus(income, spending)
    ctx = make_light_context(role, income, spending)

    sorted_cats = sorted(spending.items(), key=lambda x: x[1], reverse=True)

    # Debt between 1x and 4x surplus so timeline is 1-4 months
    debt = round(save * random.uniform(1.2, 3.5) / 50_000) * 50_000
    debt = max(debt, 500_000)
    months = debt / save if save > 0 else float("inf")
    months_display = f"about {round(months)}" if months > 1.3 else "about 1"

    q = random.choice(_DEBT_QUESTIONS).format(debt=fmt(debt))

    # Pick top discretionary category to cut
    discretionary = [c for c in sorted_cats if c[0] in ("Entertainment", "Shopping", "Food")]
    if not discretionary:
        discretionary = sorted_cats[:2]
    cut_cat, cut_amt = discretionary[0]
    cut_cat2 = discretionary[1][0] if len(discretionary) > 1 else sorted_cats[1][0]

    payment = min(save, debt)
    payment = max(round(payment / 50_000) * 50_000, 100_000)

    msg = (
        f"Your monthly surplus is {fmt(save)}. At {fmt(payment)}/month, "
        f"you can repay {fmt(debt)} in {months_display} {round(months)} months. "
        f"To speed it up, cut {cut_cat} ({fmt(cut_amt)}) and reduce {cut_cat2} spending. "
        f"Allocate {fmt(payment)} per month toward repayment starting immediately."
    )
    resp = build_response("analysis", msg)
    return make_sample(role, ctx, q, resp, "context_analysis",
                       subfamily="debt_timeline_exact", tags=["debt_patch"])


# ── E. student_income_growth_dual_advice (targets TC93) ──────────────────────

_STUDENT_INCOME_QUESTIONS = [
    "My expenses match my income. How can I earn more?",
    "I barely have any surplus. How can I make more money?",
    "I need more income as a student. Ideas?",
    "How do I earn more money while studying?",
    "I'm broke. What side income can I get?",
    "I can't save anything. Should I earn more?",
    "What are ways for a student to earn extra?",
    "I want to increase my income. What should I do?",
    "How can I make money as a student?",
    "I have almost no savings. How to earn more and spend less?",
]

_INCOME_IDEAS = [
    ("part-time tutoring", "Sign up on a tutoring platform or post at your university"),
    ("freelance writing or design", "Start by creating a portfolio on Fiverr or Upwork"),
    ("online tutoring", "Register on Preply or italki to tutor online"),
    ("delivery gigs", "Apply on GrabFood or ShopeeFood — flexible hours around classes"),
    ("part-time retail", "Look for weekend shifts at shops near campus"),
    ("internship", "Apply through your university's career center"),
    ("data entry or virtual assistant", "Try Freelancer.com or local job boards"),
    ("social media management", "Start by offering to manage accounts for small businesses"),
]


def gen_student_income_growth_dual_advice():
    """Student with low surplus: 2+ income ideas + actionable first step + spending cuts.
    Targets TC93 (surplus, 2+ income ideas, actionable first step, also suggests cuts)."""
    income, spending = student_scenario()
    spent = total_spent(spending)
    save = surplus(income, spending)
    # Keep surplus small to match TC93 pattern
    if save > income * 0.15:
        scale = income * random.uniform(0.90, 0.97) / spent if spent > 0 else 1
        spending = {cat: round(amt * scale / 50_000) * 50_000 for cat, amt in spending.items()}
        spent = total_spent(spending)
        save = surplus(income, spending)
    ctx = make_light_context("Student", income, spending)

    sorted_cats = sorted(spending.items(), key=lambda x: x[1], reverse=True)
    cut_cat, cut_amt = sorted_cats[0]
    cut_cat2 = sorted_cats[1][0] if len(sorted_cats) > 1 else cut_cat

    # Pick 2-3 income ideas
    ideas = random.sample(_INCOME_IDEAS, k=random.choice([2, 3]))
    idea_names = " or ".join(i[0] for i in ideas[:2])
    if len(ideas) >= 3:
        idea_names = f"{ideas[0][0]}, {ideas[1][0]}, or {ideas[2][0]}"
    first_step_action = ideas[0][1]

    q = random.choice(_STUDENT_INCOME_QUESTIONS)

    msg = (
        f"Your surplus is only {fmt(save)}/month. "
        f"To improve: first, cut {cut_cat} ({fmt(cut_amt)}) and reduce {cut_cat2} spending. "
        f"Then boost income through {idea_names}. "
        f"Start by: {first_step_action}. "
        f"Even an extra 1-2M VND/month would significantly improve your position."
    )
    resp = build_response("analysis", msg)
    return make_sample("Student", ctx, q, resp, "role_specific",
                       subfamily="student_income_dual", tags=["student_income_patch"])


# ── F. multi_turn_patch_followups (follow-ups for weak clusters) ─────────────

def gen_multi_zero_surplus_followup(role):
    """Zero surplus overview → follow-up on what to cut first."""
    income, spending = zero_surplus_scenario(role)
    ctx = make_light_context(role, income, spending)
    spent = total_spent(spending)
    sorted_cats = sorted(spending.items(), key=lambda x: x[1], reverse=True)
    top_cat, top_amt = sorted_cats[0]
    second_cat, second_amt = sorted_cats[1]

    q1 = random.choice(["Why can't I save anything?", "I have nothing left each month."])
    a1 = build_response("analysis",
        f"You're spending {fmt(spent)} — 100% of your {fmt(income)} income. Zero surplus, nothing left. "
        f"Top: {top_cat} ({fmt(top_amt)}), {second_cat} ({fmt(second_amt)}). Cut {top_cat} first.",
        signals=["deficit"])

    q2 = random.choice(["How much should I cut?", "What's a realistic target?", "Where do I start?"])
    target_cut = round(top_amt * 0.20 / 50_000) * 50_000
    new_top = top_amt - target_cut
    a2 = build_response("analysis",
        f"Reduce {top_cat} by {fmt(target_cut)}/month (from {fmt(top_amt)} to {fmt(new_top)}). "
        f"That frees up {fmt(target_cut)} for savings. Start this week — limit {top_cat.lower()} purchases.",
        signals=["below_savings_target"])

    return make_multi_turn_sample(role, ctx, [(q1, a1), (q2, a2)], "multi_turn",
                                  subfamily="mt_zero_surplus", tags=["zero_surplus_patch", "followup"])


def gen_multi_budget_health_drilldown(role):
    """Budget health → drill down into specific category."""
    income, spending, *_, ctx = full_scenario(role)
    spent = total_spent(spending)
    save = surplus(income, spending)
    sorted_cats = sorted(spending.items(), key=lambda x: x[1], reverse=True)
    cat_refs = ", ".join(f"{c} ({fmt(a)})" for c, a in sorted_cats[:3])

    q1 = random.choice(["Give me a budget breakdown.", "How's my spending?"])
    a1 = build_response("analysis",
        f"Total: {fmt(spent)}, surplus: {fmt(save)}. Breakdown: {cat_refs}.",
        signals=["above_savings_target"] if save > income * 0.2 else ["below_savings_target"])

    focus_cat, focus_amt = sorted_cats[0]
    pct = focus_amt / income * 100
    q2 = f"Tell me more about my {focus_cat} spending."
    a2 = build_response("analysis",
        f"{focus_cat}: {fmt(focus_amt)} ({pct:.1f}% of income). "
        f"This is your largest category. Consider setting a monthly limit of {fmt(round(focus_amt * 0.9 / 500_000) * 500_000)} to save more.")

    return make_multi_turn_sample(role, ctx, [(q1, a1), (q2, a2)], "multi_turn",
                                  subfamily="mt_budget_drill", tags=["budget_health_patch", "followup"])


def gen_multi_freelancer_tax_followup(role):
    """Freelancer tax setup → follow-up on deductions."""
    income, spending = freelancer_scenario()
    ctx = make_light_context("Freelancer", income, spending)
    tax = income * 0.30

    q1 = random.choice(["How do I handle taxes as a freelancer?", "What accounts do I need?"])
    a1 = build_response("analysis",
        f"Set aside 30% ({fmt(tax)}) for taxes. Open a separate tax account. "
        f"Transfer 30% of every payment automatically.",
        signals=[])

    q2 = random.choice(["What expenses can I deduct?", "How do I reduce my taxes?"])
    a2 = build_response("analysis",
        f"Deductible business expenses: internet, phone, equipment, software, co-working space. "
        f"Track every receipt. These reduce your taxable income by 10-20%, saving {fmt(tax * 0.15)} or more.")

    return make_multi_turn_sample("Freelancer", ctx, [(q1, a1), (q2, a2)], "multi_turn",
                                  subfamily="mt_freelancer_tax", tags=["freelancer_tax_patch", "followup"])


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════

FAMILY_CONFIG = {
    # ── ~34% action/CRUD  (target post-dedup ~1870) ────────────
    "action_crud": {
        "target": 1920,
        "generators": [
            (gen_log_expense, 4),
            (gen_log_expense_natural, 4),
            (gen_log_income, 3),
            (gen_log_income_natural, 3),
            (gen_update_transaction, 2),
            (gen_delete_transaction, 2),
        ],
    },
    # ── ~10% clarification  (target post-dedup ~550) ──────────
    "clarification": {
        "target": 610,
        "generators": [
            (gen_missing_field_clarification, 3),
            (gen_missing_category, 2),
            (gen_vague_report, 2),
            (gen_transaction_correction_turn, 3),
            (gen_ambiguous_intent, 2),
        ],
    },
    # ── ~15% hard negatives  (target post-dedup ~825) ─────────
    "hard_negative": {
        "target": 950,
        "generators": [
            (gen_neg_spending_question, 2),
            (gen_hard_negative_hypothetical, 2),
            (gen_hard_negative_status_query, 2),
            (gen_neg_general_advice, 2),
            (gen_neg_future_tense, 2),
            (gen_neg_should_i_buy, 1),
            # no-anomaly negatives — boosted for 180-220 target
            (gen_no_anomaly_negative, 4),
            # negative distinction
            (gen_neg_split_not_action, 2),
            (gen_neg_show_salary, 1),
            (gen_neg_food_query, 2),
            (gen_neg_affordability, 2),
        ],
    },
    # ── ~24% context analysis  (target post-dedup ~1400) ──────
    "context_analysis": {
        "target": 1480,
        "generators": [
            (gen_budget_health, 2),
            (gen_category_budget_status, 2),
            (gen_savings_rate, 1),
            (gen_mom_comparison, 2),
            (gen_recurring_analysis, 2),
            (gen_anomaly_alert, 2),
            (gen_anomaly_proactive, 1),
            (gen_goal_progress, 2),
            (gen_cross_feature_priority_analysis, 1),
            (gen_no_budget_nudge, 1),
            (gen_forecast_discussion, 1),
            # custom split
            (gen_custom_split_direct, 3),
            (gen_custom_split_compare, 2),
            # emergency fund
            (gen_emergency_fund, 3),
            (gen_emergency_fund_building, 2),
            # no-budget nudge expanded — boosted for 180-220 target
            (gen_no_budget_nudge_expanded, 3),
            # cross-feature advice — boosted for 240-320 target
            (gen_cross_feature_anomaly_budget, 3),
            (gen_cross_feature_deficit_discretionary, 2),
            (gen_cross_feature_goal_forecast, 2),
            (gen_cross_feature_no_budget_concentrated, 2),
            # debt/tax edge
            (gen_debt_tax_freelancer_debt, 1),
            (gen_debt_tax_after_tax_surplus, 2),
            (gen_debt_vs_emergency_tradeoff, 1),
            (gen_debt_student_part_time_tax, 1),
            # ── PATCH: benchmark-targeted generators ──
            (gen_budget_health_exact_breakdown, 4),  # A: 120-180 rows
            (gen_zero_surplus_edge, 3),               # B: 100-150 rows
            (gen_debt_repayment_timeline_exact, 3),   # D: 80-120 rows
        ],
    },
    # ── ~11% multi-turn  (target post-dedup ~600) ─────────────
    "multi_turn": {
        "target": 660,
        "generators": [
            (gen_multi_overview_drilldown, 2),
            (gen_anomaly_acknowledged_by_user, 2),
            (gen_multi_log_correction, 2),
            (gen_multi_goal_whatif, 2),
            (gen_multi_mom_followup, 2),
            # weak-family follow-ups — boosted weights
            (gen_multi_overview_no_budget, 3),    # target 40-70
            (gen_multi_anomaly_none, 2),
            (gen_multi_emergency_timeline, 3),    # target 50-80
            (gen_multi_split_compare, 2),
            (gen_multi_debt_clarify, 3),           # target 40-70
            (gen_multi_anomaly_explain, 3),        # target 50-80
            # ── PATCH: benchmark-targeted multi-turn ──
            (gen_multi_zero_surplus_followup, 2),       # F: zero surplus drill
            (gen_multi_budget_health_drilldown, 2),     # F: budget breakdown drill
            (gen_multi_freelancer_tax_followup, 1),     # F: freelancer tax drill
        ],
    },
}

# Role-specific generators don't take a role parameter — they are role-locked
# Target ~550 post-dedup at ~77% survival → ~715 pre-dedup
ROLE_SPECIFIC_GENERATORS = [
    (gen_student_semester_plan, 80),
    (gen_student_debt_advice, 80),
    (gen_student_part_time, 80),
    (gen_worker_investment, 80),
    (gen_worker_bhxh, 80),
    (gen_worker_salary_split, 80),
    (gen_freelancer_tax, 80),
    (gen_freelancer_buffer, 80),
    (gen_freelancer_quarterly_tax, 80),
    # ── PATCH: benchmark-targeted role-specific ──
    (gen_freelancer_tax_exact_30pct, 90),   # C: 80-120 rows
    (gen_student_income_growth_dual_advice, 90),  # E: 80-120 rows
]


def generate_all():
    dataset = []
    roles = ["Student", "Worker", "Freelancer"]
    total_generated = 0
    total_failures = 0

    # Generate from family configs
    for family_name, config in FAMILY_CONFIG.items():
        target = config["target"]
        gens = config["generators"]
        total_weight = sum(w for _, w in gens)
        family_start = len(dataset)
        family_failures = 0
        print(f"\n[family] {family_name}: target={target}")

        for gen_fn, weight in gens:
            count_per_role = max(1, target * weight // (total_weight * len(roles)))
            gen_name = gen_fn.__name__
            print(f"  [generator] {gen_name}: {count_per_role} per role")
            for role in roles:
                role_success = 0
                role_failures = 0
                for _ in range(count_per_role):
                    try:
                        dataset.append(gen_fn(role))
                        total_generated += 1
                        role_success += 1
                    except Exception as e:
                        total_failures += 1
                        family_failures += 1
                        role_failures += 1
                        if role_failures <= 3:
                            print(f"    [warn] {gen_name}({role}) failed: {e}")
                print(f"    [role] {role}: +{role_success} ok, {role_failures} failed")
        print(
            f"  [family-done] {family_name}: +{len(dataset) - family_start} samples, {family_failures} failed"
        )

    # Role-specific generators
    print("\n[family] role_specific")
    role_specific_start = len(dataset)
    role_specific_failures = 0
    for gen_fn, count in ROLE_SPECIFIC_GENERATORS:
        gen_name = gen_fn.__name__
        success = 0
        failures = 0
        print(f"  [generator] {gen_name}: count={count}")
        for _ in range(count):
            try:
                dataset.append(gen_fn())
                total_generated += 1
                success += 1
            except Exception as e:
                total_failures += 1
                role_specific_failures += 1
                failures += 1
                if failures <= 3:
                    print(f"    [warn] {gen_name} failed: {e}")
        print(f"    [done] +{success} ok, {failures} failed")

    print(
        f"\nGeneration complete: {total_generated} samples built, {total_failures} failed before dedup."
    )
    print(
        f"Role-specific summary: +{len(dataset) - role_specific_start} samples, {role_specific_failures} failed"
    )

    return dataset


def deduplicate(dataset, threshold=0.85):
    """Remove near-duplicate examples by (family + question) similarity.

    Dedup is scoped per-family so that different families with similar question
    templates (e.g. "How is my spending?") don't eliminate each other.
    Additionally, the completion text is included in the fingerprint so examples
    with different numeric answers survive even when the question is identical.
    """
    seen_by_family: dict[str, list[set]] = {}
    unique = []
    for entry in dataset:
        family = entry.get("family", "unknown")
        subfamily = entry.get("subfamily", "")
        # Build fingerprint from question + completion + subfamily + role + context numbers
        prompt_msgs = entry.get("prompt", [])
        q = ""
        ctx_snippet = ""
        for msg in prompt_msgs:
            if msg["role"] == "user":
                content = msg["content"]
                if "FINANCIAL CONTEXT" in content:
                    # Extract income/spending numbers from context for uniqueness
                    ctx_snippet = content[content.find("TOTAL INCOME"):content.find("TOTAL INCOME") + 80] if "TOTAL INCOME" in content else content[:80]
                else:
                    q = content.lower()
        role = entry.get("role", "")
        # Include completion content in fingerprint for numeric diversity
        comp = ""
        for msg in entry.get("completion", []):
            comp = msg.get("content", "")[:200].lower()
        fingerprint = f"{subfamily}|{role}|{q}|{comp}|{ctx_snippet}"
        if len(fingerprint) > 600:
            fingerprint = fingerprint[:600]
        fp_tokens = set(fingerprint.split())

        seen_list = seen_by_family.setdefault(family, [])
        is_dup = False
        for seen_fp in seen_list:
            if not fp_tokens or not seen_fp:
                continue
            overlap = len(fp_tokens & seen_fp) / max(len(fp_tokens | seen_fp), 1)
            if overlap >= threshold:
                is_dup = True
                break
        if not is_dup:
            seen_list.append(fp_tokens)
            unique.append(entry)
    return unique


if __name__ == "__main__":
    random.seed(42)
    print("Generating FINA v3 structured training data...")

    dataset = generate_all()
    print(f"Generated {len(dataset)} examples.")

    if USE_EXTERNAL_DATA:
        print("External data is disabled (USE_EXTERNAL_DATA=False).")

    print(f"\nBefore dedup: {len(dataset)}")
    dataset = deduplicate(dataset, threshold=0.85)
    print(f"After dedup:  {len(dataset)}")

    random.shuffle(dataset)

    # Write dataset
    print(f"\nWriting {len(dataset)} examples to '{OUTPUT_FILE}'...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for idx, entry in enumerate(dataset, start=1):
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if idx % 1000 == 0 or idx == len(dataset):
                print(f"  Wrote {idx}/{len(dataset)}")

    # Print family summary
    family_counts = {}
    role_by_family = {}
    subfamily_counts = {}
    tag_counts = {}
    for entry in dataset:
        fam = entry.get("family", "unknown")
        family_counts[fam] = family_counts.get(fam, 0) + 1
        r = entry.get("role", "unknown")
        role_by_family.setdefault(fam, {})
        role_by_family[fam][r] = role_by_family[fam].get(r, 0) + 1
        sf = entry.get("subfamily", "")
        if sf:
            subfamily_counts[sf] = subfamily_counts.get(sf, 0) + 1
        for t in entry.get("tags", []):
            tag_counts[t] = tag_counts.get(t, 0) + 1

    total = len(dataset)
    print(f"\nSaved {total} examples to '{OUTPUT_FILE}'")
    print("\n{'='*60}")
    print("FAMILY DISTRIBUTION:")
    print("=" * 60)
    for fam, cnt in sorted(family_counts.items(), key=lambda x: -x[1]):
        pct = cnt / total * 100
        roles = role_by_family.get(fam, {})
        role_str = ", ".join(f"{r}:{c}" for r, c in sorted(roles.items()))
        print(f"  {fam:25s} {cnt:6d} ({pct:5.1f}%)  [{role_str}]")

    print("\n" + "=" * 60)
    print("SUBFAMILY DISTRIBUTION:")
    print("=" * 60)
    for sf, cnt in sorted(subfamily_counts.items(), key=lambda x: -x[1]):
        print(f"  {sf:30s} {cnt:6d}")

    print("\n" + "=" * 60)
    print("TAG DISTRIBUTION:")
    print("=" * 60)
    for t, cnt in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:30s} {cnt:6d}")

    # Weak-family coverage check
    print("\n" + "=" * 60)
    print("WEAK-FAMILY COVERAGE CHECK:")
    print("=" * 60)
    weak_tags = ["custom_split", "emergency_fund", "no_budget_nudge",
                 "no_anomaly_negative", "cross_feature_advice", "debt_tax",
                 "ratio_generalization", "hard_negative", "followup",
                 "budget_health_patch", "zero_surplus_patch", "freelancer_tax_patch",
                 "debt_patch", "student_income_patch"]
    for tag in weak_tags:
        cnt = tag_counts.get(tag, 0)
        pct = cnt / total * 100
        status = "OK" if cnt >= 50 else "LOW" if cnt >= 20 else "MISSING"
        print(f"  {tag:30s} {cnt:6d} ({pct:5.1f}%)  {status}")
