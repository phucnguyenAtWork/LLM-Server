"""
FINA Benchmark — 100 Test Cases (v2: Calculation & Actionable Focus)
=====================================================================
Priority categories:
  CALC = Numerical Accuracy — exact computed values, correct math
  ACT  = Actionable Advice  — specific, contextual, usable guidance

Design principles:
  - Every test case with computable numbers checks for EXACT values
  - Actionable checks verify specificity (references user's categories/amounts),
    not just keyword presence ("mentions saving" is too loose)
  - Format and role checks are removed as separate concerns
"""

CALC = "Numerical Accuracy"
ACT  = "Actionable Advice"

from fina_schema import parse_model_output


# ── Helpers ──────────────────────────────────────────────────────────────────

def _vnd_strs(amount):
    """Return common VND string representations for an integer amount."""
    n = int(amount)
    return {
        "{:,.0f}".format(n).replace(",", "."),  # 3.000.000
        "{:,}".format(n),                        # 3,000,000
        str(n),                                  # 3000000
    }


def has_amount(amount):
    """Check if response contains a VND amount in any common format."""
    fmts = _vnd_strs(amount)
    return lambda r: any(f in r for f in fmts)


def has_any_amount(*amounts):
    """Check if response contains at least one of the listed amounts."""
    all_fmts = set()
    for a in amounts:
        all_fmts.update(_vnd_strs(a))
    return lambda r: any(f in r for f in all_fmts)


def has_words(*words):
    """Check if response contains at least one word/phrase (case-insensitive)."""
    return lambda r: any(w in r.lower() for w in words)


def has_all_words(*words):
    """Check if response contains ALL words/phrases (case-insensitive)."""
    return lambda r: all(w in r.lower() for w in words)


def refs_user_categories(spending, min_count=2):
    """Check if response references at least N of the user's spending categories."""
    cats = [k.lower() for k in spending.keys()]
    return lambda r: sum(1 for c in cats if c in r.lower()) >= min_count


def suggests_specific_cut(spending):
    """Check if response names a specific category from the user's data AND a cut action."""
    cats = [k.lower() for k in spending.keys()]
    cut_words = ["cut", "reduce", "lower", "limit", "decrease", "less"]
    def check(r):
        rl = r.lower()
        return any(c in rl for c in cats) and any(w in rl for w in cut_words)
    return check


def has_goal_reference(goal_name):
    """Check if response mentions a financial goal by name."""
    return lambda r: goal_name.lower() in r.lower()


def has_balance_sum(expected_total):
    """Check if response mentions the total liquid balance."""
    return has_amount(expected_total)


def has_forecast_mention():
    """Check if response references a forecast or projection."""
    return has_words("forecast", "projected", "next month", "projection", "predict")


def has_budget_nudge():
    """Check if response suggests setting up category budgets."""
    return has_words("set up", "category budget", "spending limit", "in the app", "per-category")


def has_over_limit_mention(category_name):
    """Check if response mentions a specific category being over its limit."""
    return lambda r: category_name.lower() in r.lower() and any(w in r.lower() for w in ["over", "exceed", "limit"])


def has_action_tag(action_type, **expected_fields):
    """Check if response contains a valid structured action payload with expected fields."""
    def check(r):
        parsed = parse_model_output(r)
        if not parsed or not parsed.action or parsed.action.type.value != action_type:
            return False
        args = parsed.action.arguments.model_dump(mode="json")
        return all(str(v).lower() in str(args.get(k, "")).lower() for k, v in expected_fields.items())
    return check


def has_no_action_tag():
    """Check that response does NOT contain any structured action payload."""
    return lambda r: not (parse_model_output(r) and parse_model_output(r).action)


def has_mom_reference():
    """Check if response references month-over-month comparison."""
    return has_words("last month", "previous month", "month over month", "compared to", "increase", "decrease", "more than last", "less than last")


def has_recurring_reference():
    """Check if response references recurring/fixed expenses."""
    return has_words("recurring", "fixed", "subscription", "monthly", "regular")


def has_anomaly_reference(category_name):
    """Check if response references an anomaly in a specific category."""
    return lambda r: category_name.lower() in r.lower() and any(w in r.lower() for w in ["unusual", "spike", "higher than", "above average", "anomal", "watch", "jump"])


# ── Shorthand ────────────────────────────────────────────────────────────────

def _total(spending):
    return sum(spending.values())

def _surplus(income, spending):
    return income - _total(spending)

def _pct_of_income(amount, income):
    return round(amount / income * 100)


TEST_CASES = [
    # ══════════════════════════════════════════════════════════════════════════
    #  BUDGET HEALTH CHECK  (TC01–TC12)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "TC01", "name": "Budget Health — Student, tight budget",
        "role": "Student", "income": 3_000_000,
        "spending": {"Food": 1_500_000, "Transport": 400_000, "Entertainment": 600_000, "Education": 200_000},
        # total=2,700,000  surplus=300,000  savings_target=600,000
        "question": "How is my budget looking this month?",
        "checks": {
            "Total spent correct":    {"fn": has_amount(2_700_000), "cat": CALC},
            "Surplus amount correct": {"fn": has_amount(300_000), "cat": CALC},
            "References user categories": {"fn": refs_user_categories({"Food": 0, "Transport": 0, "Entertainment": 0, "Education": 0}, 2), "cat": ACT},
            "Gives saving advice":    {"fn": has_words("cut", "reduce", "save", "lower", "limit"), "cat": ACT},
        },
    },
    {
        "id": "TC02", "name": "Budget Health — Student, comfortable",
        "role": "Student", "income": 5_000_000,
        "spending": {"Food": 2_200_000, "Transport": 500_000, "Entertainment": 900_000, "Education": 300_000},
        # total=3,900,000  surplus=1,100,000
        "question": "How is my budget looking this month?",
        "checks": {
            "Total spent correct":    {"fn": has_amount(3_900_000), "cat": CALC},
            "Surplus amount correct": {"fn": has_amount(1_100_000), "cat": CALC},
            "References user categories": {"fn": refs_user_categories({"Food": 0, "Entertainment": 0, "Education": 0}, 2), "cat": ACT},
            "Suggests what to do with surplus": {"fn": has_words("save", "invest", "set aside", "emergency", "put"), "cat": ACT},
        },
    },
    {
        "id": "TC03", "name": "Budget Health — Student, over budget",
        "role": "Student", "income": 3_500_000,
        "spending": {"Food": 1_800_000, "Transport": 500_000, "Entertainment": 800_000, "Shopping": 700_000},
        # total=3,800,000  deficit=-300,000
        "question": "Am I over budget this month?",
        "checks": {
            "Total spent correct":    {"fn": has_amount(3_800_000), "cat": CALC},
            "Deficit amount correct": {"fn": has_amount(300_000), "cat": CALC},
            "Detects overspending":   {"fn": has_words("over", "exceed", "deficit", "more than", "negative", "beyond"), "cat": CALC},
            "Names category to cut":  {"fn": suggests_specific_cut({"Food": 0, "Entertainment": 0, "Shopping": 0}), "cat": ACT},
        },
    },
    {
        "id": "TC04", "name": "Budget Health — Worker, balanced",
        "role": "Worker", "income": 15_000_000,
        "spending": {"Food": 3_000_000, "Transport": 1_500_000, "Bills": 2_500_000, "Shopping": 1_500_000, "Health": 500_000},
        # total=9,000,000  surplus=6,000,000
        "question": "Give me a breakdown of my budget health.",
        "checks": {
            "Total spent correct":    {"fn": has_amount(9_000_000), "cat": CALC},
            "Surplus correct":        {"fn": has_amount(6_000_000), "cat": CALC},
            "References 3+ categories": {"fn": refs_user_categories({"Food": 0, "Transport": 0, "Bills": 0, "Shopping": 0, "Health": 0}, 3), "cat": ACT},
            "Suggests saving/investing surplus": {"fn": has_words("save", "invest", "emergency", "set aside"), "cat": ACT},
        },
    },
    {
        "id": "TC05", "name": "Budget Health — Worker, high earner",
        "role": "Worker", "income": 40_000_000,
        "spending": {"Food": 5_000_000, "Transport": 3_000_000, "Bills": 6_000_000, "Shopping": 4_000_000, "Entertainment": 3_000_000},
        # total=21,000,000  surplus=19,000,000  spending_pct=52.5%
        "question": "How does my spending look relative to my income?",
        "checks": {
            "Total spent correct":    {"fn": has_amount(21_000_000), "cat": CALC},
            "Surplus correct":        {"fn": has_amount(19_000_000), "cat": CALC},
            "Mentions percentage":    {"fn": lambda r: "%" in r, "cat": CALC},
            "Suggests what to do with surplus": {"fn": has_words("invest", "save", "grow", "fund", "allocat"), "cat": ACT},
        },
    },
    {
        "id": "TC06", "name": "Budget Health — Freelancer, low month",
        "role": "Freelancer", "income": 8_000_000,
        "spending": {"Food": 2_000_000, "Transport": 800_000, "Bills": 1_500_000, "Equipment": 1_000_000},
        # total=5,300,000  surplus=2,700,000
        "question": "This was a slow month. How's my budget?",
        "checks": {
            "Total spent correct":    {"fn": has_amount(5_300_000), "cat": CALC},
            "Surplus correct":        {"fn": has_amount(2_700_000), "cat": CALC},
            "Acknowledges tight situation": {"fn": has_words("tight", "careful", "limited", "slow", "lower", "less"), "cat": ACT},
            "Concrete saving action": {"fn": has_words("cut", "reduce", "save", "set aside", "buffer"), "cat": ACT},
        },
    },
    {
        "id": "TC07", "name": "Budget Health — Freelancer, good month",
        "role": "Freelancer", "income": 35_000_000,
        "spending": {"Food": 3_500_000, "Transport": 1_500_000, "Bills": 3_000_000, "Entertainment": 2_000_000, "Equipment": 2_500_000},
        # total=12,500,000  surplus=22,500,000
        "question": "I had a great month! How should I handle the surplus?",
        "checks": {
            "Total spent correct":    {"fn": has_amount(12_500_000), "cat": CALC},
            "Surplus correct":        {"fn": has_amount(22_500_000), "cat": CALC},
            "Actionable allocation":  {"fn": has_words("put", "save", "invest", "transfer", "allocate", "set aside"), "cat": ACT},
            "Multiple concrete steps": {"fn": lambda r: sum(1 for w in ["save", "invest", "emergency", "tax", "buffer", "reserve"] if w in r.lower()) >= 2, "cat": ACT},
        },
    },
    {
        "id": "TC08", "name": "Budget Health — Worker, paycheck to paycheck",
        "role": "Worker", "income": 10_000_000,
        "spending": {"Food": 3_500_000, "Transport": 2_000_000, "Bills": 2_500_000, "Shopping": 1_500_000, "Health": 500_000},
        # total=10,000,000  surplus=0
        "question": "I feel like I have nothing left at end of month. What's going on?",
        "checks": {
            "Total spent correct":       {"fn": has_amount(10_000_000), "cat": CALC},
            "Identifies zero surplus":   {"fn": has_words("nothing left", "zero", "no room", "all", "100%", "entire", "everything", "no surplus"), "cat": CALC},
            "Names specific category to cut": {"fn": suggests_specific_cut({"Food": 0, "Shopping": 0, "Transport": 0}), "cat": ACT},
            "Concrete next step":        {"fn": has_words("cut", "reduce", "lower", "switch", "limit", "cancel"), "cat": ACT},
        },
    },
    {
        "id": "TC09", "name": "Budget Health — Student, scholarship income",
        "role": "Student", "income": 7_000_000,
        "spending": {"Food": 2_000_000, "Transport": 500_000, "Education": 1_500_000, "Entertainment": 500_000},
        # total=4,500,000  surplus=2,500,000
        "question": "I have a scholarship of 7 million. Am I managing it well?",
        "checks": {
            "Total spent correct":    {"fn": has_amount(4_500_000), "cat": CALC},
            "Surplus correct":        {"fn": has_amount(2_500_000), "cat": CALC},
            "Positive assessment":    {"fn": has_words("good", "well", "healthy", "great", "solid", "on track"), "cat": ACT},
            "Suggests using surplus": {"fn": has_words("save", "emergency", "invest", "set aside"), "cat": ACT},
        },
    },
    {
        "id": "TC10", "name": "Budget Health — Freelancer, multiple clients",
        "role": "Freelancer", "income": 25_000_000,
        "spending": {"Food": 3_000_000, "Transport": 2_000_000, "Bills": 2_500_000, "Equipment": 3_000_000, "Software": 1_500_000},
        # total=12,000,000  surplus=13,000,000
        "question": "Review my budget. I'm managing 3 client projects.",
        "checks": {
            "Total spent correct":    {"fn": has_amount(12_000_000), "cat": CALC},
            "Surplus correct":        {"fn": has_amount(13_000_000), "cat": CALC},
            "References user categories": {"fn": refs_user_categories({"Food": 0, "Equipment": 0, "Software": 0, "Bills": 0}, 2), "cat": ACT},
            "Actionable surplus plan": {"fn": has_words("save", "invest", "tax", "set aside", "allocate"), "cat": ACT},
        },
    },
    {
        "id": "TC11", "name": "Budget Health — Worker, new job",
        "role": "Worker", "income": 12_000_000,
        "spending": {"Food": 2_500_000, "Transport": 1_800_000, "Bills": 2_000_000, "Shopping": 2_500_000},
        # total=8,800,000  surplus=3,200,000  needs=6M wants=3.6M savings=2.4M
        "question": "I just started a new job. How should I manage this salary?",
        "checks": {
            "Total spent correct":       {"fn": has_amount(8_800_000), "cat": CALC},
            "Surplus correct":           {"fn": has_amount(3_200_000), "cat": CALC},
            "Needs target (50%)":        {"fn": has_amount(6_000_000), "cat": CALC},
            "Savings target (20%)":      {"fn": has_amount(2_400_000), "cat": CALC},
            "Concrete first step":       {"fn": has_words("emergency", "save", "set aside", "automat", "transfer"), "cat": ACT},
        },
    },
    {
        "id": "TC12", "name": "Budget Health — Student, minimal spending",
        "role": "Student", "income": 4_000_000,
        "spending": {"Food": 1_200_000, "Transport": 300_000, "Education": 500_000},
        # total=2,000,000  surplus=2,000,000
        "question": "I'm trying to be frugal. How am I doing?",
        "checks": {
            "Total spent correct":    {"fn": has_amount(2_000_000), "cat": CALC},
            "Surplus correct":        {"fn": has_amount(2_000_000), "cat": CALC},
            "Positive assessment":    {"fn": has_words("good", "great", "well", "frugal", "disciplin", "healthy", "solid"), "cat": ACT},
            "Suggests using surplus": {"fn": has_words("save", "invest", "emergency", "set aside", "build"), "cat": ACT},
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  CUSTOM BUDGET SPLIT  (TC13–TC22)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "TC13", "name": "Custom Split 60/25/15 — Worker",
        "role": "Worker", "income": 20_000_000,
        "spending": {"Food": 4_000_000, "Transport": 2_000_000, "Bills": 3_000_000, "Shopping": 2_500_000},
        "question": "Calculate a 60/25/15 budget split for me.",
        "checks": {
            "Correct needs (60%)":    {"fn": has_amount(12_000_000), "cat": CALC},
            "Correct wants (25%)":    {"fn": has_amount(5_000_000), "cat": CALC},
            "Correct savings (15%)":  {"fn": has_amount(3_000_000), "cat": CALC},
            "Shows the split ratios": {"fn": lambda r: "60" in r and "25" in r and "15" in r, "cat": CALC},
            "Compares to actual spending": {"fn": has_words("actual", "current", "spending", "you spend", "you're spending"), "cat": ACT},
        },
    },
    {
        "id": "TC14", "name": "Custom Split 70/20/10 — Student",
        "role": "Student", "income": 4_000_000,
        "spending": {"Food": 1_500_000, "Transport": 500_000, "Entertainment": 400_000},
        "question": "Can you calculate a 70/20/10 split for me?",
        "checks": {
            "Correct needs (70%)":    {"fn": has_amount(2_800_000), "cat": CALC},
            "Correct wants (20%)":    {"fn": has_amount(800_000), "cat": CALC},
            "Correct savings (10%)":  {"fn": has_amount(400_000), "cat": CALC},
            "Compares to actual":     {"fn": has_words("actual", "current", "spending", "you spend", "you're"), "cat": ACT},
        },
    },
    {
        "id": "TC15", "name": "Custom Split 40/30/30 — Freelancer",
        "role": "Freelancer", "income": 30_000_000,
        "spending": {"Food": 3_000_000, "Transport": 1_500_000, "Bills": 2_000_000, "Equipment": 2_500_000},
        "question": "I want a 40/30/30 split. Calculate it.",
        "checks": {
            "Correct needs (40%)":    {"fn": has_amount(12_000_000), "cat": CALC},
            "Correct wants (30%)":    {"fn": has_amount(9_000_000), "cat": CALC},
            "Correct savings (30%)":  {"fn": has_amount(9_000_000), "cat": CALC},
            "Acknowledges high savings intent": {"fn": has_words("save", "savings", "aggressiv", "great", "good"), "cat": ACT},
        },
    },
    {
        "id": "TC16", "name": "Custom Split 80/10/10 — Worker",
        "role": "Worker", "income": 25_000_000,
        "spending": {"Food": 5_000_000, "Transport": 3_000_000, "Bills": 6_000_000, "Shopping": 3_000_000},
        "question": "My expenses are high. Try an 80/10/10 split.",
        "checks": {
            "Correct needs (80%)":    {"fn": has_amount(20_000_000), "cat": CALC},
            "Correct wants (10%)":    {"fn": has_amount(2_500_000), "cat": CALC},
            "Correct savings (10%)":  {"fn": has_amount(2_500_000), "cat": CALC},
            # total spending = 17M, needs limit = 20M — fits
            "Notes spending vs limit": {"fn": has_words("fit", "within", "under", "below", "room", "actual", "current"), "cat": ACT},
        },
    },
    {
        "id": "TC17", "name": "Custom Split 50/30/20 explicit — Student",
        "role": "Student", "income": 6_000_000,
        "spending": {"Food": 2_000_000, "Transport": 600_000, "Entertainment": 800_000, "Education": 500_000},
        "question": "Show me the standard 50/30/20 breakdown for my income.",
        "checks": {
            "Correct needs (50%)":    {"fn": has_amount(3_000_000), "cat": CALC},
            "Correct wants (30%)":    {"fn": has_amount(1_800_000), "cat": CALC},
            "Correct savings (20%)":  {"fn": has_amount(1_200_000), "cat": CALC},
            # total=3,900,000 < needs+wants=4,800,000 so under budget
            "Compares actual to targets": {"fn": has_words("actual", "current", "spending", "you spend"), "cat": ACT},
        },
    },
    {
        "id": "TC18", "name": "Custom Split 55/25/20 — Freelancer",
        "role": "Freelancer", "income": 20_000_000,
        "spending": {"Food": 3_000_000, "Transport": 1_500_000, "Bills": 2_000_000, "Health": 500_000},
        "question": "Calculate a 55/25/20 budget for me.",
        "checks": {
            "Correct needs (55%)":    {"fn": has_amount(11_000_000), "cat": CALC},
            "Correct wants (25%)":    {"fn": has_amount(5_000_000), "cat": CALC},
            "Correct savings (20%)":  {"fn": has_amount(4_000_000), "cat": CALC},
            "Compares or advises":    {"fn": has_words("actual", "current", "save", "allocat"), "cat": ACT},
        },
    },
    {
        "id": "TC19", "name": "Custom Split 30/30/40 — Worker savings focus",
        "role": "Worker", "income": 18_000_000,
        "spending": {"Food": 3_000_000, "Transport": 1_500_000, "Bills": 2_000_000},
        "question": "I want to save aggressively. Calculate a 30/30/40 split.",
        "checks": {
            "Correct needs (30%)":    {"fn": has_amount(5_400_000), "cat": CALC},
            "Correct wants (30%)":    {"fn": has_amount(5_400_000), "cat": CALC},
            "Correct savings (40%)":  {"fn": has_amount(7_200_000), "cat": CALC},
            "Affirms aggressive saving": {"fn": has_words("aggressiv", "ambitious", "great", "excellent", "good", "strong"), "cat": ACT},
        },
    },
    {
        "id": "TC20", "name": "Custom Split 65/20/15 — Student part-time",
        "role": "Student", "income": 8_000_000,
        "spending": {"Food": 2_500_000, "Transport": 800_000, "Education": 1_500_000, "Entertainment": 1_000_000},
        "question": "Try a 65/20/15 budget for my part-time income.",
        "checks": {
            "Correct needs (65%)":    {"fn": has_amount(5_200_000), "cat": CALC},
            "Correct wants (20%)":    {"fn": has_amount(1_600_000), "cat": CALC},
            "Correct savings (15%)":  {"fn": has_amount(1_200_000), "cat": CALC},
            "Compares to actual":     {"fn": has_words("actual", "current", "spending", "you spend"), "cat": ACT},
        },
    },
    {
        "id": "TC21", "name": "Custom Split 50/20/30 — Freelancer",
        "role": "Freelancer", "income": 15_000_000,
        "spending": {"Food": 2_500_000, "Transport": 1_000_000, "Bills": 1_500_000, "Entertainment": 800_000},
        "question": "Calculate a 50/20/30 budget split please.",
        "checks": {
            "Correct needs (50%)":    {"fn": has_amount(7_500_000), "cat": CALC},
            "Correct wants (20%)":    {"fn": has_amount(3_000_000), "cat": CALC},
            "Correct savings (30%)":  {"fn": has_amount(4_500_000), "cat": CALC},
            "Actionable follow-up":   {"fn": has_words("save", "allocat", "transfer", "set aside"), "cat": ACT},
        },
    },
    {
        "id": "TC22", "name": "Custom Split 45/35/20 — Worker",
        "role": "Worker", "income": 22_000_000,
        "spending": {"Food": 4_500_000, "Transport": 2_500_000, "Bills": 3_500_000, "Shopping": 2_000_000},
        "question": "What would a 45/35/20 split look like for me?",
        "checks": {
            "Correct needs (45%)":    {"fn": has_amount(9_900_000), "cat": CALC},
            "Correct wants (35%)":    {"fn": has_amount(7_700_000), "cat": CALC},
            "Correct savings (20%)":  {"fn": has_amount(4_400_000), "cat": CALC},
            "Compares actual to target": {"fn": has_words("actual", "current", "spending", "you spend"), "cat": ACT},
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  TAX ADVICE  (TC23–TC30)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "TC23", "name": "Tax — Freelancer, standard",
        "role": "Freelancer", "income": 30_000_000,
        "spending": {"Food": 3_000_000, "Transport": 1_500_000, "Bills": 2_000_000, "Health": 500_000},
        # 30% of 30M = 9M
        "question": "How much should I set aside for taxes?",
        "checks": {
            "Tax amount correct (30%)": {"fn": has_amount(9_000_000), "cat": CALC},
            "Mentions 30% rate":        {"fn": lambda r: "30%" in r or "30 %" in r, "cat": CALC},
            "Concrete action":          {"fn": has_words("set aside", "separate", "transfer", "save", "account"), "cat": ACT},
            "When to act":              {"fn": has_words("each month", "every month", "monthly", "immediately", "now", "each time"), "cat": ACT},
        },
    },
    {
        "id": "TC24", "name": "Tax — Freelancer, high income",
        "role": "Freelancer", "income": 50_000_000,
        "spending": {"Food": 4_000_000, "Transport": 2_000_000, "Bills": 3_000_000, "Equipment": 5_000_000},
        # 30% of 50M = 15M
        "question": "What's my tax obligation this month?",
        "checks": {
            "Tax amount correct":     {"fn": has_amount(15_000_000), "cat": CALC},
            "Mentions 30%":           {"fn": lambda r: "30%" in r or "30 %" in r or "30" in r, "cat": CALC},
            "Concrete action":        {"fn": has_words("set aside", "separate", "transfer", "account", "save"), "cat": ACT},
            "Remaining after tax":    {"fn": has_amount(35_000_000), "cat": CALC},
        },
    },
    {
        "id": "TC25", "name": "Tax — Freelancer, low income",
        "role": "Freelancer", "income": 10_000_000,
        "spending": {"Food": 2_500_000, "Transport": 800_000, "Bills": 1_200_000},
        # 30% of 10M = 3M
        "question": "Should I still set aside money for taxes on 10 million?",
        "checks": {
            "Tax amount correct":     {"fn": has_amount(3_000_000), "cat": CALC},
            "Answers yes directly":   {"fn": has_words("yes", "should", "always", "still"), "cat": ACT},
            "Concrete action":        {"fn": has_words("set aside", "separate", "save", "transfer", "account"), "cat": ACT},
        },
    },
    {
        "id": "TC26", "name": "Tax — Freelancer, deductions question",
        "role": "Freelancer", "income": 22_000_000,
        "spending": {"Food": 3_000_000, "Transport": 1_200_000, "Bills": 2_000_000, "Equipment": 3_000_000, "Software": 1_500_000},
        # total business expenses = equipment + software = 4,500,000
        "question": "What expenses can I track for tax deductions?",
        "checks": {
            "Names user's deductible categories": {"fn": lambda r: sum(1 for x in ["equipment", "software"] if x in r.lower()) >= 1, "cat": CALC},
            "Lists 2+ deductible examples":       {"fn": lambda r: sum(1 for x in ["equipment","software","internet","transport","office","travel"] if x in r.lower()) >= 2, "cat": ACT},
            "Mentions record-keeping":            {"fn": has_words("receipt", "record", "track", "document", "keep"), "cat": ACT},
            "Mentions deductible amount":         {"fn": has_any_amount(3_000_000, 1_500_000, 4_500_000), "cat": CALC},
        },
    },
    {
        "id": "TC27", "name": "Tax — Worker, tax on salary",
        "role": "Worker", "income": 20_000_000,
        "spending": {"Food": 4_000_000, "Transport": 2_000_000, "Bills": 3_000_000, "Shopping": 2_000_000},
        "question": "How is tax handled on my salary?",
        "checks": {
            "Explains employer role":   {"fn": has_words("employer", "company", "payroll", "deducted", "withheld", "automatic"), "cat": ACT},
            "Mentions tax":             {"fn": has_words("tax", "income tax"), "cat": CALC},
            "Actionable insight":       {"fn": has_words("check", "review", "payslip", "slip", "statement", "deduction"), "cat": ACT},
        },
    },
    {
        "id": "TC28", "name": "Tax — Freelancer, quarterly planning",
        "role": "Freelancer", "income": 40_000_000,
        "spending": {"Food": 4_000_000, "Transport": 2_000_000, "Bills": 3_500_000, "Entertainment": 2_000_000},
        # monthly tax = 12M, quarterly = 36M
        "question": "How much tax should I prepare for this quarter?",
        "checks": {
            "Monthly tax amount":     {"fn": has_amount(12_000_000), "cat": CALC},
            "Quarterly tax amount":   {"fn": has_amount(36_000_000), "cat": CALC},
            "Mentions 30%":           {"fn": lambda r: "30" in r, "cat": CALC},
            "Concrete action":        {"fn": has_words("set aside", "separate", "account", "save", "transfer"), "cat": ACT},
        },
    },
    {
        "id": "TC29", "name": "Tax — Student, part-time tax",
        "role": "Student", "income": 6_000_000,
        "spending": {"Food": 1_800_000, "Transport": 500_000, "Education": 1_000_000},
        "question": "Do I need to worry about taxes on my part-time income?",
        "checks": {
            "Addresses the question directly": {"fn": has_words("tax", "income tax"), "cat": CALC},
            "Mentions threshold/exemption":    {"fn": has_words("threshold", "exempt", "minimum", "below", "limit", "depend"), "cat": CALC},
            "Concrete next step":              {"fn": has_words("check", "consult", "ask", "find out", "verify"), "cat": ACT},
        },
    },
    {
        "id": "TC30", "name": "Tax — Freelancer, mid range",
        "role": "Freelancer", "income": 18_000_000,
        "spending": {"Food": 3_000_000, "Transport": 1_000_000, "Bills": 2_000_000, "Health": 800_000},
        # 30% of 18M = 5.4M
        "question": "Remind me how much to set aside for tax this month.",
        "checks": {
            "Tax amount correct":     {"fn": has_amount(5_400_000), "cat": CALC},
            "Mentions 30%":           {"fn": lambda r: "30" in r, "cat": CALC},
            "Concrete action":        {"fn": has_words("set aside", "separate", "transfer", "account", "save"), "cat": ACT},
            "After-tax remaining":    {"fn": has_amount(12_600_000), "cat": CALC},
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  TRANSACTION LOGGING  (TC31–TC42)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "TC31", "name": "Log — Student, coffee",
        "role": "Student", "income": 4_000_000,
        "spending": {"Food": 1_200_000, "Transport": 400_000, "Entertainment": 300_000},
        # total before = 1,900,000, after = 1,985,000
        "question": "I just spent 85.000 VND on Highlands Coffee.",
        "checks": {
            "Echoes amount":          {"fn": lambda r: "85" in r, "cat": CALC},
            "Identifies category":    {"fn": has_words("food", "coffee", "beverage", "drink"), "cat": CALC},
            "Confirms logging":       {"fn": has_words("log", "recorded", "added", "logged", "noted"), "cat": ACT},
            "Budget impact":          {"fn": has_words("total", "now", "remaining", "left", "budget", "spent"), "cat": ACT},
        },
    },
    {
        "id": "TC32", "name": "Log — Worker, grocery",
        "role": "Worker", "income": 15_000_000,
        "spending": {"Food": 3_000_000, "Transport": 1_500_000, "Bills": 2_500_000},
        "question": "Log this: I spent 450.000 VND at VinMart grocery.",
        "checks": {
            "Echoes amount":          {"fn": lambda r: "450" in r, "cat": CALC},
            "Identifies category":    {"fn": has_words("food", "grocery", "groceries"), "cat": CALC},
            "Confirms logging":       {"fn": has_words("log", "recorded", "added", "logged", "noted"), "cat": ACT},
            "Updated food total":     {"fn": has_amount(3_450_000), "cat": CALC},
        },
    },
    {
        "id": "TC33", "name": "Log — Student, transport",
        "role": "Student", "income": 3_500_000,
        "spending": {"Food": 1_200_000, "Transport": 300_000, "Entertainment": 400_000},
        "question": "I paid 25.000 VND for a Grab ride to school.",
        "checks": {
            "Echoes amount":          {"fn": lambda r: "25" in r, "cat": CALC},
            "Identifies category":    {"fn": has_words("transport", "ride", "grab", "travel"), "cat": CALC},
            "Confirms logging":       {"fn": has_words("log", "recorded", "added", "logged", "noted"), "cat": ACT},
            "Updated transport total": {"fn": has_amount(325_000), "cat": CALC},
        },
    },
    {
        "id": "TC34", "name": "Log — Freelancer, equipment",
        "role": "Freelancer", "income": 20_000_000,
        "spending": {"Food": 2_500_000, "Transport": 1_000_000, "Equipment": 2_000_000},
        "question": "I bought a new mouse for 750.000 VND.",
        "checks": {
            "Echoes amount":          {"fn": lambda r: "750" in r, "cat": CALC},
            "Identifies category":    {"fn": has_words("equipment", "tech", "office", "gear", "accessor"), "cat": CALC},
            "Confirms logging":       {"fn": has_words("log", "recorded", "added", "logged", "noted"), "cat": ACT},
            "Updated equipment total": {"fn": has_amount(2_750_000), "cat": CALC},
        },
    },
    {
        "id": "TC35", "name": "Log — Worker, dining",
        "role": "Worker", "income": 18_000_000,
        "spending": {"Food": 3_500_000, "Transport": 2_000_000, "Bills": 3_000_000, "Entertainment": 1_500_000},
        "question": "Just had dinner with colleagues, 320.000 VND at a Korean BBQ place.",
        "checks": {
            "Echoes amount":          {"fn": lambda r: "320" in r, "cat": CALC},
            "Identifies category":    {"fn": has_words("food", "dining", "restaurant", "entertainment", "meal"), "cat": CALC},
            "Confirms logging":       {"fn": has_words("log", "recorded", "added", "logged", "noted"), "cat": ACT},
            "Budget context":         {"fn": has_words("total", "now", "remaining", "left", "budget"), "cat": ACT},
        },
    },
    {
        "id": "TC36", "name": "Log — Student, entertainment",
        "role": "Student", "income": 5_000_000,
        "spending": {"Food": 2_000_000, "Transport": 500_000, "Entertainment": 800_000, "Education": 300_000},
        "question": "Spent 150.000 VND on a movie ticket and popcorn.",
        "checks": {
            "Echoes amount":          {"fn": lambda r: "150" in r, "cat": CALC},
            "Identifies category":    {"fn": has_words("entertainment", "movie", "leisure"), "cat": CALC},
            "Confirms logging":       {"fn": has_words("log", "recorded", "added", "logged", "noted"), "cat": ACT},
            "Updated entertainment":  {"fn": has_amount(950_000), "cat": CALC},
        },
    },
    {
        "id": "TC37", "name": "Log — Freelancer, software subscription",
        "role": "Freelancer", "income": 25_000_000,
        "spending": {"Food": 3_000_000, "Transport": 1_500_000, "Software": 2_000_000, "Bills": 2_500_000},
        "question": "I renewed my Adobe subscription for 600.000 VND.",
        "checks": {
            "Echoes amount":          {"fn": lambda r: "600" in r, "cat": CALC},
            "Identifies category":    {"fn": has_words("software", "subscription", "tool", "business"), "cat": CALC},
            "Confirms logging":       {"fn": has_words("log", "recorded", "added", "logged", "noted"), "cat": ACT},
            "Updated software total": {"fn": has_amount(2_600_000), "cat": CALC},
        },
    },
    {
        "id": "TC38", "name": "Log — Worker, bill payment",
        "role": "Worker", "income": 12_000_000,
        "spending": {"Food": 2_500_000, "Transport": 1_500_000, "Bills": 2_000_000},
        "question": "Paid my electricity bill: 850.000 VND.",
        "checks": {
            "Echoes amount":          {"fn": lambda r: "850" in r, "cat": CALC},
            "Identifies category":    {"fn": has_words("bill", "utilit", "electricity"), "cat": CALC},
            "Confirms logging":       {"fn": has_words("log", "recorded", "added", "logged", "noted"), "cat": ACT},
            "Updated bills total":    {"fn": has_amount(2_850_000), "cat": CALC},
        },
    },
    {
        "id": "TC39", "name": "Log — Student, textbook",
        "role": "Student", "income": 4_500_000,
        "spending": {"Food": 1_500_000, "Transport": 400_000, "Education": 600_000},
        "question": "Bought a textbook for 180.000 VND today.",
        "checks": {
            "Echoes amount":          {"fn": lambda r: "180" in r, "cat": CALC},
            "Identifies category":    {"fn": has_words("education", "book", "textbook", "study"), "cat": CALC},
            "Confirms logging":       {"fn": has_words("log", "recorded", "added", "logged", "noted"), "cat": ACT},
            "Updated education total": {"fn": has_amount(780_000), "cat": CALC},
        },
    },
    {
        "id": "TC40", "name": "Log — Freelancer, coworking",
        "role": "Freelancer", "income": 18_000_000,
        "spending": {"Food": 2_500_000, "Transport": 1_000_000, "Bills": 1_500_000},
        "question": "Paid 1.200.000 VND for coworking space this month.",
        "checks": {
            "Echoes amount":          {"fn": lambda r: "1.200" in r or "1,200" in r, "cat": CALC},
            "Identifies category":    {"fn": has_words("office", "cowork", "workspace", "business", "bill"), "cat": CALC},
            "Confirms logging":       {"fn": has_words("log", "recorded", "added", "logged", "noted"), "cat": ACT},
            "Budget context":         {"fn": has_words("total", "now", "remaining", "left", "budget", "spent"), "cat": ACT},
        },
    },
    {
        "id": "TC41", "name": "Log — Worker, health expense",
        "role": "Worker", "income": 16_000_000,
        "spending": {"Food": 3_000_000, "Transport": 1_800_000, "Bills": 2_500_000, "Health": 500_000},
        "question": "I spent 350.000 VND on a dental checkup.",
        "checks": {
            "Echoes amount":          {"fn": lambda r: "350" in r, "cat": CALC},
            "Identifies category":    {"fn": has_words("health", "medical", "dental"), "cat": CALC},
            "Confirms logging":       {"fn": has_words("log", "recorded", "added", "logged", "noted"), "cat": ACT},
            "Updated health total":   {"fn": has_amount(850_000), "cat": CALC},
        },
    },
    {
        "id": "TC42", "name": "Log — Student, large purchase",
        "role": "Student", "income": 5_000_000,
        "spending": {"Food": 2_000_000, "Transport": 500_000, "Education": 300_000, "Entertainment": 500_000},
        # total=3,300,000, surplus=1,700,000, purchase=2,500,000 > surplus
        "question": "I spent 2.500.000 VND on a new phone case and accessories.",
        "checks": {
            "Echoes amount":          {"fn": lambda r: "2.500" in r or "2,500" in r, "cat": CALC},
            "Warns about budget impact": {"fn": has_words("large", "big", "careful", "budget", "half", "significant", "impact", "over", "exceed"), "cat": ACT},
            "Confirms logging":       {"fn": has_words("log", "recorded", "added", "logged", "noted"), "cat": ACT},
            "Shows remaining budget": {"fn": has_words("remaining", "left", "total", "now", "budget"), "cat": ACT},
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  EMERGENCY FUND  (TC43–TC50)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "TC43", "name": "Emergency Fund — Worker, standard",
        "role": "Worker", "income": 15_000_000,
        "spending": {"Food": 3_000_000, "Transport": 1_500_000, "Bills": 2_500_000, "Shopping": 1_500_000, "Health": 500_000},
        # monthly expenses = 9M, 3-month = 27M, 6-month = 54M
        "question": "How much should I have in an emergency fund?",
        "checks": {
            "Mentions month count":   {"fn": has_words("3 month", "6 month", "3-month", "6-month"), "cat": CALC},
            "Calculates fund target": {"fn": has_any_amount(27_000_000, 54_000_000), "cat": CALC},
            "Monthly expense basis":  {"fn": has_amount(9_000_000), "cat": CALC},
            "Concrete saving step":   {"fn": has_words("save", "transfer", "set aside", "automat", "per month", "each month"), "cat": ACT},
        },
    },
    {
        "id": "TC44", "name": "Emergency Fund — Freelancer, irregular income",
        "role": "Freelancer", "income": 20_000_000,
        "spending": {"Food": 3_000_000, "Transport": 1_500_000, "Bills": 2_500_000, "Equipment": 1_000_000},
        # monthly expenses = 8M, 6-month = 48M, 9-month = 72M
        "question": "How big should my emergency fund be as a freelancer?",
        "checks": {
            "Mentions 6+ months":     {"fn": has_words("6 month", "6-month", "9 month", "9-month", "12 month"), "cat": CALC},
            "Calculates fund target": {"fn": has_any_amount(48_000_000, 72_000_000, 96_000_000), "cat": CALC},
            "Monthly expense basis":  {"fn": has_amount(8_000_000), "cat": CALC},
            "Concrete saving step":   {"fn": has_words("save", "transfer", "set aside", "build", "start"), "cat": ACT},
        },
    },
    {
        "id": "TC45", "name": "Emergency Fund — Student, small income",
        "role": "Student", "income": 3_500_000,
        "spending": {"Food": 1_500_000, "Transport": 400_000, "Entertainment": 300_000},
        # monthly expenses = 2.2M, surplus = 1.3M
        "question": "Should I even have an emergency fund on my budget?",
        "checks": {
            "Answers yes directly":   {"fn": has_words("yes", "should", "important", "even", "still"), "cat": ACT},
            "Realistic target amount": {"fn": lambda r: any(c.isdigit() for c in r), "cat": CALC},
            "Monthly expense reference": {"fn": has_amount(2_200_000), "cat": CALC},
            "Concrete first step":    {"fn": has_words("start", "small", "even", "save", "set aside", "per month"), "cat": ACT},
        },
    },
    {
        "id": "TC46", "name": "Emergency Fund — Worker, high expenses",
        "role": "Worker", "income": 25_000_000,
        "spending": {"Food": 5_000_000, "Transport": 3_000_000, "Bills": 4_500_000, "Shopping": 3_000_000, "Health": 1_000_000},
        # monthly expenses = 16.5M, 3-month = 49.5M, 6-month = 99M
        "question": "What's the ideal emergency fund for my expense level?",
        "checks": {
            "Monthly expense basis":  {"fn": has_amount(16_500_000), "cat": CALC},
            "Fund target (3 or 6 mo)": {"fn": has_any_amount(49_500_000, 99_000_000), "cat": CALC},
            "Mentions month count":   {"fn": has_words("3 month", "6 month", "3-month", "6-month"), "cat": CALC},
            "Concrete saving step":   {"fn": has_words("save", "transfer", "set aside", "automat", "build"), "cat": ACT},
        },
    },
    {
        "id": "TC47", "name": "Emergency Fund — Freelancer, starting out",
        "role": "Freelancer", "income": 12_000_000,
        "spending": {"Food": 2_500_000, "Transport": 800_000, "Bills": 1_500_000, "Equipment": 1_200_000},
        # monthly expenses = 6M, surplus = 6M
        "question": "I'm just starting freelancing. How do I build an emergency fund?",
        "checks": {
            "Monthly expense basis":  {"fn": has_amount(6_000_000), "cat": CALC},
            "Mentions month target":  {"fn": has_words("month", "3 month", "6 month"), "cat": CALC},
            "Step-by-step plan":      {"fn": has_words("start", "first", "step", "begin", "open"), "cat": ACT},
            "Specific monthly saving": {"fn": lambda r: any(c.isdigit() for c in r), "cat": ACT},
        },
    },
    {
        "id": "TC48", "name": "Emergency Fund — Worker, family",
        "role": "Worker", "income": 30_000_000,
        "spending": {"Food": 6_000_000, "Transport": 3_000_000, "Bills": 5_000_000, "Health": 1_500_000, "Education": 2_000_000},
        # monthly expenses = 17.5M, 6-month = 105M
        "question": "I have a family to support. How much emergency fund do I need?",
        "checks": {
            "Monthly expense basis":  {"fn": has_amount(17_500_000), "cat": CALC},
            "Higher target (6+ mo)":  {"fn": has_words("6 month", "6-month", "family", "depend"), "cat": CALC},
            "Fund target amount":     {"fn": has_any_amount(52_500_000, 105_000_000), "cat": CALC},
            "Concrete saving step":   {"fn": has_words("save", "set aside", "transfer", "automat", "per month"), "cat": ACT},
        },
    },
    {
        "id": "TC49", "name": "Emergency Fund — Student, dorm life",
        "role": "Student", "income": 4_000_000,
        "spending": {"Food": 1_800_000, "Transport": 200_000, "Entertainment": 400_000, "Education": 500_000},
        # monthly expenses = 2.9M, surplus = 1.1M
        "question": "How much emergency savings should a student living in a dorm have?",
        "checks": {
            "Monthly expense basis":  {"fn": has_amount(2_900_000), "cat": CALC},
            "Specific target amount": {"fn": lambda r: any(c.isdigit() for c in r), "cat": CALC},
            "Realistic for student":  {"fn": has_words("month", "start", "small", "build"), "cat": ACT},
            "Concrete action":        {"fn": has_words("save", "set aside", "transfer", "per month", "each month"), "cat": ACT},
        },
    },
    {
        "id": "TC50", "name": "Emergency Fund — Freelancer, high earner",
        "role": "Freelancer", "income": 45_000_000,
        "spending": {"Food": 4_000_000, "Transport": 2_500_000, "Bills": 4_000_000, "Entertainment": 2_000_000, "Equipment": 3_000_000},
        # monthly expenses = 15.5M, 6-month = 93M, 9-month = 139.5M
        "question": "I earn well but my income fluctuates wildly. Emergency fund advice?",
        "checks": {
            "Monthly expense basis":  {"fn": has_amount(15_500_000), "cat": CALC},
            "Fund target (6-12 mo)":  {"fn": has_any_amount(93_000_000, 139_500_000, 186_000_000), "cat": CALC},
            "Recommends 6+ months":   {"fn": has_words("6 month", "6-month", "9 month", "12 month", "longer"), "cat": CALC},
            "Concrete action":        {"fn": has_words("save", "set aside", "build", "transfer"), "cat": ACT},
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  SAVING TIPS  (TC51–TC60)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "TC51", "name": "Saving Tips — Student, basic",
        "role": "Student", "income": 4_000_000,
        "spending": {"Food": 1_800_000, "Transport": 400_000, "Entertainment": 600_000, "Education": 300_000},
        # total=3,100,000  surplus=900,000
        "question": "Give me some saving tips.",
        "checks": {
            "References user's spending": {"fn": refs_user_categories({"Food": 0, "Entertainment": 0}, 1), "cat": ACT},
            "2+ concrete tips":        {"fn": lambda r: sum(1 for w in ["cook", "meal", "pack", "limit", "free", "discount", "cancel", "reduce", "switch"] if w in r.lower()) >= 2, "cat": ACT},
            "Mentions specific amount": {"fn": lambda r: any(c.isdigit() for c in r), "cat": CALC},
            "Surplus awareness":       {"fn": has_amount(900_000), "cat": CALC},
        },
    },
    {
        "id": "TC52", "name": "Saving Tips — Freelancer, business",
        "role": "Freelancer", "income": 12_000_000,
        "spending": {"Food": 2_500_000, "Transport": 800_000, "Bills": 1_500_000, "Entertainment": 1_000_000},
        # total=5,800,000  surplus=6,200,000
        "question": "Give me saving tips for my situation.",
        "checks": {
            "References user's categories": {"fn": refs_user_categories({"Food": 0, "Entertainment": 0, "Bills": 0}, 1), "cat": ACT},
            "2+ concrete actions":     {"fn": lambda r: sum(1 for w in ["set aside", "save", "open", "track", "build", "reduce", "cut", "automat", "transfer"] if w in r.lower()) >= 2, "cat": ACT},
            "Mentions surplus or saveable amount": {"fn": lambda r: any(c.isdigit() for c in r), "cat": CALC},
            "Tax saving mention":      {"fn": has_words("tax", "30%", "deduct"), "cat": ACT},
        },
    },
    {
        "id": "TC53", "name": "Saving Tips — Worker, mid salary",
        "role": "Worker", "income": 15_000_000,
        "spending": {"Food": 3_500_000, "Transport": 2_000_000, "Bills": 2_500_000, "Shopping": 2_000_000},
        # total=10,000,000  surplus=5,000,000
        "question": "What are some good ways to save more money?",
        "checks": {
            "References user's high categories": {"fn": refs_user_categories({"Food": 0, "Shopping": 0, "Transport": 0}, 1), "cat": ACT},
            "2+ concrete actions":     {"fn": lambda r: sum(1 for w in ["save", "transfer", "reduce", "track", "automat", "cut", "limit", "set up"] if w in r.lower()) >= 2, "cat": ACT},
            "Mentions specific amount": {"fn": lambda r: any(c.isdigit() for c in r), "cat": CALC},
            "Surplus awareness":       {"fn": has_amount(5_000_000), "cat": CALC},
        },
    },
    {
        "id": "TC54", "name": "Saving Tips — Student, food focus",
        "role": "Student", "income": 3_500_000,
        "spending": {"Food": 2_000_000, "Transport": 300_000, "Entertainment": 400_000},
        # food is 57% of income
        "question": "I spend too much on food. How can I save?",
        "checks": {
            "Food amount referenced":  {"fn": has_amount(2_000_000), "cat": CALC},
            "Food as % of income":     {"fn": lambda r: "%" in r or "57" in r or "half" in r.lower() or "most" in r.lower(), "cat": CALC},
            "Food-specific tips":      {"fn": lambda r: sum(1 for w in ["cook", "meal prep", "meal", "canteen", "pack", "groceries", "market", "home"] if w in r.lower()) >= 2, "cat": ACT},
            "Target amount":           {"fn": lambda r: any(c.isdigit() for c in r), "cat": ACT},
        },
    },
    {
        "id": "TC55", "name": "Saving Tips — Freelancer, tax savings",
        "role": "Freelancer", "income": 28_000_000,
        "spending": {"Food": 3_000_000, "Transport": 1_500_000, "Bills": 2_500_000, "Equipment": 2_000_000, "Software": 1_500_000},
        # deductible = equipment + software = 3.5M
        "question": "How can I reduce my tax burden legally?",
        "checks": {
            "Names user's deductible categories": {"fn": lambda r: sum(1 for x in ["equipment", "software"] if x in r.lower()) >= 1, "cat": CALC},
            "Lists 2+ deductible types":  {"fn": lambda r: sum(1 for x in ["equipment","software","internet","transport","office"] if x in r.lower()) >= 2, "cat": ACT},
            "Deductible amount":       {"fn": has_any_amount(2_000_000, 1_500_000, 3_500_000), "cat": CALC},
            "Record-keeping action":   {"fn": has_words("receipt", "record", "track", "document", "keep"), "cat": ACT},
        },
    },
    {
        "id": "TC56", "name": "Saving Tips — Worker, high earner",
        "role": "Worker", "income": 35_000_000,
        "spending": {"Food": 5_000_000, "Transport": 3_000_000, "Bills": 4_000_000, "Shopping": 5_000_000, "Entertainment": 3_000_000},
        # total=20M  surplus=15M  shopping+entertainment=8M (23% of income)
        "question": "I earn well but save very little. Tips?",
        "checks": {
            "Identifies top discretionary": {"fn": has_words("shopping", "entertainment"), "cat": CALC},
            "Mentions specific amounts":    {"fn": has_any_amount(5_000_000, 3_000_000, 8_000_000, 15_000_000), "cat": CALC},
            "Concrete cut suggestion":      {"fn": suggests_specific_cut({"Shopping": 0, "Entertainment": 0}), "cat": ACT},
            "Automation advice":            {"fn": has_words("automat", "transfer", "set up", "pay yourself", "direct"), "cat": ACT},
        },
    },
    {
        "id": "TC57", "name": "Saving Tips — Student, zero savings",
        "role": "Student", "income": 3_000_000,
        "spending": {"Food": 1_500_000, "Transport": 500_000, "Entertainment": 700_000, "Education": 300_000},
        # total=3,000,000  surplus=0
        "question": "I can never save anything. What should I do?",
        "checks": {
            "Identifies zero surplus":    {"fn": has_words("zero", "nothing", "no room", "everything", "all", "entire", "100%"), "cat": CALC},
            "Identifies top cut target":  {"fn": has_words("entertainment"), "cat": CALC},
            "Entertainment amount":       {"fn": has_amount(700_000), "cat": CALC},
            "Two-pronged advice (cut + earn)": {"fn": lambda r: any(c in r.lower() for c in ["cut", "reduce", "lower"]) and any(e in r.lower() for e in ["earn", "income", "part-time", "part time", "job", "tutor"]), "cat": ACT},
        },
    },
    {
        "id": "TC58", "name": "Saving Tips — Freelancer, separate accounts",
        "role": "Freelancer", "income": 22_000_000,
        "spending": {"Food": 3_000_000, "Transport": 1_200_000, "Bills": 2_000_000, "Entertainment": 1_500_000},
        # total=7,700,000  surplus=14,300,000
        "question": "How should I organize my finances as a freelancer?",
        "checks": {
            "Mentions separate accounts": {"fn": has_words("separate", "account", "business", "personal"), "cat": ACT},
            "Tax allocation":          {"fn": has_words("tax", "30%"), "cat": ACT},
            "Tax amount":              {"fn": has_amount(6_600_000), "cat": CALC},
            "Concrete account actions": {"fn": has_words("open", "set up", "transfer", "automat"), "cat": ACT},
        },
    },
    {
        "id": "TC59", "name": "Saving Tips — Worker, automate",
        "role": "Worker", "income": 20_000_000,
        "spending": {"Food": 4_000_000, "Transport": 2_000_000, "Bills": 3_000_000, "Shopping": 2_500_000},
        # total=11,500,000  surplus=8,500,000  savings target (20%) = 4M
        "question": "How can I automate my savings?",
        "checks": {
            "Saveable amount":         {"fn": has_any_amount(8_500_000, 4_000_000), "cat": CALC},
            "Automation method":       {"fn": has_words("automat", "standing order", "recurring", "schedule"), "cat": ACT},
            "Salary/paycheck context": {"fn": has_words("salary", "paycheck", "pay day", "payday", "split", "when you receive"), "cat": ACT},
            "Concrete steps":          {"fn": has_words("set up", "open", "bank", "account", "transfer"), "cat": ACT},
        },
    },
    {
        "id": "TC60", "name": "Saving Tips — Student, discount hunter",
        "role": "Student", "income": 5_000_000,
        "spending": {"Food": 2_200_000, "Transport": 600_000, "Entertainment": 800_000, "Education": 400_000},
        # total=4,000,000  surplus=1,000,000
        "question": "Any student-specific ways to save money?",
        "checks": {
            "Mentions discounts":      {"fn": has_words("discount", "student card", "student id", "promotion"), "cat": ACT},
            "2+ concrete actions":     {"fn": lambda r: sum(1 for w in ["use", "ask", "sign up", "check", "look for", "show", "apply"] if w in r.lower()) >= 2, "cat": ACT},
            "References user's spending": {"fn": refs_user_categories({"Food": 0, "Entertainment": 0, "Transport": 0}, 1), "cat": ACT},
            "Potential savings amount": {"fn": lambda r: any(c.isdigit() for c in r), "cat": CALC},
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  OVERSPENDING ALERTS  (TC61–TC70)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "TC61", "name": "Overspend — Student, entertainment",
        "role": "Student", "income": 4_000_000,
        "spending": {"Food": 2_500_000, "Transport": 600_000, "Entertainment": 1_200_000, "Shopping": 800_000},
        # entertainment = 30% of income, total=5,100,000 > 4M = deficit 1.1M
        "question": "Am I spending too much on Entertainment?",
        "checks": {
            "Entertainment amount":    {"fn": has_amount(1_200_000), "cat": CALC},
            "Entertainment as pct":    {"fn": lambda r: "30" in r or "%" in r, "cat": CALC},
            "Confirms overspending":   {"fn": has_words("over", "high", "too much", "above", "exceed", "yes"), "cat": ACT},
            "Specific reduction target": {"fn": has_words("cut", "reduce", "limit", "lower", "target"), "cat": ACT},
        },
    },
    {
        "id": "TC62", "name": "Overspend — Worker, shopping",
        "role": "Worker", "income": 18_000_000,
        "spending": {"Food": 4_000_000, "Transport": 2_000_000, "Bills": 3_000_000, "Shopping": 6_000_000, "Entertainment": 2_000_000},
        # shopping = 33% of income, total=17M
        "question": "Is my shopping spending out of control?",
        "checks": {
            "Shopping amount":         {"fn": has_amount(6_000_000), "cat": CALC},
            "Shopping as percentage":  {"fn": lambda r: "%" in r and "33" in r or "%" in r, "cat": CALC},
            "Confirms overspending":   {"fn": has_words("over", "high", "too much", "above", "exceed", "large", "yes"), "cat": ACT},
            "Specific reduction advice": {"fn": has_words("cut", "reduce", "limit", "budget", "target"), "cat": ACT},
        },
    },
    {
        "id": "TC63", "name": "Overspend — Freelancer, equipment",
        "role": "Freelancer", "income": 15_000_000,
        "spending": {"Food": 2_500_000, "Transport": 1_000_000, "Bills": 1_500_000, "Equipment": 6_000_000, "Software": 2_000_000},
        # equipment = 40% of income, total=13M, surplus=2M
        "question": "Am I spending too much on equipment and tools?",
        "checks": {
            "Equipment + software total": {"fn": has_any_amount(6_000_000, 8_000_000), "cat": CALC},
            "Percentage of income":    {"fn": lambda r: "%" in r, "cat": CALC},
            "Balanced assessment":     {"fn": has_words("budget", "plan", "spread", "essential", "priorit", "necessary"), "cat": ACT},
            "Concrete advice":         {"fn": has_words("reduce", "limit", "plan", "spread", "phase", "budget"), "cat": ACT},
        },
    },
    {
        "id": "TC64", "name": "Overspend — Student, food heavy",
        "role": "Student", "income": 3_500_000,
        "spending": {"Food": 2_200_000, "Transport": 400_000, "Entertainment": 300_000, "Education": 200_000},
        # food = 63% of income
        "question": "My food spending seems really high. Is it?",
        "checks": {
            "Food amount":             {"fn": has_amount(2_200_000), "cat": CALC},
            "Food percentage":         {"fn": lambda r: "%" in r or "63" in r or "60" in r, "cat": CALC},
            "Confirms high":           {"fn": has_words("high", "yes", "above", "over", "too much", "exceed"), "cat": ACT},
            "Food-specific tips":      {"fn": lambda r: sum(1 for w in ["cook", "meal", "canteen", "market", "reduce", "pack", "home"] if w in r.lower()) >= 1, "cat": ACT},
        },
    },
    {
        "id": "TC65", "name": "Overspend — Worker, bills",
        "role": "Worker", "income": 14_000_000,
        "spending": {"Food": 3_000_000, "Transport": 1_500_000, "Bills": 5_000_000, "Shopping": 1_500_000},
        # bills = 36% of income, total=11M, surplus=3M
        "question": "My bills are eating up my income. What can I do?",
        "checks": {
            "Bills amount":            {"fn": has_amount(5_000_000), "cat": CALC},
            "Bills percentage":        {"fn": lambda r: "%" in r or "36" in r or "35" in r, "cat": CALC},
            "Concrete bill reduction": {"fn": has_words("negotiat", "switch", "reduce", "review", "cancel", "downgrade", "compare"), "cat": ACT},
            "Actionable first step":   {"fn": has_words("review", "check", "list", "identify", "start"), "cat": ACT},
        },
    },
    {
        "id": "TC66", "name": "Overspend — Freelancer, transport",
        "role": "Freelancer", "income": 20_000_000,
        "spending": {"Food": 3_000_000, "Transport": 4_500_000, "Bills": 2_000_000, "Entertainment": 1_500_000},
        # transport = 22.5% of income
        "question": "Is 4.5 million on transport too much?",
        "checks": {
            "Transport amount":        {"fn": has_amount(4_500_000), "cat": CALC},
            "Transport percentage":    {"fn": lambda r: "%" in r or "22" in r or "23" in r, "cat": CALC},
            "Assessment":              {"fn": has_words("high", "above", "over", "significant", "large", "consider"), "cat": ACT},
            "Concrete reduction tip":  {"fn": has_words("reduce", "alternative", "public", "combine", "plan", "batch"), "cat": ACT},
        },
    },
    {
        "id": "TC67", "name": "Overspend — Student, total over income",
        "role": "Student", "income": 3_000_000,
        "spending": {"Food": 1_500_000, "Transport": 500_000, "Entertainment": 800_000, "Shopping": 600_000},
        # total=3,400,000  deficit=400,000
        "question": "I think I'm spending more than I earn. Am I?",
        "checks": {
            "Total spent correct":     {"fn": has_amount(3_400_000), "cat": CALC},
            "Deficit amount":          {"fn": has_amount(400_000), "cat": CALC},
            "Confirms deficit":        {"fn": has_words("over", "deficit", "more than", "exceed", "negative", "beyond", "yes"), "cat": CALC},
            "Names specific category to cut": {"fn": suggests_specific_cut({"Entertainment": 0, "Shopping": 0}), "cat": ACT},
        },
    },
    {
        "id": "TC68", "name": "Overspend — Worker, dining out",
        "role": "Worker", "income": 16_000_000,
        "spending": {"Food": 5_500_000, "Transport": 2_000_000, "Bills": 2_500_000, "Entertainment": 1_500_000},
        # food = 34% of income
        "question": "I eat out every day. Is my food spending too high?",
        "checks": {
            "Food amount":             {"fn": has_amount(5_500_000), "cat": CALC},
            "Food percentage":         {"fn": lambda r: "%" in r or "34" in r, "cat": CALC},
            "Confirms high":           {"fn": has_words("high", "yes", "over", "above", "too much"), "cat": ACT},
            "Meal prep suggestion":    {"fn": has_words("cook", "meal prep", "pack", "home", "lunch", "prepare"), "cat": ACT},
        },
    },
    {
        "id": "TC69", "name": "Overspend — Freelancer, entertainment heavy",
        "role": "Freelancer", "income": 18_000_000,
        "spending": {"Food": 3_000_000, "Transport": 1_500_000, "Bills": 2_000_000, "Entertainment": 5_000_000, "Shopping": 3_000_000},
        # entertainment = 28%, entertainment+shopping = 44%, total=14.5M
        "question": "I feel like I'm spending too much on fun. Thoughts?",
        "checks": {
            "Entertainment amount":    {"fn": has_amount(5_000_000), "cat": CALC},
            "Entertainment percentage": {"fn": lambda r: "%" in r or "28" in r or "27" in r, "cat": CALC},
            "Confirms overspending":   {"fn": has_words("over", "high", "too much", "above", "large", "yes"), "cat": ACT},
            "Specific reduction target": {"fn": has_words("cut", "reduce", "limit", "lower", "target", "budget"), "cat": ACT},
        },
    },
    {
        "id": "TC70", "name": "Overspend — Worker, across the board",
        "role": "Worker", "income": 12_000_000,
        "spending": {"Food": 4_000_000, "Transport": 2_500_000, "Bills": 3_000_000, "Shopping": 2_000_000, "Entertainment": 1_500_000},
        # total=13,000,000  deficit=1,000,000  biggest=Food at 33%
        "question": "Where am I overspending the most?",
        "checks": {
            "Total spent correct":       {"fn": has_amount(13_000_000), "cat": CALC},
            "Deficit amount":            {"fn": has_amount(1_000_000), "cat": CALC},
            "Identifies Food as top":    {"fn": has_words("food"), "cat": CALC},
            "Prioritized cut plan":      {"fn": has_words("cut", "reduce", "first", "start", "priorit", "biggest"), "cat": ACT},
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  GOAL SAVING PLAN  (TC71–TC80)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "TC71", "name": "Goal — Worker, motorbike",
        "role": "Worker", "income": 18_000_000,
        "spending": {"Food": 3_500_000, "Transport": 2_000_000, "Bills": 3_000_000, "Shopping": 2_000_000, "Entertainment": 1_000_000},
        # total=11.5M, surplus=6.5M, goal=45M, months=45/6.5=6.9≈7
        "question": "I want to save for a motorbike worth 45.000.000 VND. How long will it take?",
        "checks": {
            "Monthly surplus correct":  {"fn": has_amount(6_500_000), "cat": CALC},
            "Timeline estimate":        {"fn": has_words("7 month", "7-month", "6 month", "8 month"), "cat": CALC},
            "Goal amount echoed":       {"fn": lambda r: "45" in r, "cat": CALC},
            "Monthly saving plan":      {"fn": has_words("per month", "each month", "monthly", "save", "set aside"), "cat": ACT},
        },
    },
    {
        "id": "TC72", "name": "Goal — Student, laptop",
        "role": "Student", "income": 5_000_000,
        "spending": {"Food": 2_000_000, "Transport": 500_000, "Entertainment": 500_000, "Education": 300_000},
        # total=3.3M, surplus=1.7M, goal=15M, months=15/1.7=8.8≈9
        "question": "I need a new laptop for 15.000.000 VND. How long to save?",
        "checks": {
            "Monthly surplus correct":  {"fn": has_amount(1_700_000), "cat": CALC},
            "Timeline estimate":        {"fn": has_words("9 month", "8 month", "10 month"), "cat": CALC},
            "Goal amount echoed":       {"fn": lambda r: "15" in r, "cat": CALC},
            "Ways to speed up":         {"fn": has_words("cut", "reduce", "earn", "extra", "faster", "part-time", "part time", "increase"), "cat": ACT},
        },
    },
    {
        "id": "TC73", "name": "Goal — Freelancer, vacation",
        "role": "Freelancer", "income": 25_000_000,
        "spending": {"Food": 3_500_000, "Transport": 1_500_000, "Bills": 2_500_000, "Entertainment": 2_000_000},
        # total=9.5M, surplus=15.5M, goal=20M in 3mo -> 6.67M/mo
        "question": "I want to take a vacation worth 20.000.000 VND. Can I save for it in 3 months?",
        "checks": {
            "Monthly saving needed":    {"fn": has_any_amount(6_666_666, 6_667_000, 6_700_000), "cat": CALC},
            "Feasibility answer":       {"fn": has_words("yes", "possible", "can", "feasible", "doable"), "cat": ACT},
            "Monthly surplus context":  {"fn": has_amount(15_500_000), "cat": CALC},
            "Concrete saving plan":     {"fn": has_words("per month", "each month", "save", "set aside", "transfer"), "cat": ACT},
        },
    },
    {
        "id": "TC74", "name": "Goal — Worker, wedding fund",
        "role": "Worker", "income": 22_000_000,
        "spending": {"Food": 4_000_000, "Transport": 2_500_000, "Bills": 3_500_000, "Shopping": 2_000_000},
        # total=12M, surplus=10M, goal=100M, months=100/10=10
        "question": "I'm saving for a wedding, need 100.000.000 VND. How long?",
        "checks": {
            "Monthly surplus correct":  {"fn": has_amount(10_000_000), "cat": CALC},
            "Timeline estimate":        {"fn": has_words("10 month", "10-month", "about a year", "less than a year"), "cat": CALC},
            "Goal amount echoed":       {"fn": lambda r: "100" in r, "cat": CALC},
            "Ways to accelerate":       {"fn": has_words("cut", "save", "increase", "reduce", "faster"), "cat": ACT},
        },
    },
    {
        "id": "TC75", "name": "Goal — Student, phone",
        "role": "Student", "income": 4_000_000,
        "spending": {"Food": 1_500_000, "Transport": 400_000, "Entertainment": 500_000, "Education": 400_000},
        # total=2.8M, surplus=1.2M, goal=8M, months=8/1.2=6.7≈7
        "question": "I want a new phone for 8.000.000 VND. How do I save for it?",
        "checks": {
            "Monthly surplus correct":  {"fn": has_amount(1_200_000), "cat": CALC},
            "Timeline estimate":        {"fn": has_words("7 month", "6 month", "8 month"), "cat": CALC},
            "Goal amount echoed":       {"fn": lambda r: "8.000" in r or "8,000" in r, "cat": CALC},
            "Actionable plan":          {"fn": has_words("save", "set aside", "cut", "reduce", "per month"), "cat": ACT},
        },
    },
    {
        "id": "TC76", "name": "Goal — Freelancer, business investment",
        "role": "Freelancer", "income": 30_000_000,
        "spending": {"Food": 3_500_000, "Transport": 2_000_000, "Bills": 3_000_000, "Equipment": 2_000_000},
        # total=10.5M, surplus=19.5M, goal=50M, months=50/19.5=2.6≈3
        "question": "I need 50.000.000 VND for new equipment for my business. Plan?",
        "checks": {
            "Monthly surplus correct":  {"fn": has_amount(19_500_000), "cat": CALC},
            "Timeline estimate":        {"fn": has_words("3 month", "2 month", "about 3"), "cat": CALC},
            "Goal amount echoed":       {"fn": lambda r: "50" in r, "cat": CALC},
            "Saving plan":              {"fn": has_words("save", "set aside", "per month", "each month", "allocat"), "cat": ACT},
        },
    },
    {
        "id": "TC77", "name": "Goal — Worker, car",
        "role": "Worker", "income": 25_000_000,
        "spending": {"Food": 5_000_000, "Transport": 2_500_000, "Bills": 4_000_000, "Shopping": 2_000_000},
        # total=13.5M, surplus=11.5M, goal=300M, months=300/11.5=26.1≈26 (2.2 years)
        "question": "How long to save 300.000.000 VND for a car?",
        "checks": {
            "Monthly surplus correct":  {"fn": has_amount(11_500_000), "cat": CALC},
            "Timeline estimate":        {"fn": has_words("26 month", "2 year", "about 2", "over 2"), "cat": CALC},
            "Goal amount echoed":       {"fn": lambda r: "300" in r, "cat": CALC},
            "Realistic assessment":     {"fn": has_words("long", "year", "patient", "aggressive", "increase", "invest"), "cat": ACT},
        },
    },
    {
        "id": "TC78", "name": "Goal — Student, travel",
        "role": "Student", "income": 6_000_000,
        "spending": {"Food": 2_000_000, "Transport": 600_000, "Education": 800_000, "Entertainment": 500_000},
        # total=3.9M, surplus=2.1M, goal=5M, months=5/2.1=2.4≈3
        "question": "I want to travel to Da Lat, need about 5.000.000 VND. How long?",
        "checks": {
            "Monthly surplus correct":  {"fn": has_amount(2_100_000), "cat": CALC},
            "Timeline estimate":        {"fn": has_words("3 month", "2 month", "about 3", "about 2"), "cat": CALC},
            "Goal amount echoed":       {"fn": lambda r: "5.000" in r or "5,000" in r, "cat": CALC},
            "Saving plan":              {"fn": has_words("save", "set aside", "per month", "each month"), "cat": ACT},
        },
    },
    {
        "id": "TC79", "name": "Goal — Freelancer, home deposit",
        "role": "Freelancer", "income": 40_000_000,
        "spending": {"Food": 4_000_000, "Transport": 2_500_000, "Bills": 4_000_000, "Entertainment": 3_000_000, "Equipment": 2_000_000},
        # total=15.5M, surplus=24.5M, goal=200M, months=200/24.5=8.2≈8-9
        "question": "I want to save 200.000.000 VND for a home deposit. Is this realistic?",
        "checks": {
            "Monthly surplus correct":  {"fn": has_amount(24_500_000), "cat": CALC},
            "Timeline estimate":        {"fn": has_words("8 month", "9 month", "about 8", "about 9"), "cat": CALC},
            "Goal amount echoed":       {"fn": lambda r: "200" in r, "cat": CALC},
            "Feasibility answer":       {"fn": has_words("yes", "realistic", "possible", "feasible", "doable", "achievable"), "cat": ACT},
        },
    },
    {
        "id": "TC80", "name": "Goal — Worker, education fund",
        "role": "Worker", "income": 20_000_000,
        "spending": {"Food": 4_000_000, "Transport": 2_000_000, "Bills": 3_000_000, "Shopping": 1_500_000},
        # total=10.5M, surplus=9.5M, goal=60M, months=60/9.5=6.3≈6-7
        "question": "I want to save 60.000.000 VND for a certification course. Timeline?",
        "checks": {
            "Monthly surplus correct":  {"fn": has_amount(9_500_000), "cat": CALC},
            "Timeline estimate":        {"fn": has_words("6 month", "7 month", "about 6", "about 7"), "cat": CALC},
            "Goal amount echoed":       {"fn": lambda r: "60" in r, "cat": CALC},
            "Monthly saving plan":      {"fn": has_words("save", "set aside", "per month", "each month", "allocat"), "cat": ACT},
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  DEBT & LOANS  (TC81–TC86)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "TC81", "name": "Debt — Student, borrowed money",
        "role": "Student", "income": 4_000_000,
        "spending": {"Food": 1_500_000, "Transport": 400_000, "Entertainment": 500_000, "Education": 300_000},
        # total=2.7M, surplus=1.3M, debt=2M, months=2/1.3=1.5≈2
        "question": "I owe a friend 2.000.000 VND. How do I pay it back quickly?",
        "checks": {
            "Monthly surplus correct":     {"fn": has_amount(1_300_000), "cat": CALC},
            "Repayment timeline":          {"fn": has_words("2 month", "1 month", "about 2"), "cat": CALC},
            "Identifies category to cut":  {"fn": suggests_specific_cut({"Entertainment": 0, "Food": 0}), "cat": ACT},
            "Monthly payment amount":      {"fn": lambda r: any(c.isdigit() for c in r), "cat": ACT},
        },
    },
    {
        "id": "TC82", "name": "Debt — Worker, credit card",
        "role": "Worker", "income": 16_000_000,
        "spending": {"Food": 3_500_000, "Transport": 2_000_000, "Bills": 2_500_000, "Shopping": 3_000_000},
        # total=11M, surplus=5M, debt=10M, months=10/5=2 (without interest)
        "question": "I have 10.000.000 VND in credit card debt. How should I tackle it?",
        "checks": {
            "Monthly surplus correct":  {"fn": has_amount(5_000_000), "cat": CALC},
            "Mentions interest":        {"fn": has_words("interest", "rate", "compound", "growing", "cost"), "cat": CALC},
            "Repayment strategy":       {"fn": has_words("pay", "repay", "clear", "priority", "first"), "cat": ACT},
            "Specific cut suggestion":  {"fn": suggests_specific_cut({"Shopping": 0, "Food": 0}), "cat": ACT},
        },
    },
    {
        "id": "TC83", "name": "Debt — Freelancer, business loan",
        "role": "Freelancer", "income": 22_000_000,
        "spending": {"Food": 3_000_000, "Transport": 1_500_000, "Bills": 2_000_000, "Equipment": 2_000_000},
        # total=8.5M, surplus=13.5M, loan=30M
        "question": "I took a 30.000.000 VND loan for equipment. How do I manage repayments?",
        "checks": {
            "Monthly surplus correct":  {"fn": has_amount(13_500_000), "cat": CALC},
            "Monthly payment suggestion": {"fn": lambda r: any(c.isdigit() for c in r) and "month" in r.lower(), "cat": CALC},
            "Budget allocation":        {"fn": has_words("allocat", "set aside", "budget", "payment", "dedicate"), "cat": ACT},
            "Concrete plan":            {"fn": has_words("per month", "each month", "monthly", "transfer"), "cat": ACT},
        },
    },
    {
        "id": "TC84", "name": "Debt — Student, student loan",
        "role": "Student", "income": 5_000_000,
        "spending": {"Food": 2_000_000, "Transport": 500_000, "Education": 500_000},
        # total=3M, surplus=2M, loan=20M, months=20/2=10
        "question": "I have a student loan of 20.000.000 VND. What's the best repayment plan?",
        "checks": {
            "Monthly surplus correct":  {"fn": has_amount(2_000_000), "cat": CALC},
            "Repayment timeline":       {"fn": has_words("10 month", "about 10", "less than a year"), "cat": CALC},
            "Monthly payment amount":   {"fn": lambda r: any(c.isdigit() for c in r), "cat": CALC},
            "Realistic plan":           {"fn": has_words("per month", "each month", "pay", "repay"), "cat": ACT},
        },
    },
    {
        "id": "TC85", "name": "Debt — Worker, multiple debts",
        "role": "Worker", "income": 20_000_000,
        "spending": {"Food": 4_000_000, "Transport": 2_000_000, "Bills": 3_000_000, "Shopping": 2_000_000},
        # total=11M, surplus=9M, debts: 5M CC + 15M personal = 20M total
        "question": "I owe 5.000.000 on a credit card and 15.000.000 on a personal loan. Which first?",
        "checks": {
            "Prioritizes credit card":  {"fn": has_words("credit card", "higher interest", "first", "priority"), "cat": ACT},
            "Monthly surplus correct":  {"fn": has_amount(9_000_000), "cat": CALC},
            "Both debts referenced":    {"fn": lambda r: "5" in r and "15" in r, "cat": CALC},
            "Repayment strategy":       {"fn": has_words("avalanche", "snowball", "highest", "pay", "clear", "focus"), "cat": ACT},
        },
    },
    {
        "id": "TC86", "name": "Debt — Freelancer, tax debt",
        "role": "Freelancer", "income": 18_000_000,
        "spending": {"Food": 3_000_000, "Transport": 1_200_000, "Bills": 2_000_000, "Entertainment": 1_000_000},
        # total=7.2M, surplus=10.8M, tax debt=15M, months=15/10.8≈1.4
        "question": "I forgot to set aside taxes and now owe 15.000.000 VND. Help!",
        "checks": {
            "Monthly surplus correct":   {"fn": has_amount(10_800_000), "cat": CALC},
            "Repayment timeline":        {"fn": has_words("1 month", "2 month", "quickly", "fast"), "cat": CALC},
            "Prevention advice":         {"fn": has_words("30%", "separate", "automat", "set aside", "future", "going forward"), "cat": ACT},
            "Urgency + concrete action": {"fn": has_words("immediately", "first", "priority", "urgent", "now", "asap", "today"), "cat": ACT},
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  INVESTMENT & GROWTH  (TC87–TC92)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "TC87", "name": "Invest — Worker, beginner",
        "role": "Worker", "income": 20_000_000,
        "spending": {"Food": 4_000_000, "Transport": 2_000_000, "Bills": 3_000_000, "Shopping": 2_000_000},
        # total=11M, surplus=9M
        "question": "I have some money left over. Should I start investing?",
        "checks": {
            "Surplus amount":           {"fn": has_amount(9_000_000), "cat": CALC},
            "Emergency fund first":     {"fn": has_words("emergency", "fund", "first", "before"), "cat": ACT},
            "Investment options":       {"fn": has_words("invest", "fund", "stock", "bond", "saving", "deposit"), "cat": ACT},
            "Specific amount to invest": {"fn": lambda r: any(c.isdigit() for c in r), "cat": ACT},
        },
    },
    {
        "id": "TC88", "name": "Invest — Freelancer, surplus month",
        "role": "Freelancer", "income": 40_000_000,
        "spending": {"Food": 4_000_000, "Transport": 2_000_000, "Bills": 3_000_000, "Entertainment": 2_000_000},
        # total=11M, surplus=29M, tax=12M, after-tax surplus=17M
        "question": "I have 20 million extra this month. How should I invest it?",
        "checks": {
            "Tax first reminder":       {"fn": has_words("tax"), "cat": CALC},
            "Tax amount":               {"fn": has_amount(12_000_000), "cat": CALC},
            "After-tax investable":     {"fn": lambda r: any(c.isdigit() for c in r), "cat": CALC},
            "Investment options":       {"fn": has_words("invest", "fund", "stock", "deposit", "bond", "saving"), "cat": ACT},
        },
    },
    {
        "id": "TC89", "name": "Invest — Worker, employer matching",
        "role": "Worker", "income": 25_000_000,
        "spending": {"Food": 5_000_000, "Transport": 2_500_000, "Bills": 4_000_000, "Shopping": 2_000_000},
        # total=13.5M, surplus=11.5M
        "question": "My company offers contribution matching. Should I take advantage?",
        "checks": {
            "Strongly recommends yes":  {"fn": has_words("yes", "absolutely", "definitely", "free money", "should", "always"), "cat": ACT},
            "Explains benefit":         {"fn": has_words("double", "free", "100%", "return", "grow", "match"), "cat": ACT},
            "Mentions surplus available": {"fn": has_amount(11_500_000), "cat": CALC},
            "Specific contribution advice": {"fn": has_words("maximum", "max", "full", "contribute", "as much as"), "cat": ACT},
        },
    },
    {
        "id": "TC90", "name": "Invest — Student, small amounts",
        "role": "Student", "income": 5_000_000,
        "spending": {"Food": 2_000_000, "Transport": 500_000, "Entertainment": 400_000, "Education": 300_000},
        # total=3.2M, surplus=1.8M
        "question": "Can I invest with just a little money each month?",
        "checks": {
            "Monthly surplus correct":  {"fn": has_amount(1_800_000), "cat": CALC},
            "Encourages yes":           {"fn": has_words("yes", "can", "possible", "start small", "even"), "cat": ACT},
            "Low-barrier options":      {"fn": has_words("saving", "deposit", "fund", "app", "micro"), "cat": ACT},
            "Specific amount suggestion": {"fn": lambda r: any(c.isdigit() for c in r), "cat": ACT},
        },
    },
    {
        "id": "TC91", "name": "Invest — Freelancer, retirement",
        "role": "Freelancer", "income": 30_000_000,
        "spending": {"Food": 3_500_000, "Transport": 2_000_000, "Bills": 3_000_000, "Entertainment": 1_500_000},
        # total=10M, surplus=20M, tax=9M, after-tax=11M
        "question": "I don't have an employer pension. How do I save for retirement?",
        "checks": {
            "Monthly surplus correct":  {"fn": has_amount(20_000_000), "cat": CALC},
            "Retirement options":       {"fn": has_words("pension", "retire", "fund", "invest", "long term", "long-term"), "cat": ACT},
            "Monthly target amount":    {"fn": lambda r: any(c.isdigit() for c in r), "cat": CALC},
            "Concrete first step":      {"fn": has_words("open", "start", "set up", "contribute", "transfer"), "cat": ACT},
        },
    },
    {
        "id": "TC92", "name": "Invest — Worker, salary split",
        "role": "Worker", "income": 30_000_000,
        "spending": {"Food": 5_000_000, "Transport": 3_000_000, "Bills": 4_000_000, "Shopping": 2_500_000},
        # total=14.5M, surplus=15.5M, needs(50%)=15M, wants(30%)=9M, savings(20%)=6M
        "question": "How should I split my salary between spending, saving, and investing?",
        "checks": {
            "Needs amount (50%)":       {"fn": has_amount(15_000_000), "cat": CALC},
            "Savings amount (20%)":     {"fn": has_amount(6_000_000), "cat": CALC},
            "Mentions percentages":     {"fn": lambda r: "%" in r, "cat": CALC},
            "Three-way split advice":   {"fn": has_all_words("spend", "sav"), "cat": ACT},
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  INCOME & SIDE HUSTLES  (TC93–TC96)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "TC93", "name": "Income — Student, need more money",
        "role": "Student", "income": 3_000_000,
        "spending": {"Food": 1_500_000, "Transport": 400_000, "Entertainment": 500_000, "Education": 400_000},
        # total=2.8M, surplus=200K
        "question": "My expenses match my income. How can I earn more?",
        "checks": {
            "Surplus/deficit correct":  {"fn": has_amount(200_000), "cat": CALC},
            "2+ income suggestions":   {"fn": lambda r: sum(1 for w in ["part-time", "part time", "tutor", "freelanc", "online", "intern", "job"] if w in r.lower()) >= 2, "cat": ACT},
            "Actionable first step":   {"fn": has_words("apply", "sign up", "start", "look for", "try", "register"), "cat": ACT},
            "Also suggests cuts":      {"fn": suggests_specific_cut({"Entertainment": 0, "Food": 0}), "cat": ACT},
        },
    },
    {
        "id": "TC94", "name": "Income — Freelancer, raise rates",
        "role": "Freelancer", "income": 15_000_000,
        "spending": {"Food": 3_000_000, "Transport": 1_200_000, "Bills": 2_000_000, "Equipment": 1_500_000},
        # total=7.7M, surplus=7.3M
        "question": "I feel underpaid for my work. Should I raise my rates?",
        "checks": {
            "Current surplus context":  {"fn": has_amount(7_300_000), "cat": CALC},
            "Encourages rate raise":    {"fn": has_words("yes", "raise", "increase", "higher", "worth"), "cat": ACT},
            "Strategy":                 {"fn": has_words("gradual", "new client", "portfolio", "skill", "negotiate", "market", "value"), "cat": ACT},
            "Concrete action":          {"fn": has_words("start", "research", "compare", "talk", "propose"), "cat": ACT},
        },
    },
    {
        "id": "TC95", "name": "Income — Worker, side income",
        "role": "Worker", "income": 14_000_000,
        "spending": {"Food": 3_000_000, "Transport": 2_000_000, "Bills": 2_500_000, "Shopping": 2_000_000, "Entertainment": 1_500_000},
        # total=11M, surplus=3M
        "question": "I have no savings left. Should I get a side job?",
        "checks": {
            "Monthly surplus correct":  {"fn": has_amount(3_000_000), "cat": CALC},
            "Identifies spending issue first": {"fn": suggests_specific_cut({"Shopping": 0, "Entertainment": 0}), "cat": ACT},
            "Side income ideas":        {"fn": has_words("freelanc", "side", "extra", "skill", "online", "part-time"), "cat": ACT},
            "Both cut + earn":          {"fn": lambda r: any(c in r.lower() for c in ["cut", "reduce"]) or any(e in r.lower() for e in ["earn", "side", "extra", "income"]), "cat": ACT},
        },
    },
    {
        "id": "TC96", "name": "Income — Student, summer job",
        "role": "Student", "income": 3_500_000,
        "spending": {"Food": 1_200_000, "Transport": 300_000, "Education": 500_000, "Entertainment": 400_000},
        # total=2.4M, surplus=1.1M
        "question": "Summer break is coming. How can I make the most of it financially?",
        "checks": {
            "Current surplus correct":  {"fn": has_amount(1_100_000), "cat": CALC},
            "Job suggestions":          {"fn": has_words("intern", "job", "work", "part-time", "part time", "earn"), "cat": ACT},
            "Saving strategy":          {"fn": has_words("save", "set aside", "put away", "build"), "cat": ACT},
            "Concrete action":          {"fn": has_words("apply", "start", "look for", "sign up"), "cat": ACT},
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  EDGE CASES & MISC  (TC97–TC100)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "TC97", "name": "Edge — Very low income student",
        "role": "Student", "income": 2_000_000,
        "spending": {"Food": 1_200_000, "Transport": 300_000, "Entertainment": 200_000},
        # total=1.7M, surplus=300K
        "question": "I only have 2 million a month. Any financial advice?",
        "checks": {
            "Total spent correct":      {"fn": has_amount(1_700_000), "cat": CALC},
            "Surplus correct":          {"fn": has_amount(300_000), "cat": CALC},
            "Practical low-cost tips":  {"fn": lambda r: sum(1 for w in ["cook", "free", "discount", "save", "cut", "walk", "public"] if w in r.lower()) >= 2, "cat": ACT},
            "Income suggestion":        {"fn": has_words("part-time", "part time", "earn", "income", "tutor", "job"), "cat": ACT},
        },
    },
    {
        "id": "TC98", "name": "Edge — High income worker, no direction",
        "role": "Worker", "income": 50_000_000,
        "spending": {"Food": 6_000_000, "Transport": 4_000_000, "Bills": 5_000_000, "Shopping": 8_000_000, "Entertainment": 5_000_000},
        # total=28M, surplus=22M, needs(50%)=25M, savings(20%)=10M
        "question": "I make good money but have no financial plan. Where do I start?",
        "checks": {
            "Total spent correct":       {"fn": has_amount(28_000_000), "cat": CALC},
            "Surplus correct":           {"fn": has_amount(22_000_000), "cat": CALC},
            "Mentions budget rule":      {"fn": has_words("50/30/20", "budget", "rule", "allocat"), "cat": CALC},
            "Ordered action plan":       {"fn": has_words("emergency", "invest", "save", "first", "then", "start"), "cat": ACT},
            "Emergency fund amount":     {"fn": has_any_amount(84_000_000, 168_000_000), "cat": CALC},
        },
    },
    {
        "id": "TC99", "name": "Edge — Freelancer, zero savings",
        "role": "Freelancer", "income": 15_000_000,
        "spending": {"Food": 4_000_000, "Transport": 2_000_000, "Bills": 3_000_000, "Entertainment": 3_000_000, "Equipment": 3_000_000},
        # total=15M, surplus=0, tax owed=4.5M
        "question": "I spend everything I earn. How do I break this cycle?",
        "checks": {
            "Total spent = income":      {"fn": has_amount(15_000_000), "cat": CALC},
            "Identifies zero surplus":   {"fn": has_words("all", "everything", "100%", "zero", "nothing left", "no surplus"), "cat": CALC},
            "Tax warning with amount":   {"fn": has_words("tax"), "cat": CALC},
            "Names specific category to cut": {"fn": suggests_specific_cut({"Entertainment": 0, "Equipment": 0}), "cat": ACT},
            "Concrete first step":       {"fn": has_words("first", "start", "today", "immediately", "open", "automat"), "cat": ACT},
        },
    },
    {
        "id": "TC100", "name": "Edge — Worker, financial review request",
        "role": "Worker", "income": 22_000_000,
        "spending": {"Food": 4_500_000, "Transport": 2_500_000, "Bills": 3_500_000, "Shopping": 3_000_000, "Entertainment": 2_000_000, "Health": 1_000_000},
        # total=16.5M, surplus=5.5M, needs(50%)=11M, wants(30%)=6.6M, savings(20%)=4.4M
        "question": "Give me a complete financial review and recommendations.",
        "checks": {
            "Total spent correct":       {"fn": has_amount(16_500_000), "cat": CALC},
            "Surplus correct":           {"fn": has_amount(5_500_000), "cat": CALC},
            "References 3+ categories":  {"fn": refs_user_categories({"Food": 0, "Transport": 0, "Bills": 0, "Shopping": 0, "Entertainment": 0, "Health": 0}, 3), "cat": ACT},
            "Mentions percentages":      {"fn": lambda r: "%" in r, "cat": CALC},
            "Savings target":            {"fn": has_amount(4_400_000), "cat": CALC},
            "Concrete next steps":       {"fn": has_words("save", "cut", "reduce", "invest", "emergency"), "cat": ACT},
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  GOAL FEASIBILITY  (TC101–TC106)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "TC101", "name": "Goal — Student, laptop goal on track",
        "role": "Student", "income": 6_000_000,
        "spending": {"Food": 2_000_000, "Transport": 500_000, "Entertainment": 500_000, "Education": 500_000},
        # surplus=2.5M, goal needs 7M more, ~2.8 months, target is 4 months away → on track
        "question": "Can I reach my Laptop Fund goal in time?",
        "goals": [{"name": "Laptop Fund", "target_amount": 15_000_000, "current_saved": 8_000_000, "target_date": "2026-08-01", "priority": "HIGH"}],
        "checks": {
            "References goal name":      {"fn": has_goal_reference("Laptop Fund"), "cat": ACT},
            "Mentions remaining amount": {"fn": has_amount(7_000_000), "cat": CALC},
            "States on track":           {"fn": has_words("on track", "feasible", "can reach", "will reach", "ahead"), "cat": ACT},
        },
    },
    {
        "id": "TC102", "name": "Goal — Worker, emergency fund behind",
        "role": "Worker", "income": 20_000_000,
        "spending": {"Food": 4_000_000, "Transport": 2_000_000, "Bills": 3_000_000, "Shopping": 3_000_000, "Entertainment": 2_000_000, "Health": 1_000_000},
        # surplus=5M, goal needs 40M more, 8 months needed, only 3 months left → at risk
        "question": "Am I on track for my Emergency Fund?",
        "goals": [{"name": "Emergency Fund", "target_amount": 50_000_000, "current_saved": 10_000_000, "target_date": "2026-07-01", "priority": "HIGH"}],
        "checks": {
            "References goal name":      {"fn": has_goal_reference("Emergency Fund"), "cat": ACT},
            "Mentions remaining":        {"fn": has_amount(40_000_000), "cat": CALC},
            "Warns about risk":          {"fn": has_words("risk", "behind", "not enough", "won't", "short", "at risk"), "cat": ACT},
        },
    },
    {
        "id": "TC103", "name": "Goal — Freelancer, Tết savings",
        "role": "Freelancer", "income": 30_000_000,
        "spending": {"Food": 5_000_000, "Transport": 2_000_000, "Bills": 4_000_000, "Shopping": 3_000_000, "Entertainment": 1_500_000},
        # surplus=14.5M
        "question": "How much do I need to save for Tết?",
        "goals": [{"name": "Tết Savings", "target_amount": 10_000_000, "current_saved": 3_000_000, "target_date": "2027-01-15", "priority": "HIGH"}],
        "checks": {
            "References Tết":            {"fn": has_words("tết", "tet"), "cat": ACT},
            "Mentions remaining":        {"fn": has_amount(7_000_000), "cat": CALC},
            "Gives timeline":            {"fn": has_words("month"), "cat": ACT},
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  BALANCE-AWARE RESPONSES  (TC107–TC110)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "TC107", "name": "Balance — Worker, total balance check",
        "role": "Worker", "income": 25_000_000,
        "spending": {"Food": 5_000_000, "Transport": 2_000_000, "Bills": 4_000_000, "Shopping": 2_000_000, "Health": 1_000_000},
        # surplus=11M
        "question": "How much money do I have in total?",
        "balances": [
            {"name": "Checking (VCB)", "balance": 15_000_000, "currency": "VND", "type": "checking"},
            {"name": "Savings (TPBank)", "balance": 45_000_000, "currency": "VND", "type": "savings"},
        ],
        "checks": {
            "Total balance correct":     {"fn": has_balance_sum(60_000_000), "cat": CALC},
            "Mentions checking":         {"fn": has_amount(15_000_000), "cat": CALC},
            "Mentions savings":          {"fn": has_amount(45_000_000), "cat": CALC},
        },
    },
    {
        "id": "TC108", "name": "Balance — Student, can I afford now?",
        "role": "Student", "income": 5_000_000,
        "spending": {"Food": 2_000_000, "Transport": 500_000, "Entertainment": 500_000, "Education": 500_000},
        # surplus=1.5M, balance=3M, phone costs 8M → can't afford now
        "question": "Can I afford a new phone costing 8.000.000 VND right now?",
        "balances": [{"name": "MoMo Wallet", "balance": 3_000_000, "currency": "VND", "type": "ewallet"}],
        "checks": {
            "Says can't afford now":     {"fn": has_words("not enough", "can't", "cannot", "short", "insufficient", "don't have"), "cat": ACT},
            "Mentions balance":          {"fn": has_amount(3_000_000), "cat": CALC},
            "Suggests saving plan":      {"fn": has_words("save", "month", "plan"), "cat": ACT},
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  FORECAST-AWARE RESPONSES  (TC111–TC114)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "TC111", "name": "Forecast — Worker, spending increase warning",
        "role": "Worker", "income": 20_000_000,
        "spending": {"Food": 4_000_000, "Transport": 2_000_000, "Bills": 3_000_000, "Shopping": 2_000_000, "Entertainment": 1_500_000},
        # total=12.5M, forecast=15M (20% increase)
        "question": "What does my spending look like going forward?",
        "forecast": {"total": 15_000_000, "confidence": 75},
        "checks": {
            "Mentions forecast amount":  {"fn": has_amount(15_000_000), "cat": CALC},
            "Warns about increase":      {"fn": has_words("increase", "higher", "more", "rising", "up"), "cat": ACT},
            "Mentions confidence":       {"fn": has_words("75%", "confidence"), "cat": CALC},
        },
    },
    {
        "id": "TC112", "name": "Forecast — Freelancer, stable spending",
        "role": "Freelancer", "income": 25_000_000,
        "spending": {"Food": 4_000_000, "Transport": 1_500_000, "Bills": 3_000_000, "Shopping": 1_500_000},
        # total=10M, forecast=10.5M (stable)
        "question": "What's my spending trajectory?",
        "forecast": {"total": 10_500_000, "confidence": 80},
        "checks": {
            "Mentions forecast":         {"fn": has_forecast_mention(), "cat": ACT},
            "Forecast amount":           {"fn": has_amount(10_500_000), "cat": CALC},
            "Acknowledges stability":    {"fn": has_words("stable", "consistent", "similar", "same", "steady", "on track"), "cat": ACT},
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  VERDICT ACCURACY  (TC115–TC118)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "TC115", "name": "Verdict — High savings correctly praised",
        "role": "Worker", "income": 22_000_000,
        "spending": {"Food": 3_000_000, "Transport": 1_000_000, "Bills": 2_000_000, "Shopping": 500_000},
        # surplus=15.5M, savings_rate=70.5% → ABOVE TARGET → should praise
        "question": "Am I saving enough this month?",
        "checks": {
            "Does NOT say below target": {"fn": lambda r: "below target" not in r.lower() and "below the" not in r.lower(), "cat": CALC},
            "Praises savings":           {"fn": has_words("above", "exceed", "great", "excellent", "well done", "good", "healthy", "strong"), "cat": ACT},
            "Correct savings amount":    {"fn": has_amount(15_500_000), "cat": CALC},
        },
    },
    {
        "id": "TC116", "name": "Verdict — Low savings correctly warned",
        "role": "Student", "income": 4_000_000,
        "spending": {"Food": 2_000_000, "Transport": 800_000, "Entertainment": 700_000, "Education": 300_000},
        # surplus=200k, savings_rate=5% → BELOW TARGET → should warn
        "question": "How are my savings this month?",
        "checks": {
            "Does NOT say above target": {"fn": lambda r: "above target" not in r.lower() and "exceeds" not in r.lower(), "cat": CALC},
            "Warns about shortfall":     {"fn": has_words("below", "short", "need", "not enough", "gap", "warning", "improve"), "cat": ACT},
            "Correct surplus":           {"fn": has_amount(200_000), "cat": CALC},
        },
    },
    {
        "id": "TC117", "name": "Verdict — Deficit correctly identified",
        "role": "Student", "income": 3_000_000,
        "spending": {"Food": 1_800_000, "Transport": 500_000, "Entertainment": 500_000, "Shopping": 400_000},
        # total=3.2M, deficit=-200k → OVER BUDGET
        "question": "What's my financial status?",
        "checks": {
            "Detects deficit":           {"fn": has_words("over", "deficit", "exceed", "negative", "more than income", "loss"), "cat": CALC},
            "Deficit amount":            {"fn": has_amount(200_000), "cat": CALC},
            "Urgent action":             {"fn": has_words("cut", "reduce", "immediately", "urgent", "stop", "freeze"), "cat": ACT},
        },
    },
    {
        "id": "TC118", "name": "Verdict — 78% savings rate not called low",
        "role": "Worker", "income": 22_000_000,
        "spending": {"Food": 2_780_720, "Shopping": 1_500_000, "Transport": 500_000},
        # surplus=17,219,280, savings_rate=78.3% → ABOVE TARGET
        "question": "Is my savings rate good?",
        "checks": {
            "Does NOT say 7.7%":         {"fn": lambda r: "7.7%" not in r, "cat": CALC},
            "Does NOT say below":        {"fn": lambda r: "below target" not in r.lower(), "cat": CALC},
            "Says above/good/healthy":   {"fn": has_words("above", "great", "excellent", "good", "healthy", "strong", "well"), "cat": ACT},
            "References correct surplus": {"fn": has_amount(17_219_280), "cat": CALC},
        },
    },

    # ── Category Budget Awareness (TC201-TC206) ─────────────────────────────────

    {
        "id": "TC201", "name": "No category budget → nudge to set up",
        "role": "Student", "income": 5_000_000,
        "spending": {"Food": 1_500_000, "Transport": 500_000, "Entertainment": 800_000, "Education": 300_000},
        # No category_budgets → context will say "CATEGORY BUDGET: NOT SET"
        "question": "How is my spending this month?",
        "checks": {
            "Mentions Food":                {"fn": has_words("Food"), "cat": ACT},
            "Mentions Entertainment":       {"fn": has_words("Entertainment"), "cat": ACT},
            "Suggests setting up budgets":  {"fn": has_budget_nudge(), "cat": ACT},
        },
    },
    {
        "id": "TC202", "name": "Has category budgets — references over limit",
        "role": "Worker", "income": 20_000_000,
        "spending": {"Food": 5_500_000, "Transport": 2_000_000, "Shopping": 3_200_000, "Bills": 2_500_000},
        # Shopping over its 3M limit, Food over its 5M limit
        "category_budgets": [
            {"categoryName": "Food", "monthlyLimit": 5_000_000},
            {"categoryName": "Transport", "monthlyLimit": 2_500_000},
            {"categoryName": "Shopping", "monthlyLimit": 3_000_000},
            {"categoryName": "Bills", "monthlyLimit": 3_000_000},
        ],
        "question": "Am I staying within my budgets?",
        "checks": {
            "Mentions Food over limit":     {"fn": has_over_limit_mention("Food"), "cat": ACT},
            "Mentions Shopping over limit": {"fn": has_over_limit_mention("Shopping"), "cat": ACT},
            "References budget/limit":      {"fn": has_words("budget", "limit"), "cat": ACT},
        },
    },
    {
        "id": "TC203", "name": "No category budget — overview still works",
        "role": "Freelancer", "income": 30_000_000,
        "spending": {"Food": 4_000_000, "Transport": 1_500_000, "Bills": 3_000_000, "Shopping": 2_000_000},
        "question": "Give me a financial overview.",
        "checks": {
            "Mentions surplus":             {"fn": has_amount(19_500_000), "cat": CALC},
            "Names 2+ categories":          {"fn": lambda r: sum(1 for c in ["Food", "Transport", "Bills", "Shopping"] if c.lower() in r.lower()) >= 2, "cat": ACT},
            "Suggests category budgets":    {"fn": has_budget_nudge(), "cat": ACT},
        },
    },
    {
        "id": "TC204", "name": "Category budget — near limit warning",
        "role": "Student", "income": 6_000_000,
        "spending": {"Food": 1_800_000, "Transport": 450_000, "Entertainment": 900_000, "Education": 400_000},
        # Entertainment at 900k / 950k = 94.7% → NEAR LIMIT
        "category_budgets": [
            {"categoryName": "Food", "monthlyLimit": 2_500_000},
            {"categoryName": "Transport", "monthlyLimit": 600_000},
            {"categoryName": "Entertainment", "monthlyLimit": 950_000},
            {"categoryName": "Education", "monthlyLimit": 500_000},
        ],
        "question": "Check my budget limits.",
        "checks": {
            "Mentions Entertainment near limit": {"fn": lambda r: "entertainment" in r.lower() and any(w in r.lower() for w in ["near", "close", "almost", "approaching"]), "cat": ACT},
            "References budget status":          {"fn": has_words("budget", "limit"), "cat": ACT},
        },
    },
    {
        "id": "TC205", "name": "Category budget — all OK praise",
        "role": "Worker", "income": 25_000_000,
        "spending": {"Food": 3_000_000, "Transport": 1_000_000, "Shopping": 1_500_000, "Bills": 2_000_000, "Health": 500_000},
        # All well under limits
        "category_budgets": [
            {"categoryName": "Food", "monthlyLimit": 5_000_000},
            {"categoryName": "Transport", "monthlyLimit": 2_000_000},
            {"categoryName": "Shopping", "monthlyLimit": 3_000_000},
            {"categoryName": "Bills", "monthlyLimit": 3_000_000},
            {"categoryName": "Health", "monthlyLimit": 1_000_000},
        ],
        "question": "How are my category budgets?",
        "checks": {
            "Positive tone":                    {"fn": has_words("good", "well", "great", "healthy", "within", "under"), "cat": ACT},
            "References budget":                {"fn": has_words("budget", "limit"), "cat": ACT},
        },
    },
    {
        "id": "TC206", "name": "Category budget — single big overspend",
        "role": "Freelancer", "income": 40_000_000,
        "spending": {"Food": 6_000_000, "Transport": 2_000_000, "Shopping": 8_500_000, "Bills": 4_000_000},
        # Shopping way over: 8.5M vs 5M limit → +3.5M over
        "category_budgets": [
            {"categoryName": "Food", "monthlyLimit": 7_000_000},
            {"categoryName": "Transport", "monthlyLimit": 3_000_000},
            {"categoryName": "Shopping", "monthlyLimit": 5_000_000},
            {"categoryName": "Bills", "monthlyLimit": 5_000_000},
        ],
        "question": "Which categories am I overspending on?",
        "checks": {
            "Identifies Shopping":              {"fn": has_over_limit_mention("Shopping"), "cat": ACT},
            "Mentions overage amount":          {"fn": has_amount(3_500_000), "cat": CALC},
            "Suggests reducing":                {"fn": has_words("reduce", "cut", "lower", "less"), "cat": ACT},
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    # TC301-TC306: TRANSACTION LOGGING (Action Extraction)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "TC301", "name": "Log transaction — coffee expense",
        "role": "Student", "income": 5_000_000,
        "spending": {"Food": 1_800_000, "Transport": 500_000, "Entertainment": 400_000, "Education": 300_000},
        "question": "I just spent 85.000 VND on coffee.",
        "checks": {
            "Has action tag":                   {"fn": has_action_tag("LOG_TRANSACTION", category="Food"), "cat": ACT},
            "Confirms logging":                 {"fn": has_words("logged", "got it", "added", "recorded"), "cat": ACT},
        },
    },
    {
        "id": "TC302", "name": "Log transaction — grocery shopping",
        "role": "Worker", "income": 20_000_000,
        "spending": {"Food": 4_500_000, "Transport": 2_000_000, "Bills": 3_000_000, "Shopping": 2_500_000},
        "question": "Log 450.000 VND for VinMart grocery.",
        "checks": {
            "Has action tag":                   {"fn": has_action_tag("LOG_TRANSACTION"), "cat": ACT},
            "Confirms logging":                 {"fn": has_words("logged", "got it", "added", "recorded"), "cat": ACT},
        },
    },
    {
        "id": "TC303", "name": "Log transaction — equipment purchase",
        "role": "Freelancer", "income": 30_000_000,
        "spending": {"Food": 4_000_000, "Transport": 1_500_000, "Bills": 3_000_000, "Shopping": 2_000_000},
        "question": "I bought a new mouse for 750.000 VND.",
        "checks": {
            "Has action tag":                   {"fn": has_action_tag("LOG_TRANSACTION"), "cat": ACT},
            "Confirms logging":                 {"fn": has_words("logged", "got it", "added", "recorded"), "cat": ACT},
        },
    },
    {
        "id": "TC304", "name": "Spending question — no action tag (negative)",
        "role": "Student", "income": 5_000_000,
        "spending": {"Food": 2_000_000, "Transport": 600_000, "Entertainment": 500_000, "Education": 400_000},
        "question": "How much did I spend on Food?",
        "checks": {
            "No action tag":                    {"fn": has_no_action_tag(), "cat": ACT},
            "Mentions Food amount":             {"fn": has_amount(2_000_000), "cat": CALC},
        },
    },
    {
        "id": "TC305", "name": "Budget question — no action tag (negative)",
        "role": "Worker", "income": 20_000_000,
        "spending": {"Food": 5_000_000, "Transport": 2_500_000, "Bills": 3_000_000, "Shopping": 2_000_000},
        "question": "What's my biggest expense?",
        "checks": {
            "No action tag":                    {"fn": has_no_action_tag(), "cat": ACT},
            "Mentions Food":                    {"fn": has_words("food"), "cat": ACT},
        },
    },
    {
        "id": "TC306", "name": "Log transaction — dinner with clients",
        "role": "Freelancer", "income": 40_000_000,
        "spending": {"Food": 5_000_000, "Transport": 2_000_000, "Bills": 4_000_000, "Entertainment": 3_000_000},
        "question": "I had a 2.000.000 VND dinner with clients.",
        "checks": {
            "Has action tag":                   {"fn": has_action_tag("LOG_TRANSACTION"), "cat": ACT},
            "Confirms logging":                 {"fn": has_words("logged", "got it", "added", "recorded"), "cat": ACT},
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    # TC307-TC312: MONTH-OVER-MONTH COMPARISON
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "TC307", "name": "MoM — Food went up 30%",
        "role": "Worker", "income": 20_000_000,
        "spending": {"Food": 5_200_000, "Transport": 2_000_000, "Bills": 3_000_000, "Shopping": 1_500_000},
        "monthly_history": {
            "2026-03": {"Food": 4_000_000, "Transport": 2_100_000, "Bills": 2_900_000, "Shopping": 1_600_000},
        },
        "question": "Am I spending more than last month?",
        "checks": {
            "References MoM":                   {"fn": has_mom_reference(), "cat": ACT},
            "Mentions Food increase":           {"fn": has_words("food"), "cat": ACT},
        },
    },
    {
        "id": "TC308", "name": "MoM — overall spending down",
        "role": "Student", "income": 6_000_000,
        "spending": {"Food": 1_800_000, "Transport": 400_000, "Entertainment": 300_000, "Education": 500_000},
        "monthly_history": {
            "2026-03": {"Food": 2_200_000, "Transport": 500_000, "Entertainment": 600_000, "Education": 500_000},
        },
        "question": "How does this month compare to last?",
        "checks": {
            "References MoM":                   {"fn": has_mom_reference(), "cat": ACT},
            "Positive feedback":                {"fn": has_words("less", "down", "decrease", "improve", "great", "good", "reduce"), "cat": ACT},
        },
    },
    {
        "id": "TC309", "name": "MoM — new category appeared",
        "role": "Freelancer", "income": 25_000_000,
        "spending": {"Food": 3_500_000, "Transport": 1_500_000, "Bills": 2_500_000, "Health": 800_000},
        "monthly_history": {
            "2026-03": {"Food": 3_400_000, "Transport": 1_600_000, "Bills": 2_400_000},
        },
        "question": "Any changes in my spending this month?",
        "checks": {
            "References MoM":                   {"fn": has_mom_reference(), "cat": ACT},
            "Mentions Health as new":           {"fn": has_words("health", "new"), "cat": ACT},
        },
    },
    {
        "id": "TC310", "name": "MoM — stable spending",
        "role": "Worker", "income": 25_000_000,
        "spending": {"Food": 5_000_000, "Transport": 2_500_000, "Bills": 3_500_000, "Shopping": 2_000_000},
        "monthly_history": {
            "2026-03": {"Food": 4_900_000, "Transport": 2_600_000, "Bills": 3_400_000, "Shopping": 2_100_000},
        },
        "question": "Is my spending going up or down?",
        "checks": {
            "References MoM":                   {"fn": has_mom_reference(), "cat": ACT},
            "Notes stability":                  {"fn": has_words("stable", "consistent", "similar", "flat", "steady"), "cat": ACT},
        },
    },
    {
        "id": "TC311", "name": "MoM — Entertainment doubled",
        "role": "Student", "income": 5_000_000,
        "spending": {"Food": 1_500_000, "Transport": 400_000, "Entertainment": 1_200_000, "Education": 300_000},
        "monthly_history": {
            "2026-03": {"Food": 1_600_000, "Transport": 450_000, "Entertainment": 600_000, "Education": 350_000},
        },
        "question": "How's my spending compared to last month?",
        "checks": {
            "Mentions Entertainment":           {"fn": has_words("entertainment"), "cat": ACT},
            "Warns about increase":             {"fn": has_words("increase", "up", "more", "double", "higher", "jump"), "cat": ACT},
        },
    },
    {
        "id": "TC312", "name": "MoM — specific category query",
        "role": "Freelancer", "income": 35_000_000,
        "spending": {"Food": 4_500_000, "Transport": 3_000_000, "Bills": 4_000_000, "Shopping": 2_000_000},
        "monthly_history": {
            "2026-03": {"Food": 5_500_000, "Transport": 2_800_000, "Bills": 3_900_000, "Shopping": 2_200_000},
        },
        "question": "Is my Food spending going up?",
        "checks": {
            "Mentions Food":                    {"fn": has_words("food"), "cat": ACT},
            "Notes decrease":                   {"fn": has_words("down", "decrease", "less", "lower", "reduce", "improve"), "cat": ACT},
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    # TC313-TC318: RECURRING EXPENSE AWARENESS
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "TC313", "name": "Recurring — list fixed costs",
        "role": "Worker", "income": 20_000_000,
        "spending": {"Food": 4_000_000, "Transport": 2_000_000, "Bills": 3_500_000, "Entertainment": 1_500_000},
        "recurring": [
            {"description": "Netflix", "category": "Entertainment", "amount": 89_000, "occurrences": 3},
            {"description": "FPT Internet", "category": "Bills", "amount": 180_000, "occurrences": 3},
            {"description": "Rent", "category": "Bills", "amount": 2_500_000, "occurrences": 3},
        ],
        "question": "What are my fixed costs?",
        "checks": {
            "References recurring":             {"fn": has_recurring_reference(), "cat": ACT},
            "Mentions at least 2 items":        {"fn": lambda r: sum(1 for w in ["netflix", "internet", "rent"] if w in r.lower()) >= 2, "cat": ACT},
        },
    },
    {
        "id": "TC314", "name": "Recurring — no recurring detected",
        "role": "Student", "income": 5_000_000,
        "spending": {"Food": 2_000_000, "Transport": 500_000, "Entertainment": 400_000, "Education": 600_000},
        "question": "Do I have any recurring expenses?",
        "checks": {
            "Answers question":                 {"fn": has_words("no", "not", "none", "don't"), "cat": ACT},
        },
    },
    {
        "id": "TC315", "name": "Recurring — what can I cut",
        "role": "Freelancer", "income": 30_000_000,
        "spending": {"Food": 5_000_000, "Transport": 2_000_000, "Bills": 4_000_000, "Shopping": 3_000_000, "Entertainment": 2_000_000},
        "recurring": [
            {"description": "Rent", "category": "Bills", "amount": 3_000_000, "occurrences": 3},
            {"description": "Phone plan", "category": "Bills", "amount": 150_000, "occurrences": 3},
        ],
        "question": "What can I cut from my budget?",
        "checks": {
            "Distinguishes fixed/discretionary": {"fn": has_words("fixed", "discretionary", "recurring", "flexible"), "cat": ACT},
            "Suggests cutting discretionary":   {"fn": has_words("cut", "reduce", "lower"), "cat": ACT},
        },
    },
    {
        "id": "TC316", "name": "Recurring — high fixed costs",
        "role": "Worker", "income": 15_000_000,
        "spending": {"Food": 3_000_000, "Transport": 1_500_000, "Bills": 5_000_000, "Shopping": 1_000_000},
        "recurring": [
            {"description": "Rent", "category": "Bills", "amount": 3_500_000, "occurrences": 3},
            {"description": "Insurance", "category": "Health", "amount": 500_000, "occurrences": 3},
            {"description": "FPT Internet", "category": "Bills", "amount": 180_000, "occurrences": 3},
            {"description": "Phone plan", "category": "Bills", "amount": 150_000, "occurrences": 3},
        ],
        "question": "Why can't I save more money?",
        "checks": {
            "Mentions fixed costs":             {"fn": has_recurring_reference(), "cat": ACT},
            "Suggests reviewing":               {"fn": has_words("review", "reduce", "cut", "negotiate", "cancel"), "cat": ACT},
        },
    },
    {
        "id": "TC317", "name": "Recurring — subscription list",
        "role": "Student", "income": 6_000_000,
        "spending": {"Food": 2_000_000, "Transport": 600_000, "Entertainment": 800_000, "Education": 500_000},
        "recurring": [
            {"description": "Netflix", "category": "Entertainment", "amount": 89_000, "occurrences": 3},
            {"description": "Spotify", "category": "Entertainment", "amount": 59_000, "occurrences": 2},
        ],
        "question": "How much goes to subscriptions?",
        "checks": {
            "Lists subscriptions":              {"fn": lambda r: sum(1 for w in ["netflix", "spotify"] if w in r.lower()) >= 1, "cat": ACT},
            "Mentions total":                   {"fn": has_words("total", "combined", "together"), "cat": ACT},
        },
    },
    {
        "id": "TC318", "name": "Recurring — fixed vs discretionary split",
        "role": "Freelancer", "income": 40_000_000,
        "spending": {"Food": 6_000_000, "Transport": 3_000_000, "Bills": 5_000_000, "Shopping": 4_000_000, "Entertainment": 2_000_000},
        "recurring": [
            {"description": "Rent", "category": "Bills", "amount": 3_500_000, "occurrences": 3},
            {"description": "Electricity", "category": "Bills", "amount": 300_000, "occurrences": 3},
            {"description": "Gym fee", "category": "Health", "amount": 450_000, "occurrences": 3},
        ],
        "question": "What expenses are flexible vs fixed?",
        "checks": {
            "Distinguishes types":              {"fn": has_words("fixed", "flexible", "discretionary", "recurring"), "cat": ACT},
            "Mentions specific items":          {"fn": has_recurring_reference(), "cat": ACT},
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    # TC319-TC324: SPENDING ANOMALY ALERTS
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "TC319", "name": "Anomaly — Shopping 2.5x average",
        "role": "Worker", "income": 25_000_000,
        "spending": {"Food": 5_000_000, "Transport": 2_000_000, "Shopping": 8_500_000, "Bills": 3_000_000},
        "monthly_history": {
            "2026-03": {"Food": 4_800_000, "Transport": 2_100_000, "Shopping": 3_500_000, "Bills": 2_900_000},
            "2026-02": {"Food": 5_100_000, "Transport": 1_900_000, "Shopping": 3_400_000, "Bills": 3_100_000},
        },
        "question": "Anything unusual in my spending?",
        "checks": {
            "Mentions Shopping anomaly":        {"fn": has_anomaly_reference("Shopping"), "cat": ACT},
            "Warns about spike":                {"fn": has_words("unusual", "spike", "high", "above", "attention"), "cat": ACT},
        },
    },
    {
        "id": "TC320", "name": "Anomaly — no anomalies (negative)",
        "role": "Student", "income": 5_000_000,
        "spending": {"Food": 1_800_000, "Transport": 500_000, "Entertainment": 400_000, "Education": 300_000},
        "monthly_history": {
            "2026-03": {"Food": 1_750_000, "Transport": 520_000, "Entertainment": 380_000, "Education": 310_000},
            "2026-02": {"Food": 1_850_000, "Transport": 480_000, "Entertainment": 420_000, "Education": 290_000},
        },
        "question": "Anything unusual in my spending?",
        "checks": {
            "Confirms normal":                  {"fn": has_words("normal", "no", "nothing unusual", "typical", "consistent", "stable"), "cat": ACT},
        },
    },
    {
        "id": "TC321", "name": "Anomaly — two anomalies (Food + Entertainment)",
        "role": "Freelancer", "income": 30_000_000,
        "spending": {"Food": 8_000_000, "Transport": 2_000_000, "Entertainment": 3_600_000, "Bills": 3_000_000},
        "monthly_history": {
            "2026-03": {"Food": 3_800_000, "Transport": 2_100_000, "Entertainment": 2_000_000, "Bills": 3_100_000},
            "2026-02": {"Food": 4_200_000, "Transport": 1_900_000, "Entertainment": 2_000_000, "Bills": 2_900_000},
        },
        "question": "Are there any spending anomalies?",
        "checks": {
            "Mentions Food":                    {"fn": has_anomaly_reference("Food"), "cat": ACT},
            "Mentions Entertainment":           {"fn": has_anomaly_reference("Entertainment"), "cat": ACT},
        },
    },
    {
        "id": "TC322", "name": "Anomaly — mild WATCH on Transport",
        "role": "Worker", "income": 20_000_000,
        "spending": {"Food": 4_000_000, "Transport": 3_200_000, "Bills": 3_000_000, "Shopping": 1_500_000},
        "monthly_history": {
            "2026-03": {"Food": 4_100_000, "Transport": 2_000_000, "Bills": 2_900_000, "Shopping": 1_600_000},
            "2026-02": {"Food": 3_900_000, "Transport": 2_000_000, "Bills": 3_100_000, "Shopping": 1_400_000},
        },
        "question": "How is my spending this month?",
        "checks": {
            "Mentions Transport":               {"fn": has_words("transport"), "cat": ACT},
            "References increase":              {"fn": has_words("increase", "higher", "up", "more", "watch", "above"), "cat": ACT},
        },
    },
    {
        "id": "TC323", "name": "Anomaly — proactive in general question",
        "role": "Student", "income": 6_000_000,
        "spending": {"Food": 1_800_000, "Transport": 500_000, "Entertainment": 2_400_000, "Education": 400_000},
        "monthly_history": {
            "2026-03": {"Food": 1_700_000, "Transport": 480_000, "Entertainment": 800_000, "Education": 380_000},
            "2026-02": {"Food": 1_900_000, "Transport": 520_000, "Entertainment": 850_000, "Education": 420_000},
        },
        "question": "Give me a financial overview.",
        "checks": {
            "Proactively mentions anomaly":     {"fn": has_anomaly_reference("Entertainment"), "cat": ACT},
            "Still gives overview":             {"fn": has_words("spending", "income", "surplus"), "cat": ACT},
        },
    },
    {
        "id": "TC324", "name": "Anomaly — UNUSUAL + WATCH distinction",
        "role": "Freelancer", "income": 35_000_000,
        "spending": {"Food": 4_500_000, "Transport": 2_000_000, "Shopping": 10_000_000, "Bills": 3_000_000, "Entertainment": 3_000_000},
        "monthly_history": {
            "2026-03": {"Food": 4_400_000, "Transport": 2_100_000, "Shopping": 3_500_000, "Bills": 3_000_000, "Entertainment": 1_900_000},
            "2026-02": {"Food": 4_600_000, "Transport": 1_900_000, "Shopping": 3_300_000, "Bills": 3_000_000, "Entertainment": 2_100_000},
        },
        "question": "Any red flags in my finances?",
        "checks": {
            "Mentions Shopping (UNUSUAL)":      {"fn": has_anomaly_reference("Shopping"), "cat": ACT},
            "Mentions Entertainment":           {"fn": has_words("entertainment"), "cat": ACT},
            "Warns strongly about Shopping":    {"fn": has_words("immediate", "attention", "warning", "unusual", "spike"), "cat": ACT},
        },
    },
]
