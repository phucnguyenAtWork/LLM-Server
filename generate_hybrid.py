"""
FINA Training Data Generator
==============================
Generates high-quality SFT training data for the FINA financial AI.
Covers 3 user roles (Student, Worker, Freelancer) with realistic
VND amounts, diverse question types, and role-specific responses.

Output: hybrid_data.jsonl
Usage:  python generate_hybrid.py
"""

import json
import random

OUTPUT_FILE = "hybrid_data.jsonl"

# ── System prompt (must match api.py exactly) ─────────────────────────────────
SYSTEM_PROMPT = """You are FINA, an intelligent Financial AI Agent.

### CAPABILITIES:
1. **Analysis:** Analyze spending patterns.
2. **Action:** You can LOG transactions if the user tells you to (e.g., "I spent 50k on coffee").
3. **Budgeting:** - Default: 50/30/20 Rule.
   - **Custom:** IF the user asks for a different split (e.g., 70/20/10), CALCULATE it for them. Do not lecture them.

### FORMATTING:
- Use **Bold** for numbers (e.g., **50.000 VND**).
- Always use the User's Currency."""

# ── Helpers ───────────────────────────────────────────────────────────────────

def vnd(amount: float) -> str:
    return "**{:,.0f} VND**".format(amount).replace(",", ".")

def pct(amount: float, income: float) -> str:
    return f"**{(amount / income * 100):.1f}%**"

def total_spent(spending: dict) -> float:
    return sum(spending.values())

def savings(income: float, spending: dict) -> float:
    return income - total_spent(spending)

def make_context(income: float, spending: dict) -> str:
    lines = "\n".join(f"    - {k}: {vnd(v)}" for k, v in spending.items())
    return (
        f"--- FINANCIAL CONTEXT ---\n"
        f"CURRENCY: VND\n"
        f"TOTAL INCOME: {vnd(income)}\n\n"
        f"ACTUAL SPENDING BY CATEGORY:\n{lines}\n\n"
        f"TARGET BUDGETS (50/30/20 Rule):\n"
        f"- Needs Limit (50%):    {vnd(income * 0.50)}\n"
        f"- Wants Limit (30%):    {vnd(income * 0.30)}\n"
        f"- Savings Target (20%): {vnd(income * 0.20)}\n"
        f"---------------------------"
    )

def chat(role: str, context: str, question: str, response: str) -> dict:
    user_msg = (
        f"User Role: {role}\n"
        f"Selected Mode: Standard\n"
        f"Financial Context:\n{context}\n"
        f"User Question: {question}"
    )
    text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}\n<|im_end|>\n"
        f"<|im_start|>assistant\n{response}\n<|im_end|>"
    )
    return {"text": text}

def rvar(base: float, pct_range: float = 0.2) -> float:
    """Add ±pct_range random variance to a base amount, rounded to 50k."""
    v = base * random.uniform(1 - pct_range, 1 + pct_range)
    return round(v / 50_000) * 50_000


# ── Role-specific scenario factories ─────────────────────────────────────────

def student_scenario():
    income = rvar(random.choice([3_500_000, 4_000_000, 5_000_000, 6_000_000, 8_000_000]))
    spending = {
        "Food":          rvar(income * random.uniform(0.25, 0.45)),
        "Transport":     rvar(income * random.uniform(0.08, 0.15)),
        "Entertainment": rvar(income * random.uniform(0.08, 0.18)),
        "Education":     rvar(income * random.uniform(0.05, 0.15)),
        "Shopping":      rvar(income * random.uniform(0.05, 0.15)),
    }
    return income, spending

def worker_scenario():
    income = rvar(random.choice([10_000_000, 15_000_000, 20_000_000, 25_000_000, 30_000_000]))
    spending = {
        "Food":          rvar(income * random.uniform(0.20, 0.30)),
        "Transport":     rvar(income * random.uniform(0.08, 0.12)),
        "Bills":         rvar(income * random.uniform(0.10, 0.20)),
        "Shopping":      rvar(income * random.uniform(0.08, 0.15)),
        "Entertainment": rvar(income * random.uniform(0.05, 0.10)),
        "Health":        rvar(income * random.uniform(0.02, 0.06)),
    }
    return income, spending

def freelancer_scenario():
    income = rvar(random.choice([8_000_000, 12_000_000, 20_000_000, 35_000_000, 50_000_000]))
    spending = {
        "Food":          rvar(income * random.uniform(0.15, 0.25)),
        "Transport":     rvar(income * random.uniform(0.05, 0.12)),
        "Bills":         rvar(income * random.uniform(0.08, 0.15)),
        "Shopping":      rvar(income * random.uniform(0.05, 0.12)),
        "Entertainment": rvar(income * random.uniform(0.03, 0.08)),
        "Health":        rvar(income * random.uniform(0.02, 0.05)),
    }
    return income, spending

SCENARIO_MAP = {
    "Student":    student_scenario,
    "Worker":     worker_scenario,
    "Freelancer": freelancer_scenario,
}


# ════════════════════════════════════════════════════════════════════════════════
# QUESTION TYPE GENERATORS
# ════════════════════════════════════════════════════════════════════════════════

# ── 1. Budget Health Check ────────────────────────────────────────────────────

def gen_budget_health(role: str) -> dict:
    income, spending = SCENARIO_MAP[role]()
    ctx   = make_context(income, spending)
    spent = total_spent(spending)
    save  = savings(income, spending)
    ratio = spent / income * 100
    status = "healthy" if ratio < 80 else "tight"

    q = random.choice([
        "How is my budget looking this month?",
        "Am I on track financially?",
        "Give me a budget health check.",
        "What is my overall financial status?",
        "How am I doing with my money this month?",
    ])

    if ratio < 80:
        r = (
            f"Your budget is in good shape. You spent {vnd(spent)} out of {vnd(income)}, "
            f"leaving {vnd(save)} ({pct(save, income)} of income) as savings. "
            f"You are well within the 50/30/20 guideline — keep it up!"
        )
    elif ratio < 100:
        top_cat = max(spending, key=spending.get)
        r = (
            f"Your budget is tight this month. You spent {vnd(spent)}, leaving only {vnd(save)} saved. "
            f"Your biggest expense is **{top_cat}** at {vnd(spending[top_cat])}. "
            f"Try to reduce it next month to hit the 20% savings target of {vnd(income * 0.20)}."
        )
    else:
        r = (
            f"You are over budget this month — spent {vnd(spent)} against income of {vnd(income)}, "
            f"resulting in a deficit of {vnd(abs(save))}. "
            f"Immediate priority: cut discretionary spending until you are back in the black."
        )
    return chat(role, ctx, q, r)


# ── 2. Overspending Alert ─────────────────────────────────────────────────────

def gen_overspending(role: str) -> dict:
    income, spending = SCENARIO_MAP[role]()
    ctx = make_context(income, spending)

    # Force overspend on a random category
    cat = random.choice(list(spending.keys()))
    spending[cat] = income * random.uniform(0.35, 0.55)
    ctx = make_context(income, spending)

    q = random.choice([
        f"Am I spending too much on {cat}?",
        f"Why is my {cat} spend so high?",
        f"Is {cat} eating too much of my budget?",
        f"How does my {cat} spending compare to my budget?",
    ])

    budget_limit = income * 0.30
    r = (
        f"Yes — your {cat} spending of {vnd(spending[cat])} is {pct(spending[cat], income)} of your income, "
        f"which is well above a healthy allocation. "
        f"A good target is to keep discretionary categories under {vnd(budget_limit)} (30% of income). "
        f"Try to cut {vnd(spending[cat] - budget_limit * 0.3)} from {cat} next month."
    )
    return chat(role, ctx, q, r)


# ── 3. Saving Tips (role-specific) ────────────────────────────────────────────

STUDENT_TIPS = [
    "Use your student ID for discounts on transport, software (GitHub Pro, Notion), and food apps.",
    "Cook at home 3x per week — this alone can save 500.000–800.000 VND monthly.",
    "Cancel any subscriptions you use less than once a week. Netflix shared plan cuts cost by 60%.",
    "Use Grab Food vouchers and Shopee Food flash deals instead of ordering at full price.",
    "Set a weekly cash limit and withdraw it — when it's gone, spending stops.",
    "Track every expense in a notes app for 30 days. Awareness alone reduces spending by ~15%.",
    "Apply for a part-time job or freelance gig — even 2M VND/month changes everything.",
]

WORKER_TIPS = [
    "Automate savings on payday — transfer your savings target immediately before you can spend it.",
    "Build a 3–6 month emergency fund (at least 30–60M VND) before investing.",
    "Max out any employer contribution matching — that is a 100% instant return.",
    "Review all subscriptions and recurring bills annually. Cancel anything unused.",
    "Use meal prep Sundays to reduce food spend by 20–30% with no sacrifice in quality.",
    "Negotiate your salary every 12–18 months — even a 10% raise compounds massively over time.",
    "Move idle savings to a high-interest savings account or short-term T-bills.",
]

FREELANCER_TIPS = [
    "Set aside 25–30% of every payment for taxes immediately — before you touch it.",
    "Open a separate business account. Never mix personal and business spending.",
    "Build a 6-month income buffer (living expenses) before taking on risk or cutting rates.",
    "Track deductible business expenses: internet, equipment, software, co-working space.",
    "Invoice promptly and follow up on late payments — cash flow is your lifeblood.",
    "Use the lean months to upskill, not to panic-spend. Invest in tools that increase your rate.",
    "Set a fixed 'salary' to pay yourself each month, even in good months, to avoid lifestyle inflation.",
]

TIPS_MAP = {
    "Student":    STUDENT_TIPS,
    "Worker":     WORKER_TIPS,
    "Freelancer": FREELANCER_TIPS,
}

def gen_saving_tips(role: str) -> dict:
    income, spending = SCENARIO_MAP[role]()
    ctx  = make_context(income, spending)
    tips = random.sample(TIPS_MAP[role], k=random.randint(2, 3))

    q = random.choice([
        "Give me some saving tips.",
        "How can I save more money?",
        "What should I do to improve my finances?",
        "I want to save more — any advice?",
        "What are your top saving recommendations for me?",
    ])

    tip_lines = "\n".join(f"- {t}" for t in tips)
    r = f"Here are practical tips for your situation as a {role}:\n{tip_lines}"
    return chat(role, ctx, q, r)


# ── 4. Transaction Logging ────────────────────────────────────────────────────

LOG_TRANSACTIONS = [
    (50_000,   "coffee",             "Food"),
    (120_000,  "lunch at the office","Food"),
    (250_000,  "Grab Food dinner",   "Food"),
    (35_000,   "Grab bike",          "Transport"),
    (200_000,  "petrol",             "Transport"),
    (500_000,  "Shopee order",       "Shopping"),
    (150_000,  "Watsons skincare",   "Shopping"),
    (89_000,   "Netflix",            "Entertainment"),
    (1_200_000,"tuition installment","Education"),
    (300_000,  "pharmacy medicine",  "Health"),
    (2_500_000,"monthly rent",       "Bills"),
    (180_000,  "FPT Internet",       "Bills"),
    (75_000,   "CGV movie ticket",   "Entertainment"),
    (450_000,  "gym monthly fee",    "Health"),
    (80_000,   "Highlands Coffee",   "Food"),
]

def gen_log_transaction(role: str) -> dict:
    income, spending = SCENARIO_MAP[role]()
    ctx = make_context(income, spending)

    amount, item, category = random.choice(LOG_TRANSACTIONS)
    amount = rvar(amount, 0.3)

    q = random.choice([
        f"I just spent {vnd(amount)} on {item}.",
        f"Log {vnd(amount)} for {item}.",
        f"Add a {category} expense: {vnd(amount)} for {item}.",
        f"I spent {vnd(amount)} on {item} today.",
    ])

    r = (
        f"Logged: **{category}** — {vnd(amount)} for {item}. ✓\n"
        f"Your {category} total this month is now {vnd(spending.get(category, 0) + amount)}."
    )
    return chat(role, ctx, q, r)


# ── 5. Custom Budget Split ────────────────────────────────────────────────────

CUSTOM_SPLITS = [
    (70, 20, 10),
    (60, 25, 15),
    (40, 30, 30),
    (55, 25, 20),
    (50, 20, 30),
    (65, 15, 20),
    (45, 35, 20),
]

def gen_custom_budget(role: str) -> dict:
    income, spending = SCENARIO_MAP[role]()
    ctx = make_context(income, spending)
    n, w, s = random.choice(CUSTOM_SPLITS)

    q = random.choice([
        f"Calculate a {n}/{w}/{s} budget split for me.",
        f"What would a {n}/{w}/{s} rule look like on my income?",
        f"I want to try a {n}/{w}/{s} budget instead. What are my limits?",
        f"Show me a {n}% needs, {w}% wants, {s}% savings breakdown.",
    ])

    r = (
        f"Here is your custom **{n}/{w}/{s}** budget on {vnd(income)} income:\n"
        f"- Needs ({n}%): {vnd(income * n / 100)}\n"
        f"- Wants ({w}%): {vnd(income * w / 100)}\n"
        f"- Savings ({s}%): {vnd(income * s / 100)}"
    )
    return chat(role, ctx, q, r)


# ── 6. Savings Rate ───────────────────────────────────────────────────────────

def gen_savings_rate(role: str) -> dict:
    income, spending = SCENARIO_MAP[role]()
    ctx  = make_context(income, spending)
    save = savings(income, spending)
    rate = save / income * 100

    q = random.choice([
        "What is my savings rate?",
        "How much am I actually saving this month?",
        "What percentage of my income do I keep?",
        "Am I saving enough?",
    ])

    target = 20
    if rate >= target:
        verdict = f"That is {pct(save, income)}, which beats the 20% target. Well done!"
    elif rate > 0:
        gap = income * 0.20 - save
        verdict = (
            f"That is {pct(save, income)} — below the 20% target. "
            f"You need to save an extra {vnd(gap)} to hit it."
        )
    else:
        verdict = f"You are in deficit by {vnd(abs(save))} this month. No savings occurred — prioritize cutting expenses immediately."

    r = f"You saved {vnd(save)} this month. {verdict}"
    return chat(role, ctx, q, r)


# ── 7. Category Breakdown ─────────────────────────────────────────────────────

def gen_category_question(role: str) -> dict:
    income, spending = SCENARIO_MAP[role]()
    ctx = make_context(income, spending)
    cat = random.choice(list(spending.keys()))

    q = random.choice([
        f"How much did I spend on {cat} this month?",
        f"What is my {cat} budget usage?",
        f"Show me my {cat} spending.",
        f"How does my {cat} compare to my budget?",
    ])

    allocated = income * 0.30
    amt = spending[cat]
    status = "within budget" if amt <= allocated else f"over budget by {vnd(amt - allocated)}"

    r = (
        f"Your {cat} spend this month is {vnd(amt)} ({pct(amt, income)} of income). "
        f"That is {status}."
    )
    return chat(role, ctx, q, r)


# ── 8. Emergency Fund ─────────────────────────────────────────────────────────

def gen_emergency_fund(role: str) -> dict:
    income, spending = SCENARIO_MAP[role]()
    ctx    = make_context(income, spending)
    months = 3 if role == "Student" else (6 if role == "Worker" else 6)
    spent  = total_spent(spending)
    target = spent * months

    q = random.choice([
        "How much should I have in an emergency fund?",
        "What is a good emergency fund for me?",
        "I want to build an emergency fund — how much?",
        "How do I plan for emergencies?",
    ])

    r = (
        f"Based on your monthly expenses of {vnd(spent)}, your emergency fund target is "
        f"{vnd(target)} ({months} months of expenses). "
        f"Start by saving {vnd(income * 0.10)} per month into a separate account until you reach that target."
    )
    return chat(role, ctx, q, r)


# ── 9. Income vs Expense Summary ─────────────────────────────────────────────

def gen_income_vs_expense(role: str) -> dict:
    income, spending = SCENARIO_MAP[role]()
    ctx   = make_context(income, spending)
    spent = total_spent(spending)
    save  = savings(income, spending)

    q = random.choice([
        "Summarize my income and expenses.",
        "Give me an income vs expense breakdown.",
        "How much did I earn vs spend this month?",
        "What is my net cash flow this month?",
    ])

    r = (
        f"This month: Income {vnd(income)} | Expenses {vnd(spent)} | Net {vnd(save)}.\n"
        f"You spent {pct(spent, income)} of your income and saved {pct(save, income)}."
    )
    return chat(role, ctx, q, r)


# ── 10. Goal-Based Saving ─────────────────────────────────────────────────────

GOALS = [
    (5_000_000,   "a new phone"),
    (15_000_000,  "a laptop"),
    (30_000_000,  "an emergency fund"),
    (50_000_000,  "a trip abroad"),
    (100_000_000, "a motorbike"),
    (200_000_000, "a house down payment deposit"),
    (10_000_000,  "an online course"),
    (3_000_000,   "new clothes for Tet"),
]

def gen_goal_saving(role: str) -> dict:
    income, spending = SCENARIO_MAP[role]()
    ctx  = make_context(income, spending)
    save = max(savings(income, spending), 500_000)
    goal_amount, goal_name = random.choice(GOALS)

    months = goal_amount / save
    q = random.choice([
        f"How long to save for {goal_name} costing {vnd(goal_amount)}?",
        f"I want to buy {goal_name} ({vnd(goal_amount)}). When can I afford it?",
        f"Help me plan saving for {goal_name}.",
    ])

    extra = goal_amount / max(months - 1, 1) - save
    r = (
        f"At your current savings rate of {vnd(save)}/month, you can afford {goal_name} in "
        f"**{months:.1f} months**. "
        f"To reach it faster, increase your monthly savings by {vnd(extra)} "
        f"and you'll get there in under {int(months)} months."
    )
    return chat(role, ctx, q, r)


# ── 11. Role-Specific Deep Advice ─────────────────────────────────────────────

def gen_student_debt_advice() -> dict:
    income, spending = student_scenario()
    ctx = make_context(income, spending)
    q   = random.choice([
        "Should I use a credit card as a student?",
        "Is it okay to borrow money for lifestyle expenses?",
        "My friend says credit cards are fine. Should I get one?",
    ])
    r = (
        "Avoid credit card debt as a student — interest rates of 20–35%/year will trap you fast. "
        f"Your income is {vnd(income)}/month, which is tight. "
        "If you need a card, only use it for emergencies and pay the full balance every month. "
        "Build an emergency fund of 3–5M VND first before considering any credit products."
    )
    return chat("Student", ctx, q, r)

def gen_freelancer_tax_advice() -> dict:
    income, spending = freelancer_scenario()
    ctx    = make_context(income, spending)
    tax_30 = income * 0.30
    q      = random.choice([
        "How much should I set aside for taxes?",
        "I just got paid 20M VND — how much is for taxes?",
        "When should I save for taxes as a freelancer?",
        "What is the 30% tax rule?",
    ])
    r = (
        f"As a freelancer, set aside **30%** of every payment for taxes immediately. "
        f"On your income of {vnd(income)}, that is {vnd(tax_30)} — move it to a separate account today. "
        "This prevents the common mistake of spending tax money and scrambling at tax time. "
        "Track deductible business expenses (internet, equipment, software) to reduce your taxable income."
    )
    return chat("Freelancer", ctx, q, r)

def gen_worker_investment_advice() -> dict:
    income, spending = worker_scenario()
    ctx  = make_context(income, spending)
    save = savings(income, spending)
    q    = random.choice([
        "How should I start investing?",
        "I have some savings — what should I invest in?",
        "Is now a good time to start investing?",
        "What should a worker like me invest in?",
    ])
    r = (
        f"Before investing, ensure your emergency fund is at least {vnd(income * 6)} (6 months of income). "
        f"With {vnd(save)} saved monthly, start with: "
        f"(1) Low-cost index funds or government bonds for long-term growth, "
        f"(2) BHXH / pension maximization if your employer matches contributions. "
        "Only invest money you won't need for 3+ years."
    )
    return chat("Worker", ctx, q, r)

def gen_freelancer_income_buffer() -> dict:
    income, spending = freelancer_scenario()
    ctx   = make_context(income, spending)
    spent = total_spent(spending)
    buf   = spent * 6
    q     = random.choice([
        "My income fluctuates a lot — how do I manage that?",
        "Some months I earn 50M, some months 5M. What should I do?",
        "How do I budget with irregular income?",
        "What is an income buffer and how do I build one?",
    ])
    r = (
        f"With irregular income, pay yourself a fixed 'salary' each month — say {vnd(spent * 1.1)} — "
        f"and put all excess into an income buffer account. "
        f"Your buffer target: {vnd(buf)} (6 months of expenses). "
        "In lean months you draw from the buffer; in good months you refill it. "
        "This kills the feast-or-famine cycle and lets you plan predictably."
    )
    return chat("Freelancer", ctx, q, r)

def gen_student_part_time() -> dict:
    income, spending = student_scenario()
    ctx   = make_context(income, spending)
    spent = total_spent(spending)
    q     = random.choice([
        "Should I get a part-time job?",
        "Is it worth working while studying?",
        "I'm always short on money — should I work part-time?",
    ])
    extra = 2_000_000
    r = (
        f"Yes — even a part-time income of {vnd(extra)}/month would cover your current deficit of "
        f"{vnd(max(spent - income, 0))} and build savings. "
        "Look for flexible work: tutoring, delivery, content creation, or online freelance tasks. "
        "Keep it under 15–20 hours/week so it does not affect your studies."
    )
    return chat("Student", ctx, q, r)

def gen_worker_salary_split() -> dict:
    income, spending = worker_scenario()
    ctx = make_context(income, spending)
    q   = random.choice([
        "I just got paid. What should I do first?",
        "How should I split my salary on payday?",
        "What is the right order to use my paycheck?",
    ])
    r = (
        f"On payday, split {vnd(income)} in this order:\n"
        f"1. **Savings first**: Transfer {vnd(income * 0.20)} immediately — treat it like a bill.\n"
        f"2. **Fixed bills**: Pay rent, utilities, loans ({vnd(income * 0.30)}).\n"
        f"3. **Remaining ({vnd(income * 0.50)})**: Use for food, transport, and wants.\n"
        "This 'pay yourself first' method is the single most effective budgeting habit."
    )
    return chat("Worker", ctx, q, r)


# ── 12. Positive Reinforcement ────────────────────────────────────────────────

def gen_positive_feedback(role: str) -> dict:
    income, spending = SCENARIO_MAP[role]()
    # Force a healthy scenario
    spending = {k: v * 0.6 for k, v in spending.items()}
    ctx  = make_context(income, spending)
    save = savings(income, spending)

    q = random.choice([
        "How am I doing this month?",
        "Did I do well this month?",
        "Am I saving enough?",
    ])
    r = (
        f"Excellent work this month! You saved {vnd(save)} ({pct(save, income)} of income), "
        f"well above the 20% target of {vnd(income * 0.20)}. "
        "Keep this up and you will build significant wealth over time."
    )
    return chat(role, ctx, q, r)


# ── 13. Top Expense Identification ────────────────────────────────────────────

def gen_top_expense(role: str) -> dict:
    income, spending = SCENARIO_MAP[role]()
    ctx     = make_context(income, spending)
    top_cat = max(spending, key=spending.get)
    top_amt = spending[top_cat]

    q = random.choice([
        "What is my biggest expense?",
        "Where does most of my money go?",
        "Which category am I spending the most on?",
    ])
    r = (
        f"Your biggest expense this month is **{top_cat}** at {vnd(top_amt)} "
        f"({pct(top_amt, income)} of income). "
        f"If you can reduce this by 20%, you would save an extra {vnd(top_amt * 0.20)} per month."
    )
    return chat(role, ctx, q, r)


# ── 14. No Income / Edge Cases ────────────────────────────────────────────────

def gen_broke_student() -> dict:
    income  = 2_000_000
    spending = {"Food": 1_800_000, "Transport": 300_000, "Entertainment": 400_000}
    ctx     = make_context(income, spending)
    spent   = total_spent(spending)

    q = "I'm running out of money every month. What do I do?"
    r = (
        f"You are spending {vnd(spent)} on {vnd(income)} income — a deficit of {vnd(spent - income)}. "
        "Immediate actions: "
        f"(1) Cut Entertainment to under {vnd(100_000)}. "
        "(2) Cook instead of ordering food — target Food under 1.000.000 VND. "
        "(3) Speak to family about a temporary increase in allowance or find a part-time income source. "
        "You cannot save your way out of a deficit — income must increase or spending must drop drastically."
    )
    return chat("Student", ctx, q, r)


# ── 15. Spending Trend Warning ────────────────────────────────────────────────

def gen_trend_warning(role: str) -> dict:
    income, spending = SCENARIO_MAP[role]()
    # Inflate to simulate over-budget
    spending = {k: v * random.uniform(1.2, 1.6) for k, v in spending.items()}
    ctx   = make_context(income, spending)
    spent = total_spent(spending)

    q = random.choice([
        "Is my spending getting worse?",
        "My expenses seem high this month.",
        "I feel like I am spending too much.",
    ])
    over = spent - income
    r = (
        f"Your total spending of {vnd(spent)} exceeds your income of {vnd(income)} by {vnd(over)}. "
        f"This is unsustainable. The main culprits are: "
        + ", ".join(f"**{k}** ({vnd(v)})" for k, v in sorted(spending.items(), key=lambda x: -x[1])[:2])
        + ". Cut these categories first to return to a positive cash flow."
    )
    return chat(role, ctx, q, r)


# ════════════════════════════════════════════════════════════════════════════════
# ASSEMBLE DATASET
# ════════════════════════════════════════════════════════════════════════════════

GENERATORS = [
    gen_budget_health,
    gen_overspending,
    gen_saving_tips,
    gen_log_transaction,
    gen_custom_budget,
    gen_savings_rate,
    gen_category_question,
    gen_emergency_fund,
    gen_income_vs_expense,
    gen_goal_saving,
    gen_top_expense,
    gen_positive_feedback,
    gen_trend_warning,
]

ROLE_ONLY_GENERATORS = {
    "Student":    [gen_student_debt_advice, gen_student_part_time, gen_broke_student],
    "Worker":     [gen_worker_investment_advice, gen_worker_salary_split],
    "Freelancer": [gen_freelancer_tax_advice, gen_freelancer_income_buffer],
}

def fetch_finance_alpaca(n: int = 2000) -> list[dict]:
    """Pull general finance knowledge from the gbharti/finance-alpaca dataset."""
    try:
        from datasets import load_dataset
        print(f"Downloading finance-alpaca dataset ({n} examples)...")
        ds = load_dataset("gbharti/finance-alpaca", split="train")
        ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
        results = []
        for row in ds:
            user_text = row["instruction"]
            if row.get("input"):
                user_text += f"\nContext: {row['input']}"
            text = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
                f"<|im_start|>user\n{user_text}\n<|im_end|>\n"
                f"<|im_start|>assistant\n{row['output']}\n<|im_end|>"
            )
            results.append({"text": text})
        print(f"Loaded {len(results)} Alpaca examples.")
        return results
    except Exception as e:
        print(f"Could not load Alpaca dataset: {e} — skipping.")
        return []


def generate_all(n_per_generator: int = 40) -> list[dict]:
    dataset = []
    roles   = ["Student", "Worker", "Freelancer"]

    # Generic generators across all roles
    for gen in GENERATORS:
        for role in roles:
            for _ in range(n_per_generator):
                try:
                    dataset.append(gen(role))
                except Exception as e:
                    print(f"  [skip] {gen.__name__}/{role}: {e}")

    # Role-specific generators
    for role, gens in ROLE_ONLY_GENERATORS.items():
        for gen in gens:
            for _ in range(n_per_generator * 2):
                try:
                    result = gen() if gen in [gen_student_debt_advice, gen_student_part_time,
                                               gen_broke_student, gen_worker_investment_advice,
                                               gen_worker_salary_split, gen_freelancer_tax_advice,
                                               gen_freelancer_income_buffer] else gen(role)
                    dataset.append(result)
                except Exception as e:
                    print(f"  [skip] {gen.__name__}: {e}")

    return dataset


if __name__ == "__main__":
    random.seed(42)
    dataset = []

    # 1. Generate FINA-specific synthetic data
    print("Generating FINA synthetic training data...")
    synthetic = generate_all(n_per_generator=40)
    dataset.extend(synthetic)
    print(f"Generated {len(synthetic)} synthetic examples.")

    # 2. Fetch Finance-Alpaca for broad financial knowledge
    alpaca = fetch_finance_alpaca(n=2000)
    dataset.extend(alpaca)

    # 3. Shuffle
    random.shuffle(dataset)

    # 4. Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nDone! Saved {len(dataset)} total training examples to '{OUTPUT_FILE}'.")
    print(f"  - Synthetic (FINA): {len(synthetic)}")
    print(f"  - Finance-Alpaca:   {len(alpaca)}")
