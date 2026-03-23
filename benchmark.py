"""
FINA Benchmark
===============
Tests the fine-tuned model across all 3 roles with fixed scenarios.
Evaluates: format compliance, calculation accuracy, role specificity, latency.

Usage: python benchmark.py
"""

import re
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL   = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_NAME = "financial_qwen_native_v1"

SYSTEM_PROMPT = """You are FINA, an intelligent Financial AI Agent.

### CAPABILITIES:
1. **Analysis:** Analyze spending patterns.
2. **Action:** You can LOG transactions if the user tells you to (e.g., "I spent 50k on coffee").
3. **Budgeting:** - Default: 50/30/20 Rule.
   - **Custom:** IF the user asks for a different split (e.g., 70/20/10), CALCULATE it for them. Do not lecture them.

### FORMATTING:
- Use **Bold** for numbers (e.g., **50.000 VND**).
- Always use the User's Currency."""


def vnd(n):
    return "{:,.0f}".format(n).replace(",", ".")

def make_context(income, spending):
    lines = "\n".join(f"    - {k}: {vnd(v)} VND" for k, v in spending.items())
    return (
        f"--- FINANCIAL CONTEXT ---\n"
        f"CURRENCY: VND\n"
        f"TOTAL INCOME: {vnd(income)} VND\n\n"
        f"ACTUAL SPENDING BY CATEGORY:\n{lines}\n\n"
        f"TARGET BUDGETS (50/30/20 Rule):\n"
        f"- Needs Limit (50%):    {vnd(income*0.5)} VND\n"
        f"- Wants Limit (30%):    {vnd(income*0.3)} VND\n"
        f"- Savings Target (20%): {vnd(income*0.2)} VND\n"
        f"---------------------------"
    )


# ── Test cases ────────────────────────────────────────────────────────────────
TEST_CASES = [
    {
        "id":       "TC01",
        "name":     "Budget Health Check — Student",
        "role":     "Student",
        "income":   5_000_000,
        "spending": {"Food": 2_200_000, "Transport": 500_000, "Entertainment": 900_000, "Education": 300_000},
        "question": "How is my budget looking this month?",
        "checks": {
            "Uses VND":             lambda r: "VND" in r,
            "Uses bold formatting": lambda r: "**" in r,
            "Mentions savings":     lambda r: "sav" in r.lower(),
            "Mentions spending":    lambda r: "spent" in r.lower() or "spend" in r.lower(),
        },
    },
    {
        "id":       "TC02",
        "name":     "Custom Budget Split — Worker",
        "role":     "Worker",
        "income":   20_000_000,
        "spending": {"Food": 4_000_000, "Transport": 2_000_000, "Bills": 3_000_000, "Shopping": 2_500_000},
        "question": "Calculate a 60/25/15 budget split for me.",
        "checks": {
            "Uses VND":             lambda r: "VND" in r,
            "Correct needs (60%)":  lambda r: "12.000.000" in r or "12,000,000" in r or "12000000" in r,
            "Correct wants (25%)":  lambda r: "5.000.000" in r or "5,000,000" in r or "5000000" in r,
            "Correct savings (15%)":lambda r: "3.000.000" in r or "3,000,000" in r or "3000000" in r,
            "Mentions 60/25/15":    lambda r: "60" in r and "25" in r and "15" in r,
        },
    },
    {
        "id":       "TC03",
        "name":     "Tax Advice — Freelancer",
        "role":     "Freelancer",
        "income":   30_000_000,
        "spending": {"Food": 3_000_000, "Transport": 1_500_000, "Bills": 2_000_000, "Health": 500_000},
        "question": "How much should I set aside for taxes?",
        "checks": {
            "Uses VND":             lambda r: "VND" in r,
            "Mentions 30% rule":    lambda r: "30%" in r or "30 %" in r or "30" in r,
            "Mentions tax amount":  lambda r: "9.000.000" in r or "9,000,000" in r,
            "Mentions separate":    lambda r: "separate" in r.lower() or "account" in r.lower(),
        },
    },
    {
        "id":       "TC04",
        "name":     "Transaction Log — Student",
        "role":     "Student",
        "income":   4_000_000,
        "spending": {"Food": 1_200_000, "Transport": 400_000, "Entertainment": 300_000},
        "question": "I just spent 85.000 VND on Highlands Coffee.",
        "checks": {
            "Confirms log":         lambda r: "log" in r.lower() or "recorded" in r.lower() or "added" in r.lower() or "logged" in r.lower(),
            "Mentions amount":      lambda r: "85" in r,
            "Mentions category":    lambda r: "food" in r.lower() or "coffee" in r.lower() or "beverage" in r.lower(),
        },
    },
    {
        "id":       "TC05",
        "name":     "Emergency Fund — Worker",
        "role":     "Worker",
        "income":   15_000_000,
        "spending": {"Food": 3_000_000, "Transport": 1_500_000, "Bills": 2_500_000, "Shopping": 1_500_000, "Health": 500_000},
        "question": "How much should I have in an emergency fund?",
        "checks": {
            "Uses VND":             lambda r: "VND" in r,
            "Mentions months":      lambda r: "month" in r.lower(),
            "Mentions amount":      lambda r: any(x in r for x in ["9.000.000","27.000.000","54.000.000","45.000.000","36.000.000"]),
            "Gives actionable tip": lambda r: "save" in r.lower() or "transfer" in r.lower() or "set aside" in r.lower(),
        },
    },
    {
        "id":       "TC06",
        "name":     "Saving Tips — Freelancer",
        "role":     "Freelancer",
        "income":   12_000_000,
        "spending": {"Food": 2_500_000, "Transport": 800_000, "Bills": 1_500_000, "Entertainment": 1_000_000},
        "question": "Give me saving tips for my situation.",
        "checks": {
            "Role-specific advice": lambda r: any(x in r.lower() for x in ["freelanc", "tax", "irregular", "income buffer", "invoice"]),
            "Multiple tips":        lambda r: r.count("-") >= 2 or r.count("\n") >= 2,
            "Actionable language":  lambda r: any(x in r.lower() for x in ["set aside", "save", "open", "track", "build", "avoid"]),
        },
    },
    {
        "id":       "TC07",
        "name":     "Overspending Alert — Student",
        "role":     "Student",
        "income":   4_000_000,
        "spending": {"Food": 2_500_000, "Transport": 600_000, "Entertainment": 1_200_000, "Shopping": 800_000},
        "question": "Am I spending too much on Entertainment?",
        "checks": {
            "Uses VND":             lambda r: "VND" in r,
            "Mentions amount":      lambda r: "1.200.000" in r or "1,200,000" in r or "1200" in r,
            "Warns overspending":   lambda r: any(x in r.lower() for x in ["over", "high", "too much", "above", "exceed"]),
            "Gives recommendation": lambda r: any(x in r.lower() for x in ["cut", "reduc", "limit", "target"]),
        },
    },
    {
        "id":       "TC08",
        "name":     "Goal Saving Plan — Worker",
        "role":     "Worker",
        "income":   18_000_000,
        "spending": {"Food": 3_500_000, "Transport": 2_000_000, "Bills": 3_000_000, "Shopping": 2_000_000, "Entertainment": 1_000_000},
        "question": "I want to save for a motorbike worth 45.000.000 VND. How long will it take?",
        "checks": {
            "Uses VND":             lambda r: "VND" in r,
            "Mentions months":      lambda r: "month" in r.lower(),
            "Mentions goal amount": lambda r: "45" in r,
            "Calculates timeline":  lambda r: any(c.isdigit() for c in r),
        },
    },
]


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, tokenizer, role, context, question):
    user_msg = (
        f"User Role: {role}\n"
        f"Selected Mode: Standard\n"
        f"Financial Context:\n{context}\n"
        f"User Question: {question}"
    )
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_msg},
    ]
    text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.time() - t0

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    clean    = response.split("assistant")[-1].strip() if "assistant" in response else response
    tokens   = outputs.shape[-1] - inputs["input_ids"].shape[-1]
    return clean, latency, tokens


# ── Runner ────────────────────────────────────────────────────────────────────

def run_benchmark():
    print("Loading FINA model...")
    tokenizer  = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_NAME)
    model.eval()
    print("Model loaded.\n")
    print("=" * 70)
    print("FINA BENCHMARK RESULTS")
    print("=" * 70)

    total_checks = 0
    total_passed = 0
    results      = []

    for tc in TEST_CASES:
        ctx  = make_context(tc["income"], tc["spending"])
        resp, latency, tokens = run_inference(model, tokenizer, tc["role"], ctx, tc["question"])

        passed = {name: fn(resp) for name, fn in tc["checks"].items()}
        n_pass = sum(passed.values())
        n_total = len(passed)
        score   = n_pass / n_total * 100

        total_checks += n_total
        total_passed += n_pass

        results.append({
            "tc": tc, "response": resp, "passed": passed,
            "score": score, "latency": latency, "tokens": tokens,
        })

        print(f"\n[{tc['id']}] {tc['name']}")
        print(f"  Role: {tc['role']} | Income: {vnd(tc['income'])} VND")
        print(f"  Q: {tc['question']}")
        safe_resp = resp[:300].encode("ascii", errors="replace").decode("ascii")
        print(f"  A: {safe_resp}{'...' if len(resp) > 300 else ''}")
        print(f"  Checks: ", end="")
        for name, ok in passed.items():
            print(f"{'PASS' if ok else 'FAIL'}({name})", end="  ")
        print(f"\n  Score: {n_pass}/{n_total} ({score:.0f}%)  |  Latency: {latency:.1f}s  |  Tokens: {tokens}")
        print("-" * 70)

    # ── Summary ───────────────────────────────────────────────────────────────
    overall = total_passed / total_checks * 100
    avg_lat = sum(r["latency"] for r in results) / len(results)
    avg_tok = sum(r["tokens"]  for r in results) / len(results)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Overall Score : {total_passed}/{total_checks} checks passed ({overall:.1f}%)")
    print(f"  Avg Latency   : {avg_lat:.1f}s per response")
    print(f"  Avg Tokens    : {avg_tok:.0f} tokens generated")
    print()
    print("  Per-test scores:")
    for r in results:
        bar = "#" * int(r["score"] / 10) + "-" * (10 - int(r["score"] / 10))
        print(f"    [{r['tc']['id']}] [{bar}] {r['score']:5.1f}%  {r['tc']['name']}")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
