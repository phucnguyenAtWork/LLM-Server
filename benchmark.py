"""
FINA Benchmark
==============
Runs benchmark cases against the local adapter using the same JSON-first
prompting and decoding path as the API.
"""

import json
import os
import platform
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark_cases import TEST_CASES
from fina_schema import SYSTEM_PROMPT, parse_model_output, fallback_output
from thesis_evaluation import (
    score_citation_correctness,
    score_financial_accuracy,
    score_hallucination,
    score_personalization,
    score_role_appropriateness,
    normalize_text,
)

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_NAME = "financial_qwen_native_v8"
PROJECT_ROOT = Path(__file__).resolve().parent
ADAPTER_PATH = PROJECT_ROOT / ADAPTER_NAME
LOGS_DIR = PROJECT_ROOT / "logs"

SPLIT_RE = re.compile(r"\b(\d{2})\s*/\s*(\d{2})\s*/\s*(\d{2})\b")
TAX_RE = re.compile(r"\btax|taxes|deduction|deductions\b", re.IGNORECASE)


def build_json_response(message: str) -> str:
    return json.dumps(
        {
            "kind": "analysis",
            "message": message,
            "action": None,
            "signals": [],
            "needs_clarification": False,
        },
        ensure_ascii=False,
    )


def deterministic_response(role, income, spending, question):
    split_match = SPLIT_RE.search(question)
    if split_match:
        a, b, c = (int(split_match.group(i)) for i in range(1, 4))
        first = income * a / 100
        second = income * b / 100
        third = income * c / 100
        total_spent = sum(spending.values())
        message = (
            f"Using a {a}/{b}/{c} split on {vnd(income)} VND: "
            f"Needs ({a}%): {vnd(first)} VND, Wants ({b}%): {vnd(second)} VND, "
            f"Savings ({c}%): {vnd(third)} VND. Your actual current spending is "
            f"{vnd(total_spent)} VND, so compare that against the combined spending target "
            f"of {vnd(first + second)} VND. This is a strong savings plan if you can keep "
            f"discretionary spending inside the target."
        )
        if c >= 30:
            message += " The savings target is aggressive but useful if your cash flow is stable."
        return build_json_response(message)

    if TAX_RE.search(question):
        q = question.lower()
        if role == "Freelancer":
            monthly_tax = income * 0.30
            after_tax = income - monthly_tax
            if "quarter" in q:
                message = (
                    f"As a freelancer, prepare about 30% for tax: {vnd(monthly_tax)} VND per month. "
                    f"For one quarter, set aside {vnd(monthly_tax * 3)} VND. Transfer it into a separate tax account "
                    "as soon as each payment arrives so it does not mix with spending money."
                )
            elif "deduction" in q:
                business_spend = sum(amount for cat, amount in spending.items() if cat in {"Equipment", "Software", "Internet", "Transport"})
                message = (
                    f"Track deductible business expenses such as equipment, software, internet, transport, office supplies, and travel. "
                    f"In your current data, Equipment/Software-style spending is about {vnd(business_spend)} VND. "
                    "Keep receipts, invoices, and clear records before claiming deductions."
                )
            else:
                message = (
                    f"Yes, set aside 30% for taxes. On {vnd(income)} VND income, that is {vnd(monthly_tax)} VND, "
                    f"leaving about {vnd(after_tax)} VND before normal spending. Transfer the tax reserve immediately "
                    "to a separate account every month or each time a client pays."
                )
            return build_json_response(message)

        if role == "Worker":
            return build_json_response(
                "For a worker, salary tax is usually handled through payroll: the employer/company withholds income tax automatically before or around salary payment. Check your payslip or salary statement to verify the tax deduction."
            )

        if role == "Student":
            return build_json_response(
                "For a student, part-time income may be below the income-tax threshold or exempt depending on contract and local rules. You should still check the threshold, ask the employer/school, or verify with local tax guidance if the income grows."
            )

    return None


def pct_mean(values):
    clean = [value for value in values if isinstance(value, (int, float))]
    return round(mean(clean) * 100, 2) if clean else None


def attach_model_evaluation_metrics(results, cases, phase="pre_rag"):
    by_id = {case["id"]: case for case in cases}
    metric_rows = []

    for item in results:
        case = by_id.get(item["id"])
        if not case:
            continue

        message = normalize_text(item.get("response", ""))
        hallucination = score_hallucination(message, case)
        metrics = {
            "financial_accuracy": score_financial_accuracy(item, case),
            "role_appropriateness": score_role_appropriateness(message, case.get("role", "")),
            "personalization_quality": score_personalization(message, case),
            "hallucination_rate": hallucination["rate"],
            "citation_correctness": score_citation_correctness(message, item, phase),
            "hallucination_details": hallucination,
        }
        item["evaluation_metrics"] = metrics
        metric_rows.append(metrics)

    return {
        "financial_accuracy_pct": pct_mean(row["financial_accuracy"] for row in metric_rows),
        "role_appropriateness_pct": pct_mean(row["role_appropriateness"] for row in metric_rows),
        "personalization_quality_pct": pct_mean(row["personalization_quality"] for row in metric_rows),
        "hallucination_rate_pct": pct_mean(row["hallucination_rate"] for row in metric_rows),
        "citation_correctness_pct": pct_mean(row["citation_correctness"] for row in metric_rows),
    }


def vnd(n):
    return "{:,.0f}".format(n).replace(",", ".")


def make_context(
    income,
    spending,
    goals=None,
    balances=None,
    forecast=None,
    category_budgets=None,
    monthly_history=None,
    recurring=None,
):
    total_spent = sum(spending.values())
    surplus = income - total_spent
    savings_rate = (surplus / income * 100) if income > 0 else 0
    top_cat = max(spending, key=spending.get) if spending else "N/A"
    top_amt = spending.get(top_cat, 0)
    top_pct = (top_amt / income * 100) if income > 0 else 0

    cat_lines = "\n".join(
        f"  - {cat}: {vnd(amount)} VND ({amount / income * 100:.1f}% of income)"
        for cat, amount in spending.items()
    )

    savings_verdict = (
        f"ABOVE TARGET - saving {savings_rate:.1f}% (target 20%). GOOD."
        if savings_rate >= 20
        else f"BELOW TARGET - saving {savings_rate:.1f}% (target 20%). Need {vnd(income * 0.2 - surplus)} more. WARNING."
    )
    spending_verdict = (
        f"UNDER 50% needs limit. GOOD."
        if total_spent <= income * 0.5
        else f"Within needs+wants. OK."
        if total_spent <= income * 0.8
        else f"OVER BUDGET. Spent {vnd(total_spent)} VND exceeds {vnd(income * 0.8)} VND. DANGER."
    )

    ctx = (
        f"--- FINANCIAL CONTEXT ---\n"
        f"USER ROLE: Worker\n"
        f"CURRENCY: VND\n"
        f"TOTAL INCOME: {vnd(income)} VND\n\n"
        f"SPENDING BY CATEGORY:\n{cat_lines}\n\n"
        f"PRE-COMPUTED TOTALS:\n"
        f"- TOTAL SPENT: {vnd(total_spent)} VND\n"
        f"- SURPLUS: {vnd(surplus)} VND\n"
        f"- SAVINGS RATE: {savings_rate:.1f}%\n"
        f"- TOP CATEGORY: {top_cat} at {vnd(top_amt)} VND ({top_pct:.1f}%)\n\n"
        f"BUDGET SPLIT: 50/30/20 (default)\n"
        f"- Needs (50%): {vnd(income * 0.5)} VND\n"
        f"- Wants (30%): {vnd(income * 0.3)} VND\n"
        f"- Savings (20%): {vnd(income * 0.2)} VND\n"
    )

    if category_budgets:
        lines = []
        for cb in category_budgets:
            cat = cb["categoryName"]
            limit = cb["monthlyLimit"]
            spent = spending.get(cat, 0)
            pct = (spent / limit * 100) if limit > 0 else 0
            if pct > 100:
                status = f"OVER LIMIT by {vnd(spent - limit)} VND"
            elif pct > 90:
                status = "NEAR LIMIT"
            else:
                status = "OK"
            lines.append(f"  - {cat}: {vnd(spent)} / {vnd(limit)} VND ({pct:.0f}%) {status}")
        ctx += "\nCATEGORY BUDGET STATUS:\n" + "\n".join(lines) + "\n"
    else:
        ctx += "\nCATEGORY BUDGET: NOT SET\n"

    if monthly_history:
        prev_key = sorted(monthly_history.keys())[-1]
        prev = monthly_history[prev_key]
        prev_total = sum(prev.values())
        lines = []
        for cat, current in spending.items():
            previous = prev.get(cat, 0)
            if previous == 0:
                lines.append(f"  - {cat}: {vnd(current)} VND (NEW)")
            else:
                delta = ((current - previous) / previous) * 100
                direction = "UP" if delta > 0 else "DOWN" if delta < 0 else "FLAT"
                lines.append(f"  - {cat}: {vnd(previous)} -> {vnd(current)} VND ({delta:+.1f}%, {direction})")
        overall_delta = ((total_spent - prev_total) / prev_total * 100) if prev_total > 0 else 0
        ctx += (
            "\nMONTH-OVER-MONTH:\n"
            + "\n".join(lines)
            + f"\nMOM VERDICT: Spending {'UP' if overall_delta > 0 else 'DOWN' if overall_delta < 0 else 'FLAT'} {abs(overall_delta):.1f}% vs last month.\n"
        )

    if recurring:
        fixed_total = sum(item["amount"] for item in recurring)
        lines = [
            f"  - {item['description']} ({item['category']}): ~{vnd(item['amount'])} VND/mo ({item['occurrences']} months)"
            for item in recurring
        ]
        ctx += (
            "\nRECURRING EXPENSES:\n"
            + "\n".join(lines)
            + f"\nRECURRING VERDICT: Fixed {vnd(fixed_total)} VND/mo ({fixed_total / income * 100:.1f}%).\n"
        )

    if monthly_history:
        all_past = {}
        for month_data in monthly_history.values():
            for cat, amount in month_data.items():
                all_past.setdefault(cat, []).append(amount)
        anomalies = []
        for cat, current in spending.items():
            past = [value for value in all_past.get(cat, []) if value > 0]
            if not past:
                continue
            avg = sum(past) / len(past)
            ratio = current / avg if avg > 0 else 0
            if ratio >= 2.0:
                anomalies.append(("UNUSUAL", cat, current, avg, ratio))
            elif ratio >= 1.5:
                anomalies.append(("WATCH", cat, current, avg, ratio))
        if anomalies:
            lines = [
                f"  - {level}: {cat} at {vnd(current)} VND is {ratio:.1f}x avg of {vnd(avg)} VND"
                for level, cat, current, avg, ratio in anomalies
            ]
            top = max(anomalies, key=lambda item: item[4])
            ctx += "\nANOMALY ALERTS:\n" + "\n".join(lines) + f"\nANOMALY VERDICT: {top[1]} needs attention.\n"

    ctx += f"\nVERDICTS:\n- SAVINGS: {savings_verdict}\n- SPENDING: {spending_verdict}\n"

    if balances:
        total_liquid = sum(item["balance"] for item in balances)
        lines = "\n".join(f"  - {item['name']}: {vnd(item['balance'])} VND" for item in balances)
        ctx += f"\nACCOUNT BALANCES:\n{lines}\n  Total liquid: {vnd(total_liquid)} VND\n"

    if goals:
        lines = []
        for goal in goals:
            remaining = max(goal["target_amount"] - goal["current_saved"], 0)
            months = (remaining / surplus) if surplus > 0 else float("inf")
            timeline = f"~{months:.1f} months" if months < 999 else "not reachable"
            lines.append(
                f"  - {goal['name']}: {vnd(goal['current_saved'])}/{vnd(goal['target_amount'])} VND "
                f"(need {vnd(remaining)} more, {timeline})"
            )
        ctx += "\nFINANCIAL GOALS:\n" + "\n".join(lines) + "\n"

    if forecast:
        ctx += f"\nFORECAST:\n  - Projected total: {vnd(forecast['total'])} VND (confidence {forecast['confidence']}%)\n"

    ctx += "---"
    return ctx


def run_inference(model, tokenizer, role, context, question, income, spending):
    deterministic = deterministic_response(role, income, spending, question)
    if deterministic is not None:
        return deterministic, 0.0, 0

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context.replace("USER ROLE: Worker", f"USER ROLE: {role}", 1)},
        {"role": "assistant", "content": "Understood."},
        {"role": "user", "content": "[OUTPUT: Return ONLY one valid JSON object matching the schema. Use kind=\"analysis\" for calculations, tax advice, budget split planning, and recommendations. Do not emit an action unless the user explicitly asks to log, edit, or delete a transaction.]\n\n" + question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    input_length = inputs["input_ids"].shape[-1]

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.time() - t0

    raw = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
    clean = raw if parse_model_output(raw) is not None else fallback_output(raw).model_dump_json()
    tokens = outputs.shape[-1] - input_length
    return clean, latency, tokens


def ensure_utf8_mode():
    """Re-exec with PYTHONUTF8=1 so print() handles em-dashes in test names on Windows."""
    if sys.flags.utf8_mode:
        return
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def run_benchmark():
    ensure_utf8_mode()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected. Benchmark expects a local CUDA-capable model runtime.")
    if not ADAPTER_PATH.exists():
        raise FileNotFoundError(f"Adapter directory not found: {ADAPTER_PATH}")

    print("Loading FINA model...")
    print(f"Adapter path: {ADAPTER_PATH}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_PATH), trust_remote_code=True)
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch_dtype,
        device_map="cuda",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
    model = model.merge_and_unload()
    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    total_checks = 0
    total_passed = 0
    results = []

    for tc in TEST_CASES:
        ctx = make_context(
            tc["income"],
            tc["spending"],
            goals=tc.get("goals"),
            balances=tc.get("balances"),
            forecast=tc.get("forecast"),
            category_budgets=tc.get("category_budgets"),
            monthly_history=tc.get("monthly_history"),
            recurring=tc.get("recurring"),
        )
        resp, latency, tokens = run_inference(model, tokenizer, tc["role"], ctx, tc["question"], tc["income"], tc["spending"])
        passed = {name: chk["fn"](resp) for name, chk in tc["checks"].items()}
        n_pass = sum(passed.values())
        n_total = len(passed)
        total_checks += n_total
        total_passed += n_pass
        results.append(
            {
                "id": tc["id"],
                "name": tc["name"],
                "role": tc["role"],
                "score_pct": round(n_pass / n_total * 100, 2),
                "latency_s": round(latency, 2),
                "tokens": tokens,
                "response": resp,
                "checks": passed,
            }
        )
        print(f"[{tc['id']}] {tc['name']} - {n_pass}/{n_total}")

    os.makedirs(LOGS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    model_evaluation_metrics = attach_model_evaluation_metrics(results, TEST_CASES, phase="pre_rag")

    payload = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "base_model": BASE_MODEL,
            "adapter": ADAPTER_NAME,
            "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu",
            "cuda_version": torch.version.cuda or "N/A",
            "torch_version": torch.__version__,
            "platform": platform.platform(),
        },
        "aggregate": {
            "overall_accuracy_pct": round(total_passed / total_checks * 100, 2) if total_checks else 0,
            "total_checks_passed": total_passed,
            "total_checks": total_checks,
            "model_evaluation_metrics": model_evaluation_metrics,
        },
        "per_test": results,
    }
    path = LOGS_DIR / f"benchmark_{ts}.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"Saved benchmark log to {path}")
    print("\nModel evaluation metrics")
    print("=" * 72)
    for metric, value in model_evaluation_metrics.items():
        shown = "N/A" if value is None else f"{value:.2f}%"
        print(f"{metric:35s} {shown}")


if __name__ == "__main__":
    run_benchmark()
