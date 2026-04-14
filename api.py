"""
FINA Financial AI API v3
=========================
FastAPI backend for FINA Brain. JSON-first assistant output.
Inference matches training: same SYSTEM_PROMPT, same context format.
"""

import os
import asyncio
import torch
import json
import logging
from decimal import Decimal
from threading import Thread
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from peft import PeftModel
import uvicorn
from dotenv import load_dotenv
from pathlib import Path

from forecasting.predict import forecast_user
from categorizer.predict import categorize
from rag.retriever import retrieve_context
from rag.store import index_user
from nlp.pipeline import normalize_transaction
import mac_client
from fina_schema import (
    SYSTEM_PROMPT, ModelOutput, parse_model_output, fallback_output,
)

load_dotenv()
logger = logging.getLogger("fina.api")

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_NAME = "financial_qwen_native_v7"

MAC_API_URL = os.getenv("MAC_API_URL", "http://100.109.225.15:4001/api")
FINA_HOST = os.getenv("FINA_HOST", "100.126.232.108")
FINA_PORT = int(os.getenv("FINA_PORT", "8105"))

# ═══════════════════════════════════════════════════════════════════════════════
# APP SETUP
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="FINA Financial AI API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://100.109.225.15:4001",
        "http://100.109.225.15:4002",
        "http://100.109.225.15:4003",
        "http://100.109.225.15:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
tokenizer = None

# ═══════════════════════════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════════════════════════


def _run_startup_benchmark(mdl, tok):
    import time
    messages = [
        {"role": "system", "content": "You are FINA, a financial AI."},
        {"role": "user", "content": "Explain the 50/30/20 rule in 2 sentences."},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        mdl.generate(**inputs, max_new_tokens=1, pad_token_id=tok.eos_token_id)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = mdl.generate(**inputs, max_new_tokens=128, temperature=0.3,
                               do_sample=True, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    out_tokens = outputs.shape[-1] - input_len
    tps = out_tokens / elapsed if elapsed > 0 else 0
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    mem = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    print(f"\n{'='*50}\n  FINA Startup Benchmark\n{'='*50}")
    print(f"  GPU: {gpu} | VRAM: {mem:.2f} GB | {out_tokens} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)")
    print(f"{'='*50}\n")


@app.on_event("startup")
async def load_model():
    global model, tokenizer
    print("Loading FINA Brain (8-bit inference)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_NAME)
    model.eval()
    _run_startup_benchmark(model, tokenizer)
    print(f"FINA online at {FINA_HOST}:{FINA_PORT} | Mac: {MAC_API_URL}")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

VALID_ROLES = {"Student", "Worker", "Freelancer"}


async def resolve_role(user_id: str, provided_role: str | None = None) -> str:
    if provided_role and provided_role in VALID_ROLES:
        return provided_role
    profile = await mac_client.get_user_profile(user_id)
    if profile and profile.get("role") in VALID_ROLES:
        return profile["role"]
    return "Student"


def convert_decimals(obj):
    if isinstance(obj, list):
        return [convert_decimals(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, Decimal):
        return float(obj)
    return obj


def format_vnd(amount):
    try:
        if amount is None:
            return "0"
        return "{:,.0f}".format(float(amount)).replace(",", ".")
    except Exception:
        return str(amount)


def _lstm_available(user_id: str) -> bool:
    return (Path("models") / f"user_{user_id}" / "lstm.pt").exists()


def _get_monthly_forecast(user_id: str, fallback_total: float) -> float:
    try:
        return forecast_user(user_id)["monthly"]["total"]
    except Exception as e:
        print(f"[Forecast fallback]: {e}")
        return fallback_total * 1.05


def _decode_new_tokens(outputs, input_length: int, tok) -> str:
    """Decode only the newly generated tokens (not the prompt)."""
    new_ids = outputs[0][input_length:]
    return tok.decode(new_ids, skip_special_tokens=True).strip()


def _model_output_to_dict(mo: ModelOutput) -> dict:
    """Convert ModelOutput to a plain dict for JSON response."""
    d = mo.model_dump()
    # Clean up None action
    if d.get("action") is None:
        d["action"] = None
    return d


def _extract_json_object(raw: str) -> dict | None:
    """Extract a JSON object from raw model output or return None."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        data = json.loads(text[start:end + 1])
        return data if isinstance(data, dict) else None
    except Exception:
        logger.warning("Dashboard JSON extraction failed: %.200s", raw)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# FINANCIAL CONTEXT BUILDER (matches training format)
# ═══════════════════════════════════════════════════════════════════════════════

async def fetch_raw_data(user_id: str, period: str = "month"):
    data = await mac_client.get_income_and_spending(user_id, period)
    if data:
        print(f"[Data] User {user_id} | Income: {data.get('income')} | "
              f"Spending: {len(data.get('spending', []))}")
    return data


async def format_financial_context(user_id: str, role: str, period: str = "month") -> str:
    """Build FINANCIAL CONTEXT block matching training data format."""
    results = await asyncio.gather(
        fetch_raw_data(user_id, period),
        mac_client.get_financial_goals(user_id),
        mac_client.get_account_balances(user_id),
        mac_client.get_budget_preferences(user_id),
        mac_client.get_category_budgets(user_id),
        mac_client.get_monthly_history(user_id, months=3),
        return_exceptions=True,
    )
    data = results[0] if not isinstance(results[0], Exception) else None
    goals = results[1] if not isinstance(results[1], Exception) else []
    balances = results[2] if not isinstance(results[2], Exception) else []
    budget_prefs = results[3] if not isinstance(results[3], Exception) else None
    category_budgets = results[4] if not isinstance(results[4], Exception) else []
    monthly_history = results[5] if not isinstance(results[5], Exception) else None

    if not data:
        return "Error: Could not fetch financial data."

    cur = data["currency"]
    income = data["income"]
    spending_data = data["spending"]
    computed = data.get("computed", {})

    total_spent_val = computed.get("total_spent", sum(s["spent"] for s in spending_data))
    surplus_val = computed.get("surplus", income - total_spent_val)
    savings_rate = computed.get("savings_rate_pct", (surplus_val / income * 100) if income > 0 else 0)
    top_cat = computed.get("top_category", "N/A")
    top_cat_spent = computed.get("top_category_spent", 0)
    top_cat_pct = computed.get("top_category_pct", 0)

    if budget_prefs:
        n_pct, w_pct, s_pct = budget_prefs["needs_pct"], budget_prefs["wants_pct"], budget_prefs["savings_pct"]
        split_label = f"{n_pct}/{w_pct}/{s_pct} (custom)"
    else:
        n_pct, w_pct, s_pct = 50, 30, 20
        split_label = "50/30/20 (default)"

    needs_limit = income * n_pct / 100
    wants_limit = income * w_pct / 100
    savings_target = income * s_pct / 100

    spending_lines = "\n".join(
        f"  - {s['category_name']}: {format_vnd(s['spent'])} {cur} ({s['spent'] / income * 100:.1f}%)"
        if income > 0 else f"  - {s['category_name']}: {format_vnd(s['spent'])} {cur}"
        for s in spending_data
    ) if spending_data else "  - No expenses recorded."

    # Verdicts
    if savings_rate >= s_pct:
        savings_verdict = f"ABOVE TARGET - saving {savings_rate:.1f}% (target {s_pct}%). GOOD."
    else:
        shortfall = savings_target - surplus_val
        savings_verdict = f"BELOW TARGET - saving {savings_rate:.1f}% (target {s_pct}%). Need {format_vnd(shortfall)} {cur} more. WARNING."

    if total_spent_val <= needs_limit:
        spending_verdict = f"UNDER {n_pct}% needs limit. GOOD."
    elif total_spent_val <= needs_limit + wants_limit:
        spending_verdict = f"Within needs+wants. OK."
    else:
        spending_verdict = f"OVER BUDGET. Spent {format_vnd(total_spent_val)} {cur} exceeds {format_vnd(needs_limit + wants_limit)} {cur}. DANGER."

    ctx = (
        f"--- FINANCIAL CONTEXT ---\n"
        f"USER ROLE: {role}\n"
        f"CURRENCY: {cur}\n"
        f"TOTAL INCOME: {format_vnd(income)} {cur}\n\n"
        f"SPENDING BY CATEGORY:\n{spending_lines}\n\n"
        f"PRE-COMPUTED TOTALS:\n"
        f"- TOTAL SPENT: {format_vnd(total_spent_val)} {cur}\n"
        f"- SURPLUS: {format_vnd(surplus_val)} {cur}\n"
        f"- SAVINGS RATE: {savings_rate:.1f}%\n"
        f"- TOP CATEGORY: {top_cat} at {format_vnd(top_cat_spent)} {cur} ({top_cat_pct:.1f}%)\n\n"
        f"BUDGET SPLIT: {split_label}\n"
        f"- Needs ({n_pct}%): {format_vnd(needs_limit)} {cur}\n"
        f"- Wants ({w_pct}%): {format_vnd(wants_limit)} {cur}\n"
        f"- Savings ({s_pct}%): {format_vnd(savings_target)} {cur}\n"
    )

    # Category budgets
    if category_budgets:
        spending_lookup = {s["category_name"]: s["spent"] for s in spending_data}
        cb_lines = []
        over_details, near_details = [], []
        for cb in category_budgets:
            cat_name = cb.get("categoryName", "Unknown")
            limit = cb.get("monthlyLimit", 0)
            spent = spending_lookup.get(cat_name, 0)
            pct_used = (spent / limit * 100) if limit > 0 else 0
            if pct_used > 100:
                status = f"OVER LIMIT by {format_vnd(spent - limit)} {cur}"
                over_details.append(cat_name)
            elif pct_used > 90:
                status = "NEAR LIMIT"
                near_details.append(cat_name)
            else:
                status = "OK"
            cb_lines.append(f"  - {cat_name}: {format_vnd(spent)} / {format_vnd(limit)} {cur} ({pct_used:.0f}%) {status}")
        verdict_parts = []
        if over_details:
            verdict_parts.append(f"{len(over_details)} OVER ({', '.join(over_details)})")
        if near_details:
            verdict_parts.append(f"{len(near_details)} NEAR ({', '.join(near_details)})")
        ctx += "\nCATEGORY BUDGET STATUS:\n" + "\n".join(cb_lines) + f"\nBUDGET VERDICT: {', '.join(verdict_parts) if verdict_parts else 'All OK'}\n"
    else:
        ctx += "\nCATEGORY BUDGET: NOT SET\n"

    # Month-over-month
    if monthly_history and monthly_history.get("months"):
        months_data = monthly_history["months"]
        prev_key = sorted(months_data.keys())[-1]
        prev = months_data[prev_key]
        prev_total = sum(prev.values())
        current_cats = {s["category_name"]: s["spent"] for s in spending_data}
        mom_lines = []
        for cat, curr_amt in current_cats.items():
            prev_amt = prev.get(cat, 0)
            if prev_amt == 0:
                mom_lines.append(f"  - {cat}: {format_vnd(curr_amt)} {cur} (NEW)")
            else:
                delta = ((curr_amt - prev_amt) / prev_amt) * 100
                d = "UP" if delta > 0 else "DOWN" if delta < 0 else "FLAT"
                mom_lines.append(f"  - {cat}: {format_vnd(prev_amt)} -> {format_vnd(curr_amt)} {cur} ({delta:+.1f}%, {d})")
        overall_delta = ((total_spent_val - prev_total) / prev_total * 100) if prev_total > 0 else 0
        overall_d = "UP" if overall_delta > 0 else "DOWN" if overall_delta < 0 else "FLAT"
        ctx += f"\nMONTH-OVER-MONTH:\n" + "\n".join(mom_lines) + f"\nMOM VERDICT: Spending {overall_d} {abs(overall_delta):.1f}% vs last month.\n"

    # Recurring
    if monthly_history and monthly_history.get("recurring"):
        rec_list = monthly_history["recurring"]
        rec_lines = []
        total_fixed = 0
        for rec in rec_list:
            rec_lines.append(f"  - {rec['description']} ({rec['category']}): ~{format_vnd(rec['amount'])} {cur}/mo ({rec['occurrences']} months)")
            total_fixed += rec["amount"]
        discretionary = max(total_spent_val - total_fixed, 0)
        fixed_pct = (total_fixed / income * 100) if income > 0 else 0
        ctx += f"\nRECURRING EXPENSES:\n" + "\n".join(rec_lines)
        ctx += f"\nRECURRING VERDICT: Fixed {format_vnd(total_fixed)} {cur}/mo ({fixed_pct:.1f}%). Discretionary: {format_vnd(discretionary)} {cur}.\n"

    # Anomalies
    if monthly_history and monthly_history.get("months"):
        months_data = monthly_history["months"]
        current_cats = {s["category_name"]: s["spent"] for s in spending_data}
        all_past = {}
        for month_data in months_data.values():
            for cat, amt in month_data.items():
                all_past.setdefault(cat, []).append(amt)
        anomalies = []
        for cat, curr_amt in current_cats.items():
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
            anom_lines = [f"  - {lvl}: {cat} at {format_vnd(cur_a)} {cur} is {r:.1f}x avg of {format_vnd(avg_a)} {cur}"
                          for lvl, cat, cur_a, avg_a, r in anomalies]
            top_a = max(anomalies, key=lambda a: a[4])
            ctx += f"\nANOMALY ALERTS:\n" + "\n".join(anom_lines) + f"\nANOMALY VERDICT: {top_a[1]} needs attention.\n"

    # Verdicts
    ctx += f"\nVERDICTS:\n- SAVINGS: {savings_verdict}\n- SPENDING: {spending_verdict}\n"

    # Balances
    if balances:
        total_liquid = sum(a.get("balance", 0) for a in balances)
        bal_lines = "\n".join(f"  - {a.get('name', 'Account')}: {format_vnd(a.get('balance', 0))} {cur}" for a in balances)
        ctx += f"\nACCOUNT BALANCES:\n{bal_lines}\n  Total liquid: {format_vnd(total_liquid)} {cur}\n"

    # Goals
    if goals:
        goal_lines = []
        for g in goals:
            target = g.get("target_amount", 0)
            saved = g.get("current_saved", 0)
            remaining = max(target - saved, 0)
            pct_done = (saved / target * 100) if target > 0 else 0
            m_needed = (remaining / surplus_val) if surplus_val > 0 else float("inf")
            m_str = f"~{m_needed:.1f} months" if m_needed < 999 else "not reachable"
            goal_lines.append(f"  - {g.get('name', 'Goal')}: {format_vnd(saved)}/{format_vnd(target)} {cur} ({pct_done:.0f}% done, {m_str})")
        ctx += "\nFINANCIAL GOALS:\n" + "\n".join(goal_lines) + "\n"

    # LSTM forecast
    if _lstm_available(user_id):
        try:
            fc = forecast_user(user_id)
            fc_total = fc.get("monthly", {}).get("total", 0)
            fc_conf = fc.get("confidence", 60)
            ctx += f"\nFORECAST: Projected {format_vnd(fc_total)} {cur} next month (confidence {fc_conf}%)\n"
        except Exception as e:
            print(f"[Forecast context] Error: {e}")

    ctx += "---"
    return ctx


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD (separate from main chat schema)
# ═══════════════════════════════════════════════════════════════════════════════

async def generate_dashboard_json(user_id: str, role: str = "Student", period: str = "month"):
    data = await fetch_raw_data(user_id, period)
    if not data:
        return None

    cur = data["currency"]
    total_spent_val = sum(item["spent"] for item in data["spending"])
    savings_val = data["income"] - total_spent_val
    spend_ratio = (total_spent_val / data["income"] * 100) if data["income"] > 0 else 0

    sorted_spending = sorted(data["spending"], key=lambda x: x["spent"], reverse=True)
    highest_cat = sorted_spending[0]["category_name"] if sorted_spending else "General"
    highest_amt = sorted_spending[0]["spent"] if sorted_spending else 0

    # Try AI generation
    prompt = f"""Analyze this data and return JSON only.
Role: {role} | Income: {format_vnd(data['income'])} {cur} | Spent: {format_vnd(total_spent_val)} {cur} | Savings: {format_vnd(savings_val)} {cur}
Top: {highest_cat} {format_vnd(highest_amt)} {cur}
NUMBER FORMAT: Vietnamese style with dots (1.500.000 VND). Currency always VND.
Return: {{"initial_message": "...", "summary_cards": [...], "smart_insights": [...]}}"""

    messages = [
        {"role": "system", "content": "You are FINA. Output valid JSON only. No markdown."},
        {"role": "user", "content": prompt},
    ]

    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        input_length = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=600, temperature=0.2,
                                     do_sample=True, pad_token_id=tokenizer.eos_token_id)
        raw = _decode_new_tokens(outputs, input_length, tokenizer)
        result = _extract_json_object(raw)
        if result and "smart_insights" in result:
            result["prediction"] = {
                "amount": _get_monthly_forecast(user_id, total_spent_val),
                "confidence": 80 if _lstm_available(user_id) else 60,
                "label": "Projected spending next month",
            }
            return result
    except Exception as e:
        print(f"[Dashboard AI failed - fallback]: {e}")

    # Fallback
    savings_ratio = (savings_val / data["income"] * 100) if data["income"] > 0 else 0
    budget_health = "success" if spend_ratio < 80 else "warning" if spend_ratio < 100 else "danger"
    savings_type = "success" if savings_ratio >= 20 else "warning"

    return {
        "initial_message": f"Welcome back! Net savings: {format_vnd(savings_val)} {cur}.",
        "summary_cards": [
            {"id": 1, "type": "info", "title": "Total Income", "subtitle": f"{format_vnd(data['income'])} {cur}", "badge": "Verified"},
            {"id": 2, "type": "danger", "title": "Total Spent", "subtitle": f"{format_vnd(total_spent_val)} {cur}", "badge": "Tracked"},
            {"id": 3, "type": "success", "title": "Net Savings", "subtitle": f"{format_vnd(savings_val)} {cur}", "badge": "Cash Flow"},
        ],
        "smart_insights": [
            {"id": 1, "type": "warning" if highest_amt > data["income"] * 0.30 else "info",
             "title": "Top Expense", "desc": f"{highest_cat} at {format_vnd(highest_amt)} {cur} ({highest_amt / data['income'] * 100:.1f}% of income)."},
            {"id": 2, "type": budget_health, "title": "Budget Health",
             "desc": f"Spent {format_vnd(total_spent_val)} {cur} ({spend_ratio:.1f}%)."},
            {"id": 3, "type": savings_type, "title": "Savings Rate",
             "desc": f"Saved {format_vnd(savings_val)} {cur} ({savings_ratio:.1f}%). " +
                     ("Above 20% target." if savings_ratio >= 20 else f"Target: {format_vnd(data['income'] * 0.20)} {cur}.")},
        ],
        "prediction": {
            "amount": _get_monthly_forecast(user_id, total_spent_val),
            "confidence": 80 if _lstm_available(user_id) else 60,
            "label": "Projected spending next month",
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class CamelModel(BaseModel):
    model_config = {"populate_by_name": True}


class HistoryMessage(BaseModel):
    role: str
    content: str


class UserRequest(CamelModel):
    user_id: str = Field(alias="userId")
    role: str | None = None
    mode: str = "Standard"
    period: str = "month"
    message: str
    history: list[HistoryMessage] = Field(default_factory=list)


class CategorizeRequest(CamelModel):
    description: str
    transaction_id: str | None = Field(default=None, alias="transactionId")


class NormalizeRequest(BaseModel):
    description: str


class BatchCategorizeRequest(BaseModel):
    items: list[CategorizeRequest]


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS — Health & Status
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def health_check():
    return {"status": "online", "model": BASE_MODEL, "version": "3.0.0"}


@app.get("/status")
async def detailed_status():
    return {
        "status": "online",
        "model": BASE_MODEL,
        "adapter": ADAPTER_NAME,
        "version": "3.0.0",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "output_format": "json",
        "endpoints": [
            "GET  /                      — health check",
            "GET  /status                — detailed status",
            "GET  /dashboard/{user_id}   — AI dashboard JSON",
            "GET  /history/{user_id}     — transaction history",
            "POST /chat                  — main chat (JSON output)",
            "POST /chat/stream           — streaming chat (SSE)",
            "POST /categorize            — categorize transaction",
            "POST /categorize/batch      — batch categorize",
            "POST /nlp/normalize         — normalize text",
            "POST /rag/index/{user_id}   — re-index RAG",
            "POST /train/lstm/{user_id}  — LSTM training",
        ],
        "mac_api": MAC_API_URL,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS — AI Features
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/dashboard/{user_id}")
async def get_dashboard_summary(user_id: str, role: str | None = None, period: str = "month"):
    role = await resolve_role(user_id, role)
    print(f"[Dashboard] User {user_id} | Role: {role} | Period: {period}")
    result = await generate_dashboard_json(user_id, role, period)
    if result is None:
        raise HTTPException(status_code=502, detail="Could not fetch user data")
    return result


@app.get("/history/{user_id}")
async def get_history(user_id: str):
    data = await fetch_raw_data(user_id)
    return {"history": data["history"] if data else []}


@app.post("/chat")
async def chat_endpoint(request: UserRequest):
    global model, tokenizer
    role = await resolve_role(request.user_id, request.role)
    print(f"\n[CHAT] User: {request.user_id} | Role: {role} | Q: {request.message}")

    financial_context = await format_financial_context(request.user_id, role, request.period)
    rag_context = retrieve_context(request.user_id, request.message)

    context_block = financial_context
    if rag_context:
        context_block += f"\n{rag_context}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context_block},
        {"role": "assistant", "content": "Understood."},
    ]

    for turn in request.history:
        if turn.role in ("user", "assistant"):
            messages.append({"role": turn.role, "content": turn.content})

    messages.append({"role": "user", "content": request.message})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    input_length = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    raw_output = _decode_new_tokens(outputs, input_length, tokenizer)
    print(f"[CHAT] Raw output: {raw_output[:200]}")

    # Parse JSON output
    parsed = parse_model_output(raw_output)
    if parsed is None:
        logger.warning("Model output parse failed, using fallback")
        parsed = fallback_output(raw_output)

    result = _model_output_to_dict(parsed)

    # Backward compatibility for frontend migration
    response = {
        "model_output": result,
        "response": result["message"],
        "actions": [result["action"]] if result.get("action") else [],
    }
    return response


@app.post("/chat/stream")
async def chat_stream_endpoint(request: UserRequest):
    global model, tokenizer
    role = await resolve_role(request.user_id, request.role)
    print(f"\n[STREAM] User: {request.user_id} | Role: {role} | Q: {request.message}")

    financial_context = await format_financial_context(request.user_id, role, request.period)
    rag_context = retrieve_context(request.user_id, request.message)

    context_block = financial_context
    if rag_context:
        context_block += f"\n{rag_context}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context_block},
        {"role": "assistant", "content": "Understood."},
    ]

    for turn in request.history:
        if turn.role in ("user", "assistant"):
            messages.append({"role": turn.role, "content": turn.content})

    messages.append({"role": "user", "content": request.message})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = {
        **inputs,
        "max_new_tokens": 1024,
        "temperature": 0.3,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }

    thread = Thread(target=lambda: model.generate(**gen_kwargs))
    thread.start()

    async def token_generator():
        full_text = ""
        for token in streamer:
            full_text += token
            yield f"data: {json.dumps({'token': token})}\n\n"

        # Parse complete response as JSON
        parsed = parse_model_output(full_text)
        if parsed is None:
            parsed = fallback_output(full_text)

        result = _model_output_to_dict(parsed)
        yield f"data: {json.dumps({'final': result})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS — ML Services
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/categorize")
async def categorize_transaction(request: CategorizeRequest):
    print(f"[Categorize] '{request.description}'")
    result = categorize(request.description)
    if request.transaction_id is not None:
        await mac_client.notify_categorized(request.transaction_id, result["category"], result["confidence"])
    return result


@app.post("/categorize/batch")
async def batch_categorize(request: BatchCategorizeRequest):
    print(f"[Categorize/Batch] {len(request.items)} items")
    results = []
    for item in request.items:
        result = categorize(item.description)
        if item.transaction_id is not None:
            await mac_client.notify_categorized(item.transaction_id, result["category"], result["confidence"])
        results.append({**result, "transaction_id": item.transaction_id})
    return {"results": results}


@app.post("/nlp/normalize")
async def normalize_text(request: NormalizeRequest):
    print(f"[NLP] '{request.description}'")
    return normalize_transaction(request.description)


@app.post("/rag/index/{user_id}")
async def rag_index(user_id: str):
    print(f"[RAG] Indexing for User {user_id}")
    count = index_user(user_id)
    return {"indexed": count, "user_id": user_id}


@app.post("/train/lstm/{user_id}")
async def trigger_lstm_training(user_id: str, epochs: int = 50):
    print(f"[LSTM Train] User {user_id} | Epochs: {epochs}")
    try:
        from forecasting.train_lstm import train_for_user
        from forecasting.data import SEQ_LEN
        result = train_for_user(user_id, epochs=epochs, seq_len=SEQ_LEN)
        forecast = forecast_user(user_id)
        return {"status": "trained", "user_id": user_id, "forecast": forecast}
    except Exception as e:
        print(f"[LSTM Train Error]: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=FINA_PORT)
