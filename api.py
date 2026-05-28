"""
FINA Financial AI API v3
=========================
FastAPI backend for FINA Brain. JSON-first assistant output.
Inference matches training: same SYSTEM_PROMPT, same context format.
"""

import os
import re
import sys
import asyncio
import calendar
import torch
import json
import logging
from datetime import date
from decimal import Decimal
from threading import Thread

# Windows default stdout is cp1252; the financial-context debug print contains
# unicode arrows (→) that crash the request handler. Force utf-8 on stdout/stderr.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import transformers.utils.import_utils as _tf_import_utils

# Text-only model runtime. Some local environments have a mismatched torchvision
# build, and transformers may import it indirectly through image utilities.
_tf_import_utils.is_torchvision_available = lambda: False
_tf_import_utils.is_torchvision_v2_available = lambda: False

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from peft import PeftModel
import uvicorn
from dotenv import load_dotenv
from pathlib import Path

from forecasting.predict import forecast_user
from categorizer.predict import categorize
from nlp.pipeline import normalize_transaction
from rag.retriever import retrieve_context_with_sources, RetrievalResult
import mac_client
from fina_schema import (
    SYSTEM_PROMPT, ModelOutput, parse_model_output,
    parse_model_output_with_status, fallback_output, validate_action_safety,
)
from period_utils import PeriodSpec, parse_period, infer_period_from_text

load_dotenv()

# Ensure WARNING-level events from "fina" / "fina.api" reach the console
# (e.g. signal normalization, fallback parses, unsafe-action rejections) even
# when uvicorn's default config would swallow them. Idempotent.
def _configure_fina_logging() -> None:
    for name in ("fina", "fina.api"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.WARNING)
        if not any(getattr(h, "_fina_handler", False) for h in lg.handlers):
            handler = logging.StreamHandler()
            handler.setLevel(logging.WARNING)
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            ))
            handler._fina_handler = True  # type: ignore[attr-defined]
            lg.addHandler(handler)
        lg.propagate = False

_configure_fina_logging()
logger = logging.getLogger("fina.api")

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ACTIVE_ADAPTER = "v8"
ACTIVE_ADAPTER_PATH = "financial_qwen_native_v8"

MAC_API_URL = os.getenv("MAC_API_URL", "http://localhost:4001/api")
FINA_HOST = os.getenv("FINA_HOST", "0.0.0.0")
FINA_PORT = int(os.getenv("FINA_PORT", "8105"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))

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
    print("Loading FINA Brain (fixed v8, 8-bit inference)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, ACTIVE_ADAPTER_PATH, adapter_name=ACTIVE_ADAPTER)
    try:
        model.set_adapter(ACTIVE_ADAPTER)
    except Exception as _e:
        print(f"[ADAPTER] set_adapter({ACTIVE_ADAPTER}) failed: {_e}")
    model.eval()
    _run_startup_benchmark(model, tokenizer)
    try:
        from rag import store as _rag_store
        _rag_store._client_singleton()
        print("[RAG] Chroma client warmed.")
    except Exception as _e:
        print(f"[RAG] Chroma warmup skipped: {_e}")
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
    spec = parse_period(period)
    # Pull at least 3 months of history so MoM still works for short windows;
    # for longer windows (e.g. 6m), pull spec.months_span + 1 to expose the
    # month preceding the requested window.
    history_months = max(3, spec.months_span + 1)
    results = await asyncio.gather(
        fetch_raw_data(user_id, period),
        mac_client.get_financial_goals(user_id),
        mac_client.get_account_balances(user_id),
        mac_client.get_budget_preferences(user_id),
        mac_client.get_monthly_history(user_id, months=history_months),
        return_exceptions=True,
    )
    data = results[0] if not isinstance(results[0], Exception) else None
    goals = results[1] if not isinstance(results[1], Exception) else []
    balances = results[2] if not isinstance(results[2], Exception) else []
    budget_prefs = results[3] if not isinstance(results[3], Exception) else None
    monthly_history = results[4] if not isinstance(results[4], Exception) else None

    if not data:
        return "Error: Could not fetch financial data."

    cur = data["currency"]
    income = data["income"]
    spending_data = data["spending"]
    computed = data.get("computed", {})
    budget_limits = computed.get("category_budget_status") or computed.get("budget_status") or []

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

    period_label = spec.label  # e.g. "May 2026 (current month)", "April 2026 (last month)", "last 3 months"
    ctx = (
        f"--- FINANCIAL CONTEXT ---\n"
        f"USER ROLE: {role}\n"
        f"CURRENCY: {cur}\n"
        f"PERIOD: {period_label}\n"
        f"TOTAL INCOME ({period_label}): {format_vnd(income)} {cur}\n\n"
        f"SPENDING BY CATEGORY ({period_label}):\n{spending_lines}\n\n"
        f"PRE-COMPUTED TOTALS ({period_label}):\n"
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
    if budget_limits:
        today = date.today()
        # Pace-factor projection is meaningful only when the requested window is
        # the *current* calendar month and we are still inside it. For rolling
        # multi-month windows or year-to-date the spend total is already a
        # historical fact — no extrapolation.
        if spec.is_calendar_month and spec.start.year == today.year and spec.start.month == today.month:
            days_in_month = calendar.monthrange(today.year, today.month)[1]
            day_of_month = today.day
            pace_factor = days_in_month / day_of_month if day_of_month > 0 else 1.0
            project_label = "month-end"
        else:
            pace_factor = 1.0
            project_label = "window total"

        spending_lookup = {s["category_name"]: s["spent"] for s in spending_data}
        cb_lines = []
        over_details, near_details = [], []
        projected_over_total = 0.0
        for cb in budget_limits:
            cat_name = cb.get("categoryName", "Unknown")
            limit = cb.get("monthlyLimit", cb.get("limit", 0))
            spent = spending_lookup.get(cat_name, 0)
            pct_used = (spent / limit * 100) if limit > 0 else 0
            projected = spent * pace_factor
            projected_diff = projected - limit
            if projected_diff > 0:
                projection_note = f"projected {project_label} {format_vnd(projected)} {cur} → OVER by {format_vnd(projected_diff)} {cur}"
            else:
                projection_note = f"projected {project_label} {format_vnd(projected)} {cur} → under by {format_vnd(-projected_diff)} {cur}"
            if pct_used > 100:
                status = f"OVER LIMIT by {format_vnd(spent - limit)} {cur}"
                over_details.append(cat_name)
            elif pct_used > 90:
                status = "NEAR LIMIT"
                near_details.append(cat_name)
            else:
                status = "OK"
            if projected_diff > 0:
                projected_over_total += projected_diff
            cb_lines.append(
                f"  - {cat_name}: {format_vnd(spent)} / {format_vnd(limit)} {cur} ({pct_used:.0f}%) {status}; {projection_note}"
            )
        verdict_parts = []
        if over_details:
            verdict_parts.append(f"{len(over_details)} OVER ({', '.join(over_details)})")
        if near_details:
            verdict_parts.append(f"{len(near_details)} NEAR ({', '.join(near_details)})")
        if spec.is_calendar_month and spec.start.year == today.year and spec.start.month == today.month:
            header_detail = f"today={today.isoformat()}, day {day_of_month}/{days_in_month}"
            overage_label = "PROJECTED TOTAL OVERAGE BY MONTH-END"
        else:
            header_detail = f"period={spec.label} ({spec.start.isoformat()}..{spec.end.isoformat()})"
            overage_label = "TOTAL OVERAGE OVER WINDOW"
        ctx += (
            f"\nCATEGORY BUDGET STATUS ({header_detail}):\n"
            + "\n".join(cb_lines)
            + f"\nBUDGET VERDICT: {', '.join(verdict_parts) if verdict_parts else 'All OK'}"
            + f"\n{overage_label}: {format_vnd(projected_over_total)} {cur}\n"
        )
    else:
        ctx += "\nCATEGORY BUDGET: NOT SET\n"

    # Month-over-month — only meaningful when the requested window is a single
    # calendar month. For rolling Nm / year / all, the "previous month" in
    # monthly_history overlaps the requested window, producing double-counted
    # diffs (e.g. 3.595.000 -> 3.595.000 FLAT). Skip in that case.
    today_now = date.today()
    is_current_month = (
        spec.is_calendar_month
        and spec.start.year == today_now.year
        and spec.start.month == today_now.month
    )
    is_prev_month = spec.raw == "prev_month"
    if not (is_current_month or is_prev_month):
        ctx += (
            "\nMONTH-OVER-MONTH: not applicable for rolling-window queries "
            f"(period={spec.label}).\n"
        )
    elif monthly_history and monthly_history.get("months"):
        months_data = monthly_history["months"]
        sorted_keys = sorted(months_data.keys())
        # For current-month requests: the latest history slice IS the previous
        # calendar month. For prev_month requests: the latest history slice is
        # the prev_month itself, so we need the one before that.
        if is_prev_month and len(sorted_keys) >= 2:
            prev_key = sorted_keys[-2]
        else:
            prev_key = sorted_keys[-1] if sorted_keys else None
        if prev_key is None:
            ctx += "\nMONTH-OVER-MONTH: no prior month available for comparison.\n"
        else:
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
            ctx += (
                f"\nMONTH-OVER-MONTH (vs {prev_key}):\n" + "\n".join(mom_lines)
                + f"\nMOM VERDICT: Spending {overall_d} {abs(overall_delta):.1f}% vs {prev_key}.\n"
            )

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


async def build_hybrid_context(user_id: str, role: str, message: str, period: str = "month"):
    """Build structured financial context plus source-cited retrieved evidence."""
    print(
        f"[HYBRID] stage=context.start user_id={user_id} role={role} "
        f"period={period} query={message!r}"
    )
    period_spec = parse_period(period)
    financial_context, rag_result = await asyncio.gather(
        format_financial_context(user_id, role, period),
        retrieve_context_with_sources(
            user_id, message, top_k=RAG_TOP_K, period=period_spec
        ),
    )

    if rag_result.context:
        hybrid_context = financial_context + "\n\n" + rag_result.context
        print(
            f"[HYBRID] stage=context.attach user_id={user_id} "
            f"structured_chars={len(financial_context)} rag_chars={len(rag_result.context)}"
        )
    else:
        hybrid_context = financial_context
        print(
            f"[HYBRID] stage=context.no_rag user_id={user_id} "
            f"structured_chars={len(financial_context)}"
        )

    print(
        f"[HYBRID] stage=context.done user_id={user_id} "
        f"rag_status={rag_result.status} sources={len(rag_result.sources)} "
        f"total_chars={len(hybrid_context)}"
    )
    if rag_result.error:
        print(f"[RAG] Error: {rag_result.error}")

    return hybrid_context, rag_result


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

    # Deterministic dashboard response.
    # The AI-generated dashboard path proved too slow and unstable for the finance proxy.
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
    # Kept for compatibility, but the backend is fixed to v8.
    adapter: str | None = None
    use_rag: bool = True


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
        "adapter": ACTIVE_ADAPTER,
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
            "POST /train/lstm/{user_id}  — LSTM training",
        ],
        "mac_api": MAC_API_URL,
        "rag": {"enabled": True, "top_k": RAG_TOP_K},
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


@app.get("/rag/context/{user_id}")
async def get_rag_context(user_id: str, query: str = "financial overview"):
    """Diagnostic endpoint for checking AWAD2 -> FINA RAG connectivity."""
    role = await resolve_role(user_id, None)
    context, rag_result = await build_hybrid_context(user_id, role, query)
    return {
        "user_id": user_id,
        "role": role,
        "rag": rag_result.as_dict(),
        "context_preview": context[:4000],
    }


# ── Per-turn style steering ────────────────────────────────────────────
# The LoRA was trained to be terse. For analysis / overview / advice-style
# questions we prepend a hidden hint to the user turn so the model fills
# the "message" field with a proper multi-point answer even on a single turn.
_ANALYSIS_PATTERNS = re.compile(
    r"\b(analyze|analysis|overview|summary|summarize|habit|habits|pattern|patterns|"
    r"how (am|are) (i|we) doing|how is my|trend|trends|advice|recommend|"
    r"what (can|should) (i|we) do|where (is|does) my money|"
    r"review|breakdown|insight|insights|on track|save|saving|savings|cuts?|"
    r"how much|how many|what(?:'s| is) my|tell me about|show me|"
    r"spend|spent|spending|earned|earnings|"
    r"compare|compared|vs|versus|"
    r"category|categories|top)\b",
    re.IGNORECASE,
)
_BUDGET_PATTERNS = re.compile(
    r"\b(budget|budgets|set (a )?limit|create (a )?budget|plan (my|a) spend)\b",
    re.IGNORECASE,
)
_CALC_AUDIT_PATTERNS = re.compile(
    r"\b("
    r"how good.*calculat|how accurate.*calculat|"
    r"calculation.*accurate|calculations?.*good|"
    r"how would i know|how do i know|can i trust|"
    r"verify.*calculat|check.*calculat|audit.*calculat"
    r")\b",
    re.IGNORECASE,
)
_ASSISTANT_IDENTITY_PATTERNS = re.compile(
    r"\b("
    r"what(?:'s| is) your name|"
    r"who are you|"
    r"what should i call you|"
    r"introduce yourself"
    r")\b",
    re.IGNORECASE,
)
# Questions FINA must refuse: market/price predictions, buy/sell tips, and
# requests to move money on the user's behalf. These get a deterministic safe
# decline (no LLM call) so a 3B model can never wander into a market opinion.
_OUT_OF_SCOPE_PATTERNS = re.compile(
    r"(?:"
    # market / price predictions
    r"(?:will|should|is|are|going to|gonna|do you think).{0,40}\b(bitcoin|btc|crypto|"
    r"coin|token|ethereum|eth|stock|stocks|shares?|market|gold|sjc|usd|forex|fund)\b"
    r".{0,30}\b(go up|going up|rise|rising|crash|drop|fall|moon|pump|dump|worth|predict|"
    r"forecast|outlook|good buy|good investment|invest in)\b"
    r"|"
    # explicit buy/sell tips
    r"\b(stock|crypto|coin|share|investment)\s+tips?\b"
    r"|"
    r"\bwhich\s+(stock|coin|crypto|share|fund)s?\s+should i (buy|invest)\b"
    r"|"
    r"\b(hot|best)\s+(stock|coin|crypto)\b"
    r"|"
    # requests to move money on the user's behalf
    r"\b(transfer|send|wire|move|pay|lend|loan)\b.{0,40}(?:money|cash|vnd|đ|\d)"
    r".{0,20}\b(for me|on my behalf|to my friend|to him|to her|to them|for my)\b"
    r")",
    re.IGNORECASE,
)


# Conceptual / advisory questions that FINANCIAL CONTEXT cannot answer directly.
# Used to nudge the model into a natural, general-guidance style (not a spending
# diagnosis). Only applied when the turn has NO personal-data anchor below.
_ADVICE_PATTERNS = re.compile(
    r"\b("
    r"what(?:'s| is| are)\s+(a|an|the)?\s*(compound interest|interest rate|etf|"
    r"mutual fund|index fund|credit score|credit card|cic|emergency fund|inflation|"
    r"diversif\w*|dollar.?cost|bhxh|insurance)|"
    r"explain|difference between|what does .* mean|how does .* work|"
    r"should i (rent|buy|invest|quit|freelance|switch|get|take|start|consider)|"
    r"rent or buy|buy or rent|is it worth|worth it|"
    r"how (do|can|should) i (start|stop|build|stay|avoid|deal|handle|begin|get out)|"
    r"how to (start|stop|build|save|invest|budget|get out)|"
    r"impulse|tips? (for|on|to)|advice (on|for|about)|"
    r"do i need .* insurance|need insurance|"
    r"side (job|hustle|income)|raise my rate|negotiate"
    r")\b",
    re.IGNORECASE,
)
# Anchors that mean "use MY tracked data" — these keep the turn in the normal
# grounded analysis style even if an advice keyword also matched.
_PERSONAL_DATA_PATTERNS = re.compile(
    r"\b("
    r"my (spending|budget|budgets|income|salary|savings?|surplus|expenses?|"
    r"transactions?|finances?|money|categor|balance|goal)|"
    r"am i (over|under|on track|saving|spending|doing|within)|"
    r"how (am|is|are) (i|my)|"
    r"this month|last month|this week|so far|"
    r"how much (did|have|do) i|what(?:'s| is) my|"
    r"do i have|where (is|does) my money|i (spent|spend|paid|earned|bought|owe)"
    r")\b",
    re.IGNORECASE,
)


def _is_advice_question(message: str) -> bool:
    return bool(
        _ADVICE_PATTERNS.search(message)
        and not _PERSONAL_DATA_PATTERNS.search(message)
    )


def _context_value(context: str, label: str) -> str | None:
    match = re.search(rf"^- {re.escape(label)}:\s*(.+)$", context, re.MULTILINE)
    return match.group(1).strip() if match else None


def _build_calculation_audit_response(financial_context: str, rag_result) -> dict:
    income_match = re.search(r"^TOTAL INCOME:\s*(.+)$", financial_context, re.MULTILINE)
    income = income_match.group(1).strip() if income_match else "the income shown in AWAD2"
    total_spent = _context_value(financial_context, "TOTAL SPENT") or "the summed expenses"
    surplus = _context_value(financial_context, "SURPLUS") or "income minus total spent"
    savings_rate = _context_value(financial_context, "SAVINGS RATE") or "surplus divided by income"
    top_category = _context_value(financial_context, "TOP CATEGORY") or "the largest spending category"

    citation_note = ""
    if rag_result.sources:
        first_ids = ", ".join(f"[{source.id}]" for source in rag_result.sources[:3])
        citation_note = f" Retrieved evidence is attached as source IDs such as {first_ids}, so you can inspect which transactions were used."

    message = (
        "My calculations are strongest for deterministic totals because they come from AWAD2's structured finance data, not from free-form guessing. "
        f"For this context, AWAD2 reports income as {income}, total spent as {total_spent}, surplus as {surplus}, savings rate as {savings_rate}, and top category as {top_category}. "
        "You can verify them by checking the terminal logs: the API fetches transactions, builds pre-computed totals, then RAG lists selected sources as [S1], [S2], etc. "
        "The formula is simple: total spent is the sum of expense transactions, surplus is income minus total spent, and savings rate is surplus divided by income. "
        f"If any answer mentions a number that is not in FINANCIAL CONTEXT or the retrieved sources, treat it as untrusted.{citation_note}"
    )

    result = {
        "kind": "analysis",
        "message": message,
        "action": None,
        "signals": [],
        "needs_clarification": False,
    }
    return {
        "model_output": result,
        "response": result["message"],
        "actions": [],
        "rag": rag_result.as_dict(),
        "retrieved_sources": [
            {"id": source.id, "text": source.text, "metadata": source.metadata}
            for source in rag_result.sources
        ],
    }


def _build_assistant_identity_response() -> dict:
    result = {
        "kind": "analysis",
        "message": "My name is FINA. I am your personal finance assistant for tracking spending, checking budgets, calculating surplus, and explaining your financial context.",
        "action": None,
        "signals": [],
        "needs_clarification": False,
    }
    return {
        "model_output": result,
        "response": result["message"],
        "actions": [],
        "rag": {"status": "skipped", "reason": "assistant_identity_question", "sources": []},
        "retrieved_sources": [],
    }


def _build_out_of_scope_response() -> dict:
    result = {
        "kind": "analysis",
        "message": (
            "That's outside what I can responsibly help with — I can't predict markets "
            "or asset prices (stocks, crypto, gold), give buy/sell tips, or move money on "
            "your behalf. What I'm good at is your day-to-day finances: tracking spending, "
            "checking budgets, planning savings goals, and figuring out how much you can "
            "safely set aside. Want me to take a look at any of those?"
        ),
        "action": None,
        "signals": [],
        "needs_clarification": False,
    }
    return {
        "model_output": result,
        "response": result["message"],
        "actions": [],
        "rag": {"status": "skipped", "reason": "out_of_scope_question", "sources": []},
        "retrieved_sources": [],
    }


def _apply_style_directive(message: str) -> str:
    """Prefix the user turn with a hidden style hint when appropriate."""
    if _CALC_AUDIT_PATTERNS.search(message):
        hint = (
            "[STYLE: The user is asking about calculation reliability, not asking for a spending diagnosis. "
            "Explain that deterministic totals come from AWAD2 structured data and pre-computed FINANCIAL CONTEXT. "
            "Show how to verify: total spent=sum expenses, surplus=income-total spent, savings rate=surplus/income, "
            "and retrieved sources [S1], [S2] support transaction-level claims. Do not mention unrelated transactions "
            "unless using them only as examples of evidence. Return ONLY one valid JSON object.]\n\n"
        )
        return hint + message
    if _is_advice_question(message):
        hint = (
            "[STYLE: This is a general/advisory question that FINANCIAL CONTEXT does not "
            "directly answer. Reply naturally and warmly with sound, conventional guidance, "
            "hedged ('generally...', 'a common rule is...'). Answer the question itself — do "
            "NOT pivot into a spending summary. Mention the user's own numbers only if "
            "directly relevant, and NEVER invent a personal amount, balance, budget, or "
            "target. Generic reference figures (e.g. 3-6 months of expenses, 50/30/20) are "
            "fine. End with one concrete next step or a follow-up question. Return ONLY one "
            "valid JSON object matching the schema; put all natural language inside the "
            "`message` field.]\n\n"
        )
        return hint + message
    if _ANALYSIS_PATTERNS.search(message):
        hint = (
            "[STYLE: Give a full, conversational overview. Cite actual numbers from "
            "FINANCIAL CONTEXT. Cover: headline (savings rate / surplus / top verdict), "
            "top 1-2 spending categories with %, month-over-month change or any anomaly, "
            "goal and budget status, one concrete suggestion for the user's role, and "
            "end with a friendly follow-up question. Do not be terse. Return ONLY one "
            "valid JSON object matching the schema; put all natural language inside "
            "the `message` field.]\n\n"
        )
        return hint + message
    if _BUDGET_PATTERNS.search(message):
        hint = (
            "[STYLE: Budget conversation. If the user is still asking or you are still "
            "proposing a limit, return kind=\"analysis\" with a conversational message. "
            "If the user has now confirmed the amount and category (including short "
            "replies like 'yes', 'go ahead', 'food, 3.6m', or just an amount after you "
            "proposed one), return kind=\"action\" with type=\"create_budget\", the "
            "agreed amount as an integer in VND, and the category. For changes to an "
            "existing budget use type=\"update_budget\"; for removal use "
            "type=\"delete_budget\". Always return ONLY one valid JSON object.]\n\n"
        )
        return hint + message
    return (
        "[STYLE: Answer in 3-5 sentences. Cite actual numbers from FINANCIAL "
        "CONTEXT. End with one concrete suggestion appropriate to the user's "
        "role and a short follow-up question. Return ONLY one valid JSON "
        "object matching the schema; put all natural language inside the "
        "`message` field. Do not output prose outside JSON.]\n\n"
        + message
    )


def _is_short_budget_followup(message: str) -> bool:
    text = message.strip()
    if len(text) > 120 or "\n" in text or "?" in text:
        return False
    if len(text.split()) > 18:
        return False
    has_acceptance = re.search(r"\b(ok|okay|yes|sure|confirm|confirmed|do it|go ahead|make|set|create|apply|please)\b", text, re.IGNORECASE)
    has_amount = re.search(r"\d", text)
    has_category = re.search(r"\b(food|grocery|bill|shopping|transport|entertainment|health|education|equipment|software|other)\b", text, re.IGNORECASE)
    return bool(has_acceptance and (has_amount or has_category))


def _history_allowed_for_turn(message: str) -> bool:
    # Current FINANCIAL CONTEXT is authoritative. For analysis questions, old
    # assistant answers can contain stale budget claims, so do not replay them.
    return _is_short_budget_followup(message)


def _context_budget_categories(context: str) -> set[str]:
    # Match the header whether or not it carries a parenthesized period label.
    header_match = re.search(r"CATEGORY BUDGET STATUS[^:]*:", context)
    if not header_match:
        return set()
    section = context[header_match.end():]
    section = section.split("\nBUDGET VERDICT:", 1)[0]
    categories = set()
    for line in section.splitlines():
        match = re.match(r"\s*-\s*([^:]+):", line)
        if match:
            categories.add(match.group(1).strip().lower())
    return categories


def _context_known_categories(context: str) -> set[str]:
    """All categories the user has — from spending and budget sections."""
    known: set[str] = set(_context_budget_categories(context))
    spend_match = re.search(r"SPENDING BY CATEGORY[^:]*:", context)
    if spend_match:
        section = context[spend_match.end():]
        section = section.split("\n\n", 1)[0]
        for line in section.splitlines():
            match = re.match(r"\s*-\s*([^:]+):", line)
            if match:
                known.add(match.group(1).strip().lower())
    return known


def _user_named_a_category(message: str, known_categories: set[str]) -> bool:
    msg = message.lower()
    return any(re.search(rf"\b{re.escape(cat)}\b", msg) for cat in known_categories if cat)


_LIMIT_SYNONYMS = (
    r"(?:budget|limit|cap|ceiling|allowance|threshold|target|"
    r"amount you (?:said|proposed|set)|limit you (?:said|proposed|set))"
)
_VND_AMOUNT_RE = re.compile(
    r"\b\d{1,3}(?:[.,]\d{3})+(?:\s*(?:VND|vnd|đ|d))?\b"
    r"|\b\d+(?:[.,]\d+)?\s*(?:m|million|tr|triệu)\b",
    re.IGNORECASE,
)
_VND_MULT_RE = re.compile(
    r"^(\d+(?:[.,]\d+)?)\s*(m|million|tr|triệu)$", re.IGNORECASE
)


def _amount_to_int(token: str) -> int | None:
    """Normalize a single VND token to its integer value, or None.

    Handles both grouped thousands ("1.500.000", "45,000,000 VND") and
    multiplier shorthand ("45M", "1.5 million", "3.6tr", "45 triệu") so that a
    figure echoed by the user as "45M" matches the same value written as
    "45.000.000 VND" in context.
    """
    t = token.strip().lower()
    mult = _VND_MULT_RE.match(t)
    if mult:
        try:
            return int(round(float(mult.group(1).replace(",", ".")) * 1_000_000))
        except ValueError:
            return None
    grp = re.sub(r"\s*(?:vnd|đ|d)$", "", t).replace(".", "").replace(",", "")
    return int(grp) if grp.isdigit() else None


def _vnd_values(text: str) -> set[int]:
    """All distinct VND integer values mentioned in a block of text."""
    if not text:
        return set()
    values: set[int] = set()
    for match in _VND_AMOUNT_RE.finditer(text):
        val = _amount_to_int(match.group(0))
        if val is not None:
            values.add(val)
    return values


def _replacement_message(claimed: str) -> str:
    return (
        f"I don't see a current {claimed} category budget in your saved budget data, "
        f"so I can't say whether that category will overshoot a budget. "
        "Use the budget page or set a category budget first, then I can compare "
        "month-to-date spending against the saved limit. "
        "Based on the current financial context, any category budget claim must come from CATEGORY BUDGET STATUS."
    )


def _remove_unsupported_budget_claims(result: dict, context: str) -> dict:
    if result.get("kind") != "analysis":
        return result
    message = result.get("message")
    if not isinstance(message, str):
        return result

    from fina_schema import VALID_CATEGORIES  # local import to avoid cycles
    budget_categories = _context_budget_categories(context)
    claimed_unbudgeted: str | None = None

    # Iterate every known FINA category. For each one, flag the message if:
    #   (a) "<Category> <limit-synonym>" appears (explicit, e.g. "Shopping budget")
    #       AND the category is not in CATEGORY BUDGET STATUS.
    #   (b) The category name AND a limit-synonym AND a money amount all
    #       appear in the same sentence ("...Shopping spending... 5.000.000 VND
    #       limit you set"), AND the category is not in CATEGORY BUDGET STATUS.
    # (a) catches the obvious phrasing; (b) catches the detached-noun phrasing
    # that previously slipped past the original regex.
    sentences = re.split(r"(?<=[.!?])\s+", message)
    for cat in VALID_CATEGORIES:
        if cat.lower() in budget_categories:
            continue
        cat_word = rf"\b{re.escape(cat)}\b"

        # (a) explicit "<Category> <synonym>"
        if re.search(rf"{cat_word}\s+{_LIMIT_SYNONYMS}\b", message, re.IGNORECASE):
            claimed_unbudgeted = cat
            break

        # (b2) indirect: "<Category> spending ... <synonym>" within a single
        # sentence. Catches the phrasing observed in the wild ("Bill spending
        # of 1.100.000 VND is within the 1.500.000 VND limit you proposed")
        # which (a) missed because the synonym is detached from the category
        # by an amount phrase. Iterates sentences so VND thousand-separator
        # periods (e.g. "1.100.000") don't confuse the matcher.
        for sentence in sentences:
            if re.search(
                rf"{cat_word}\s+spend(?:ing)?.+?{_LIMIT_SYNONYMS}",
                sentence, re.IGNORECASE,
            ):
                claimed_unbudgeted = cat
                break
        if claimed_unbudgeted:
            break

        # (b) sentence-level co-occurrence with an amount + synonym
        for sentence in sentences:
            if not re.search(cat_word, sentence, re.IGNORECASE):
                continue
            if not re.search(rf"\b{_LIMIT_SYNONYMS}\b", sentence, re.IGNORECASE):
                continue
            if not _VND_AMOUNT_RE.search(sentence):
                continue
            claimed_unbudgeted = cat
            break
        if claimed_unbudgeted:
            break

    if claimed_unbudgeted is not None:
        result = dict(result)
        result["message"] = _replacement_message(claimed_unbudgeted)
        signals = result.get("signals")
        result["signals"] = signals if isinstance(signals, list) else []
        result["needs_clarification"] = False
    return result


def _redact_unsupported_amounts(
    result: dict, context: str, user_message: str
) -> tuple[dict, list[str]]:
    """Redact personal VND figures the model invented.

    An amount in the model's message is "supported" only if its value also
    appears in FINANCIAL CONTEXT or in the user's own question. Anything else is
    a fabricated personal figure (the "36M tax reserve" failure mode) and gets
    replaced with a placeholder. Only VND amounts are touched — percentages,
    ratios (50/30/20) and "3-6 months" are left alone, and context-derived
    numbers survive by construction.
    """
    if result.get("kind") != "analysis":
        return result, []
    message = result.get("message")
    if not isinstance(message, str):
        return result, []

    supported = _vnd_values(context) | _vnd_values(user_message)
    redacted: list[str] = []

    def _is_supported(value: int) -> bool:
        # Exact match, or within 5% of a supported value so the model rounding a
        # real figure for readability ("7.9M" for 7.870.000) is not redacted,
        # while a fabricated number far from any real one still is.
        return any(
            abs(value - s) <= 0.05 * max(value, s, 1) for s in supported
        )

    def _sub(match: re.Match) -> str:
        token = match.group(0)
        value = _amount_to_int(token)
        if value is None or _is_supported(value):
            return token
        redacted.append(token.strip())
        return "[an amount I can't confirm from your data]"

    new_message = _VND_AMOUNT_RE.sub(_sub, message)
    if redacted:
        result = dict(result)
        result["message"] = new_message
    return result, redacted


@app.post("/chat")
async def chat_endpoint(request: UserRequest):
    global model, tokenizer
    role = await resolve_role(request.user_id, request.role)
    print(f"\n[CHAT] User: {request.user_id} | Role: {role} | Q: {request.message}")
    if request.adapter and request.adapter.lower() != ACTIVE_ADAPTER:
        print(f"[CHAT] adapter override ignored; demo is fixed to {ACTIVE_ADAPTER}")

    if _ASSISTANT_IDENTITY_PATTERNS.search(request.message):
        print("[CHAT] Deterministic route: assistant identity")
        return _build_assistant_identity_response()

    if _OUT_OF_SCOPE_PATTERNS.search(request.message):
        print("[CHAT] Deterministic route: out of scope (refusal)")
        return _build_out_of_scope_response()

    inferred = infer_period_from_text(request.message)
    effective_period = inferred or request.period or "month"
    if inferred and inferred != request.period:
        print(f"[CHAT] Period inferred from question: {request.period!r} -> {inferred!r}")

    if request.use_rag:
        financial_context, rag_result = await build_hybrid_context(
            request.user_id,
            role,
            request.message,
            effective_period,
        )
    else:
        financial_context = await format_financial_context(
            request.user_id, role, effective_period
        )
        rag_result = RetrievalResult(context="", sources=[], status="rag_disabled")
        print("[CHAT] use_rag=False — RAG block skipped for this request.")

    if _CALC_AUDIT_PATTERNS.search(request.message):
        print("[CHAT] Deterministic route: calculation audit")
        return _build_calculation_audit_response(financial_context, rag_result)

    # ── DEBUG: dump exactly what context is fed to the model so we can see
    # whether wrong numbers come from the structured Mac fetch, the RAG block,
    # or a model hallucination. Toggle off by setting FINA_DEBUG_CTX=0.
    if os.getenv("FINA_DEBUG_CTX", "1") != "0":
        print(f"\n[CHAT-DEBUG] ===== STRUCTURED CONTEXT for {request.user_id} =====")
        print(financial_context)
        rag_block = rag_result.context if rag_result and rag_result.context else "(none)"
        print(f"\n[CHAT-DEBUG] ===== RAG BLOCK ({len(rag_block)} chars) =====")
        print(rag_block[:2000])
        print(f"[CHAT-DEBUG] ===== END CONTEXT =====\n")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": financial_context},
        {"role": "assistant", "content": "Understood."},
    ]

    if _history_allowed_for_turn(request.message):
        for turn in request.history[-6:]:
            if turn.role == "user":
                messages.append({"role": "user", "content": turn.content})
            elif turn.role == "assistant":
                parsed_turn = parse_model_output(turn.content, log_failure=False)
                if parsed_turn is not None:
                    history_result = _model_output_to_dict(parsed_turn)
                    history_result["message"] = "[previous assistant reply omitted; use current FINANCIAL CONTEXT for all numbers]"
                    messages.append({
                        "role": "assistant",
                        "content": json.dumps(history_result, ensure_ascii=False),
                    })
                else:
                    logger.debug("Skipping non-JSON assistant history turn to preserve output format")

    steered_message = _apply_style_directive(request.message)
    messages.append({"role": "user", "content": steered_message})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    input_length = inputs["input_ids"].shape[-1]

    print(f"[CHAT] adapter={ACTIVE_ADAPTER} use_rag={request.use_rag}")

    gen_kwargs = dict(
        max_new_tokens=1024,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    raw_output = _decode_new_tokens(outputs, input_length, tokenizer)
    print(f"[CHAT] Raw output: {raw_output[:200]}")

    # Parse JSON output (status surfaced for evaluation logs)
    parsed, schema_status = parse_model_output_with_status(raw_output)
    if parsed is None:
        logger.warning("Model output parse failed, using fallback")
        parsed = fallback_output(raw_output)

    safety_ok, safety_reason = validate_action_safety(
        parsed.action,
        known_categories=_context_known_categories(financial_context),
    )
    if not safety_ok:
        logger.warning("Rejected unsafe action: %s", safety_reason)
        parsed = parsed.model_copy(update={"action": None})

    result = _model_output_to_dict(parsed)
    if _user_named_a_category(request.message, _context_known_categories(financial_context)):
        result = _remove_unsupported_budget_claims(result, financial_context)
    result, redacted_amounts = _redact_unsupported_amounts(
        result, financial_context, request.message
    )
    if redacted_amounts:
        print(f"[CHAT] Numeric guard redacted unsupported amounts: {redacted_amounts}")

    response = {
        "model_output": result,
        "response": result["message"],
        "actions": [result["action"]] if result.get("action") else [],
        "rag": rag_result.as_dict(),
        "retrieved_sources": [
            {"id": source.id, "text": source.text, "metadata": source.metadata}
            for source in rag_result.sources
        ],
        "meta": {
            "schema_status": schema_status,
            "action_safety": {"ok": safety_ok, "reason": safety_reason},
            "numeric_guard": {"redacted": redacted_amounts},
        },
    }
    return response


@app.post("/chat/stream")
async def chat_stream_endpoint(request: UserRequest):
    global model, tokenizer
    role = await resolve_role(request.user_id, request.role)
    print(f"\n[STREAM] User: {request.user_id} | Role: {role} | Q: {request.message}")
    if request.adapter and request.adapter.lower() != ACTIVE_ADAPTER:
        print(f"[STREAM] adapter override ignored; demo is fixed to {ACTIVE_ADAPTER}")

    if _ASSISTANT_IDENTITY_PATTERNS.search(request.message):
        print("[STREAM] Deterministic route: assistant identity")
        response = _build_assistant_identity_response()

        async def identity_generator():
            yield f"data: {json.dumps({'final': response['model_output'], 'rag': response['rag']})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(identity_generator(), media_type="text/event-stream")

    if _OUT_OF_SCOPE_PATTERNS.search(request.message):
        print("[STREAM] Deterministic route: out of scope (refusal)")
        response = _build_out_of_scope_response()

        async def out_of_scope_generator():
            yield f"data: {json.dumps({'final': response['model_output'], 'rag': response['rag']})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(out_of_scope_generator(), media_type="text/event-stream")

    inferred = infer_period_from_text(request.message)
    effective_period = inferred or request.period or "month"
    if inferred and inferred != request.period:
        print(f"[STREAM] Period inferred from question: {request.period!r} -> {inferred!r}")

    financial_context, rag_result = await build_hybrid_context(
        request.user_id,
        role,
        request.message,
        effective_period,
    )

    if _CALC_AUDIT_PATTERNS.search(request.message):
        print("[STREAM] Deterministic route: calculation audit")
        response = _build_calculation_audit_response(financial_context, rag_result)

        async def audit_generator():
            yield f"data: {json.dumps({'final': response['model_output'], 'rag': response['rag']})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(audit_generator(), media_type="text/event-stream")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": financial_context},
        {"role": "assistant", "content": "Understood."},
    ]

    if _history_allowed_for_turn(request.message):
        for turn in request.history[-6:]:
            if turn.role == "user":
                messages.append({"role": turn.role, "content": turn.content})
            elif turn.role == "assistant":
                parsed_turn = parse_model_output(turn.content, log_failure=False)
                if parsed_turn is not None:
                    history_result = _model_output_to_dict(parsed_turn)
                    history_result["message"] = "[previous assistant reply omitted; use current FINANCIAL CONTEXT for all numbers]"
                    messages.append({
                        "role": "assistant",
                        "content": json.dumps(history_result, ensure_ascii=False),
                    })
                else:
                    logger.debug("Skipping non-JSON assistant history turn to preserve output format")

    messages.append({"role": "user", "content": _apply_style_directive(request.message)})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    print(f"[STREAM] adapter={ACTIVE_ADAPTER} use_rag={request.use_rag}")

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = {
        **inputs,
        "max_new_tokens": 1024,
        "do_sample": False,
        "num_beams": 1,
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

        # Parse complete response as JSON (status surfaced for evaluation logs)
        parsed, schema_status = parse_model_output_with_status(full_text)
        if parsed is None:
            parsed = fallback_output(full_text)

        safety_ok, safety_reason = validate_action_safety(
            parsed.action,
            known_categories=_context_known_categories(financial_context),
        )
        if not safety_ok:
            logger.warning("[STREAM] Rejected unsafe action: %s", safety_reason)
            parsed = parsed.model_copy(update={"action": None})

        result = _model_output_to_dict(parsed)
        if _user_named_a_category(request.message, _context_known_categories(financial_context)):
            result = _remove_unsupported_budget_claims(result, financial_context)
        result, redacted_amounts = _redact_unsupported_amounts(
            result, financial_context, request.message
        )
        if redacted_amounts:
            print(f"[STREAM] Numeric guard redacted unsupported amounts: {redacted_amounts}")
        meta = {
            "schema_status": schema_status,
            "action_safety": {"ok": safety_ok, "reason": safety_reason},
            "numeric_guard": {"redacted": redacted_amounts},
        }
        yield f"data: {json.dumps({'final': result, 'rag': rag_result.as_dict(), 'meta': meta})}\n\n"
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


@app.get("/forecast/{user_id}")
async def get_forecast(user_id: str):
    try:
        return forecast_user(user_id)
    except Exception as e:
        print(f"[Forecast endpoint] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
