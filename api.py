import os
import torch
import pymysql.cursors
import json
from decimal import Decimal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import uvicorn
from dotenv import load_dotenv
from forecasting.predict import forecast_user
from categorizer.predict import categorize
from rag.retriever import retrieve_context
from rag.store import index_user

load_dotenv()

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_NAME = "financial_qwen_native_v1"

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", "3308")),
    "user":     os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "financedb"),
    "cursorclass": pymysql.cursors.DictCursor
}

SYSTEM_PROMPT = """
You are FINA, an intelligent Financial AI Agent.

### CAPABILITIES:
1. **Analysis:** Analyze spending patterns.
2. **Action:** You can LOG transactions if the user tells you to (e.g., "I spent 50k on coffee").
3. **Budgeting:** - Default: 50/30/20 Rule.
   - **Custom:** IF the user asks for a different split (e.g., 70/20/10), CALCULATE it for them. Do not lecture them.

### FORMATTING:
- Use **Bold** for numbers (e.g., **50.000 VND**).
- Always use the User's Currency.
"""

app = FastAPI(title="FINA Financial AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten to Mac IP in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
tokenizer = None

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    print("API STARTUP: Loading FINA Brain...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_NAME)
    model.eval()
    print("API READY: FINA is online.")

def convert_decimals(obj):
    """Recursively converts Decimal objects (from SQL) to floats (for JSON)."""
    if isinstance(obj, list):
        return [convert_decimals(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, Decimal):
        return float(obj)
    return obj

def format_vnd(amount):
    """Formats number to Vietnamese style: 28.350.000"""
    try:
        if amount is None: return "0"
        return "{:,.0f}".format(float(amount)).replace(",", ".")
    except:
        return str(amount)

def fetch_raw_data(user_id):
    """
    Fetches transactions for the given finance user_id (fid).
    """
    try:
        connection = pymysql.connect(**DB_CONFIG)
        with connection.cursor() as cursor:
            currency = 'VND'
            # Income
            cursor.execute(
                "SELECT SUM(amount) as total FROM transactions WHERE user_id = %s AND type = 'INCOME'",
                (user_id,)
            )
            res_income = cursor.fetchone()
            income = float(res_income['total']) if res_income and res_income['total'] else 0.0

            # Expenses by category
            cursor.execute("""
                SELECT c.name as category_name, SUM(t.amount) as spent 
                FROM transactions t
                JOIN categories c ON t.category_id = c.id
                WHERE t.user_id = %s AND t.type = 'EXPENSE'
                GROUP BY c.name
            """, (user_id,))
            spending = cursor.fetchall()

            # Recent history
            cursor.execute("""
                SELECT t.id, t.amount, t.type, t.occurred_at, c.name as category 
                FROM transactions t
                LEFT JOIN categories c ON t.category_id = c.id
                WHERE t.user_id = %s
                ORDER BY t.occurred_at DESC LIMIT 10
            """, (user_id,))
            history = cursor.fetchall()

        connection.close()

        print(f"[DB] User {user_id} | Income: {income} | Spending rows: {len(spending)} | History rows: {len(history)}")

        return {
            "currency": currency,
            "income": income,
            "spending": convert_decimals(spending),
            "history": convert_decimals(history)
        }
    except Exception as e:
        print(f"[DB Error] {e}")
        return None

def format_financial_summary(user_id):
    """Formats financial data as a string context for the LLM."""
    data = fetch_raw_data(user_id)
    if not data:
        return "Error: Could not fetch financial data. The database may be unreachable."

    cur = data['currency']
    income = data['income']
    spending_data = data['spending']

    budget_needs = income * 0.50
    budget_wants = income * 0.30
    budget_savings = income * 0.20

    # Format spending breakdown
    spending_lines = "\n".join(
        [f"    - {s['category_name']}: {format_vnd(s['spent'])} {cur}" for s in spending_data]
    ) if spending_data else "    - No expenses recorded yet."

    summary = f"""
--- FINANCIAL CONTEXT ---
CURRENCY: {cur}
TOTAL INCOME: {format_vnd(income)} {cur}

ACTUAL SPENDING BY CATEGORY:
{spending_lines}

TARGET BUDGETS (50/30/20 Rule):
- Needs Limit (50%):   {format_vnd(budget_needs)} {cur}
- Wants Limit (30%):   {format_vnd(budget_wants)} {cur}
- Savings Target (20%): {format_vnd(budget_savings)} {cur}
---------------------------
"""
    return summary

def generate_dashboard_json(user_id, role="User"):
    """Generates Dashboard JSON using the User's Currency & VN Format."""
    data = fetch_raw_data(user_id)
    if not data: return None

    cur         = data['currency']
    total_spent = sum(item['spent'] for item in data['spending'])
    savings     = data['income'] - total_spent
    spend_ratio = (total_spent / data['income'] * 100) if data['income'] > 0 else 0

    sorted_spending = sorted(data['spending'], key=lambda x: x['spent'], reverse=True)
    highest_category = sorted_spending[0]['category_name'] if sorted_spending else "General"
    highest_amount   = sorted_spending[0]['spent'] if sorted_spending else 0

    # Build per-category breakdown with budget comparison
    budget_wants = data['income'] * 0.30
    category_lines = "\n".join(
        f"  - {s['category_name']}: {format_vnd(s['spent'])} {cur} "
        f"({'OVER BUDGET' if s['spent'] > budget_wants else 'OK'})"
        for s in sorted_spending
    ) if sorted_spending else "  - No expenses recorded."

    prompt = f"""You are FINA. Analyze this user's financial data and return a JSON dashboard.

FINANCIAL DATA:
  Role: {role}
  Income:       {format_vnd(data['income'])} {cur}
  Total Spent:  {format_vnd(total_spent)} {cur} ({spend_ratio:.1f}% of income)
  Net Savings:  {format_vnd(savings)} {cur}

SPENDING BY CATEGORY (30% wants limit = {format_vnd(budget_wants)} {cur}):
{category_lines}

50/30/20 BUDGET TARGETS:
  Needs  (50%): {format_vnd(data['income'] * 0.50)} {cur}
  Wants  (30%): {format_vnd(data['income'] * 0.30)} {cur}
  Savings(20%): {format_vnd(data['income'] * 0.20)} {cur}

NUMBER FORMAT RULE: Always write amounts as Vietnamese style with dots — e.g. 1.500.000 VND, NOT $1,500,000 or 1500000 or $100.
Currency is always VND, never use $ or USD.

TASK: Return ONLY a valid JSON object — no markdown, no explanation.
Use specific numbers from the data above in Vietnamese format. Generate 3 smart_insights that are actionable and data-driven.
insight types allowed: "warning", "success", "info", "danger"

JSON structure:
{{
  "initial_message": "short greeting mentioning net savings amount",
  "summary_cards": [
    {{"id": 1, "type": "info",    "title": "Total Income",  "subtitle": "{format_vnd(data['income'])} {cur}",  "badge": "Verified"}},
    {{"id": 2, "type": "danger",  "title": "Total Spent",   "subtitle": "{format_vnd(total_spent)} {cur}",     "badge": "Tracked"}},
    {{"id": 3, "type": "success", "title": "Net Savings",   "subtitle": "{format_vnd(savings)} {cur}",         "badge": "Cash Flow"}}
  ],
  "smart_insights": [
    {{"id": 1, "type": "warning|success|info|danger", "title": "...", "desc": "specific insight with numbers"}},
    {{"id": 2, "type": "...", "title": "...", "desc": "..."}},
    {{"id": 3, "type": "...", "title": "...", "desc": "..."}}
  ]
}}"""

    messages = [
        {"role": "system", "content": "You are FINA, a financial AI. Output valid JSON only. No markdown."},
        {"role": "user",   "content": prompt}
    ]

    try:
        text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=600, temperature=0.2,
                do_sample=True, pad_token_id=tokenizer.eos_token_id
            )
        response  = tokenizer.decode(outputs[0], skip_special_tokens=True)
        raw_text  = response.split("assistant")[-1].strip() if "assistant" in response else response
        clean_json = raw_text.replace('```json', '').replace('```', '').strip()
        result    = json.loads(clean_json)
        if "smart_insights" in result:
            result = _fix_currency_format(result, cur)
            # Always use LSTM prediction — never let LLM override it
            result["prediction"] = {
                "amount":     _get_monthly_forecast(user_id, total_spent),
                "confidence": 80 if _lstm_available(user_id) else 60,
                "label":      "Projected spending next month"
            }
            return result
    except Exception as e:
        print(f"[AI JSON GEN FAILED - Using Fallback]: {e}")

    # Fallback — used when LLM JSON generation fails
    spend_ratio  = (total_spent / data['income'] * 100) if data['income'] > 0 else 0
    savings_ratio = (savings / data['income'] * 100) if data['income'] > 0 else 0
    budget_health = "success" if spend_ratio < 80 else "warning" if spend_ratio < 100 else "danger"
    savings_type  = "success" if savings_ratio >= 20 else "warning"

    insights = [
        {"id": 1, "type": "warning" if highest_amount > data['income'] * 0.30 else "info",
         "title": "Top Expense",
         "desc": f"Highest spending category: {highest_category} at {format_vnd(highest_amount)} {cur} "
                 f"({highest_amount / data['income'] * 100:.1f}% of income)."},
        {"id": 2, "type": budget_health,
         "title": "Budget Health",
         "desc": f"You spent {format_vnd(total_spent)} {cur} ({spend_ratio:.1f}% of income). "
                 f"{'On track.' if spend_ratio < 80 else 'Consider cutting discretionary spending.'}"},
        {"id": 3, "type": savings_type,
         "title": "Savings Rate",
         "desc": f"You saved {format_vnd(savings)} {cur} ({savings_ratio:.1f}% of income). "
                 + ("Great — above the 20% target!" if savings_ratio >= 20
                    else f"Target is 20% ({format_vnd(data['income'] * 0.20)} {cur}).")},
    ]

    return {
        "initial_message": f"Welcome back! Your net savings this month is {format_vnd(savings)} {cur}.",
        "summary_cards": [
            {"id": 1, "type": "info",    "title": "Total Income", "subtitle": f"{format_vnd(data['income'])} {cur}", "badge": "Verified"},
            {"id": 2, "type": "danger",  "title": "Total Spent",  "subtitle": f"{format_vnd(total_spent)} {cur}",    "badge": "Tracked"},
            {"id": 3, "type": "success", "title": "Net Savings",  "subtitle": f"{format_vnd(savings)} {cur}",        "badge": "Cash Flow"}
        ],
        "smart_insights": insights,
        "prediction": {
            "amount":     _get_monthly_forecast(user_id, total_spent),
            "confidence": 80 if _lstm_available(user_id) else 60,
            "label":      "Projected spending next month"
        }
    }


# ==========================================
# FORMATTING HELPERS
# ==========================================

import re as _re

def _fix_currency_format(obj, cur="VND"):
    """
    Recursively walk the dashboard JSON and fix any rogue currency formats.
    Replaces patterns like $1,500,000 or $1.500.000 or USD 1500000
    with properly formatted Vietnamese-style amounts: 1.500.000 VND
    """
    if isinstance(obj, dict):
        return {k: _fix_currency_format(v, cur) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fix_currency_format(i, cur) for i in obj]
    if isinstance(obj, str):
        # Replace $1,234,567 or $1,234,567 VND → 1.234.567 VND
        def replace_dollar(m):
            digits = m.group(1).replace(",", "")
            try:
                return format_vnd(float(digits)) + f" {cur} "
            except:
                return m.group(0)
        obj = _re.sub(r'\$\s*([\d,]+)(\s*VND)?', replace_dollar, obj)
        # Replace USD 1234567 or USD 1,234,567
        obj = _re.sub(r'\bUSD\s*([\d,]+)(\s*VND)?', replace_dollar, obj)
        # Remove any orphan $ signs and collapse double spaces
        obj = obj.replace("$", "")
        obj = _re.sub(r' {2,}', ' ', obj)
    return obj


# ==========================================
# FORECASTING HELPERS
# ==========================================

def _lstm_available(user_id: int) -> bool:
    from pathlib import Path
    return (Path("models") / f"user_{user_id}" / "lstm.pt").exists()

def _get_monthly_forecast(user_id: int, fallback_total: float) -> float:
    try:
        result = forecast_user(user_id)
        return result["monthly"]["total"]
    except Exception as e:
        print(f"[Forecast Error - using fallback]: {e}")
        return fallback_total * 1.05


# ==========================================
# ENDPOINTS
# ==========================================

@app.get("/")
def health_check():
    return {"status": "online", "model": BASE_MODEL}

@app.get("/dashboard/{user_id}")
async def get_dashboard_summary(user_id: int, role: str = "User"):
    print(f"[Dashboard] Request for User {user_id} | Role: {role}")
    return generate_dashboard_json(user_id, role)

@app.get("/history/{user_id}")
async def get_history(user_id: int):
    print(f"[History] Request for User {user_id}")
    data = fetch_raw_data(user_id)
    return {"history": data['history'] if data else []}

@app.get("/forecast/{user_id}")
async def get_forecast(user_id: int):
    print(f"[Forecast] Request for User {user_id}")
    return forecast_user(user_id)


class CategorizeRequest(BaseModel):
    description: str

@app.post("/categorize")
async def categorize_transaction(request: CategorizeRequest):
    print(f"[Categorize] '{request.description}'")
    return categorize(request.description)

@app.post("/rag/index/{user_id}")
async def rag_index(user_id: int):
    print(f"[RAG] Indexing transactions for User {user_id}")
    count = index_user(user_id)
    return {"indexed": count, "user_id": user_id}


class UserRequest(BaseModel):
    user_id: int
    role: str = "Student"
    mode: str = "Standard"
    message: str


@app.post("/chat")
async def chat_endpoint(request: UserRequest):
    global model, tokenizer

    print(f"\n[FINA CHAT] User: {request.user_id} | Role: {request.role} | Q: {request.message}")

    financial_context = format_financial_summary(request.user_id)
    rag_context       = retrieve_context(request.user_id, request.message)
    print(f"[FINA CHAT] Context built:\n{financial_context}")

    full_message = (
        f"User Role: {request.role}\n"
        f"Selected Mode: {request.mode}\n"
        f"Financial Context:\n{financial_context}"
        f"{rag_context}"
        f"User Question: {request.message}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": full_message}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.4,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    clean_response = response.split("assistant")[-1].strip() if "assistant" in response else response

    return {"response": clean_response}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8105)