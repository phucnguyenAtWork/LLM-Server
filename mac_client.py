"""
Mac API Client — FINA → Mac (Bun/Hono) data bridge
=====================================================
Instead of hitting MySQL directly, FINA calls the Mac's
TypeScript backend for user data, transactions, and categories.

Falls back to direct DB access if the Mac API is unreachable
(development/offline mode).
"""

import os
import json
import httpx
import pymysql
import pymysql.cursors
from decimal import Decimal
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ── Mac Backend URL ──────────────────────────────────────────────────────────
MAC_API_URL = os.getenv("MAC_API_URL", "http://100.109.225.15:4001/api")

# ── Direct DB fallback (for offline / dev) ───────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "100.109.225.15"),
    "port":     int(os.getenv("DB_PORT", "3308")),
    "user":     os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "financedb"),
    "cursorclass": pymysql.cursors.DictCursor,
    "connect_timeout": 3,
}

# Timeout for Mac API calls (seconds)
_TIMEOUT = httpx.Timeout(10.0, connect=3.0)

_client = httpx.AsyncClient(base_url=MAC_API_URL, timeout=_TIMEOUT)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _convert_decimals(obj):
    """Recursively converts Decimal → float for JSON serialization."""
    if isinstance(obj, list):
        return [_convert_decimals(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _convert_decimals(v) for k, v in obj.items()}
    if isinstance(obj, Decimal):
        return float(obj)
    return obj


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — called by api.py, forecasting, rag
# ═══════════════════════════════════════════════════════════════════════════════

async def get_user_profile(user_id: str) -> Optional[dict]:
    """
    Fetch user profile from Mac.
    Returns None on failure.
    """
    try:
        r = await _client.get(f"/fina/users/{user_id}")
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[MacClient] get_user_profile failed: {e}")
        return None


async def get_user_accounts(user_id: str) -> list[dict]:
    """
    Fetch user's accounts from Mac (contains currency info).
    Returns empty list on failure.
    """
    try:
        r = await _client.get(f"/fina/users/{user_id}/accounts")
        r.raise_for_status()
        data = r.json()
        return data.get("accounts", data) if isinstance(data, dict) else data
    except Exception as e:
        print(f"[MacClient] get_user_accounts failed: {e}")
        return []


async def get_user_currency(user_id: str) -> str:
    """
    Resolve the user's primary currency from their first account.
    Falls back to 'VND' if unavailable.
    """
    accounts = await get_user_accounts(user_id)
    if accounts:
        return accounts[0].get("currency", "VND")
    # DB fallback
    try:
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT currency FROM accounts WHERE user_id = %s LIMIT 1",
                (user_id,)
            )
            row = cursor.fetchone()
        conn.close()
        if row:
            return row["currency"]
    except Exception as e:
        print(f"[MacClient] DB currency fallback error: {e}")
    return "VND"


async def get_transactions(user_id: str, limit: int = 200) -> list[dict]:
    """
    Fetch user transactions from Mac backend.
    Falls back to direct DB if Mac is unreachable.
    """
    try:
        r = await _client.get(f"/fina/users/{user_id}/transactions", params={"limit": limit})
        r.raise_for_status()
        data = r.json()
        return data.get("transactions", data) if isinstance(data, dict) else data
    except Exception as e:
        print(f"[MacClient] API unreachable ({e}), falling back to direct DB")
        return _db_fetch_transactions(user_id, limit)


async def get_income_and_spending(user_id: str, period: str = "month") -> Optional[dict]:
    """
    Fetch aggregated financial summary from Mac backend.
    Falls back to direct DB on failure.

    period: "month" (current month), "year" (current year),
            "all" (all time), or "YYYY-MM" (specific month)

    Returns: {currency, income, spending: [{category_name, spent}], history: [...], period}
    """
    try:
        r = await _client.get(f"/fina/users/{user_id}/summary", params={"period": period})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[MacClient] API unreachable ({e}), falling back to direct DB")
        return _db_fetch_raw_data(user_id, period)


async def get_daily_spending(user_id: str) -> list[dict]:
    """
    Fetch daily spending breakdown (for LSTM forecasting).
    Falls back to direct DB on failure.

    Returns: [{day, category, amount}, ...]
    """
    try:
        r = await _client.get(f"/fina/users/{user_id}/spending/daily")
        r.raise_for_status()
        data = r.json()
        return data.get("rows", data) if isinstance(data, dict) else data
    except Exception as e:
        print(f"[MacClient] API unreachable ({e}), falling back to direct DB")
        return _db_fetch_daily_spending(user_id)


async def get_categories() -> list[dict]:
    """Fetch all categories from Mac backend."""
    try:
        r = await _client.get("/fina/categories")
        r.raise_for_status()
        data = r.json()
        return data.get("categories", data) if isinstance(data, dict) else data
    except Exception as e:
        print(f"[MacClient] get_categories failed: {e}")
        return []


async def notify_categorized(transaction_id: str, category: str, confidence: float):
    """
    Callback: tell Mac the result of auto-categorization
    so it can update the transaction record.
    """
    try:
        r = await _client.post("/fina/callbacks/categorized", json={
            "transaction_id": str(transaction_id),
            "category": category,
            "confidence": confidence,
        })
        r.raise_for_status()
    except Exception as e:
        print(f"[MacClient] notify_categorized failed: {e}")



async def get_financial_goals(user_id: str) -> list[dict]:
    """
    Fetch user's active financial goals (savings targets, debt payoff, etc.).
    Returns: [{name, target_amount, current_saved, target_date, priority}]
    """
    try:
        r = await _client.get(f"/fina/users/{user_id}/goals")
        r.raise_for_status()
        data = r.json()
        return data.get("goals", data) if isinstance(data, dict) else data
    except Exception as e:
        print(f"[MacClient] API goals failed ({e}), falling back to DB")
        return _db_fetch_goals(user_id)


async def get_account_balances(user_id: str) -> list[dict]:
    """
    Fetch user's account balances — reuses get_user_accounts() which
    already hits GET /fina/users/{id}/accounts with DB fallback.
    Returns: [{name, balance, currency, type}]
    """
    return await get_user_accounts(user_id)


async def get_budget_preferences(user_id: str) -> Optional[dict]:
    """
    Fetch user's custom budget split preferences.
    Returns: {needs_pct, wants_pct, savings_pct} or None (default 50/30/20).
    """
    try:
        r = await _client.get(f"/fina/users/{user_id}/budget-preferences")
        r.raise_for_status()
        data = r.json()
        if data and data.get("needs_pct") is not None:
            return data
        return None
    except Exception as e:
        print(f"[MacClient] API budget prefs failed ({e}), falling back to DB")
        return _db_fetch_budget_preferences(user_id)


async def get_category_budgets(user_id: str) -> list[dict]:
    """
    Fetch user's per-category budget limits.
    Returns: [{categoryName, monthlyLimit}] or [] if none set.
    """
    try:
        r = await _client.get(f"/fina/users/{user_id}/category-budgets")
        r.raise_for_status()
        data = r.json()
        return data.get("budgets", data) if isinstance(data, dict) else data
    except Exception as e:
        print(f"[MacClient] API category budgets failed ({e}), falling back to DB")
        return _db_fetch_category_budgets(user_id)


async def get_monthly_history(user_id: str, months: int = 3) -> dict:
    """
    Fetch per-category monthly spending for the last N months + recurring expenses.
    Returns: {months: {"YYYY-MM": {category: amount}}, recurring: [{description, category, amount, occurrences}]}
    """
    try:
        r = await _client.get(f"/fina/users/{user_id}/spending/monthly-history", params={"months": months})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[MacClient] API monthly history failed ({e}), falling back to DB")
        return _db_fetch_monthly_history(user_id, months)


# ═══════════════════════════════════════════════════════════════════════════════
# DIRECT DB FALLBACK — used when Mac API is unreachable
# ═══════════════════════════════════════════════════════════════════════════════

def _db_fetch_transactions(user_id: str, limit: int = 200) -> list[dict]:
    try:
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT t.id, t.amount, t.type, t.occurred_at,
                       c.name AS category
                FROM transactions t
                LEFT JOIN categories c ON t.category_id = c.id
                WHERE t.user_id = %s
                ORDER BY t.occurred_at DESC
                LIMIT %s
            """, (user_id, limit))
            rows = cursor.fetchall()
        conn.close()
        return _convert_decimals(rows)
    except Exception as e:
        print(f"[MacClient DB Fallback] Error: {e}")
        return []


def _build_date_filter(period: str) -> tuple[str, list]:
    """
    Build a SQL WHERE clause fragment and params for date filtering.

    period: "month" | "year" | "all" | "YYYY-MM"
    Returns: (sql_fragment, params) e.g. ("AND t.occurred_at >= %s", ["2026-04-01"])
    """
    if period == "all":
        return "", []
    if period == "year":
        return "AND t.occurred_at >= DATE_FORMAT(NOW(), '%%Y-01-01')", []
    if period == "month":
        return "AND t.occurred_at >= DATE_FORMAT(NOW(), '%%Y-%%m-01')", []
    # Specific month: "YYYY-MM"
    try:
        year, month = period.split("-")
        start = f"{year}-{month}-01"
        # Next month start
        m = int(month)
        y = int(year)
        if m == 12:
            end = f"{y + 1}-01-01"
        else:
            end = f"{y}-{m + 1:02d}-01"
        return "AND t.occurred_at >= %s AND t.occurred_at < %s", [start, end]
    except Exception:
        # Invalid format — default to current month
        return "AND t.occurred_at >= DATE_FORMAT(NOW(), '%%Y-%%m-01')", []


def _db_fetch_raw_data(user_id: str, period: str = "month") -> Optional[dict]:
    date_clause, date_params = _build_date_filter(period)

    try:
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor() as cursor:
            # Currency from accounts table
            cursor.execute(
                "SELECT currency FROM accounts WHERE user_id = %s LIMIT 1",
                (user_id,)
            )
            acc_row = cursor.fetchone()
            currency = acc_row["currency"] if acc_row else "VND"

            # Income (filtered by period)
            cursor.execute(
                f"SELECT SUM(amount) as total FROM transactions t WHERE t.user_id = %s AND t.type = 'INCOME' {date_clause}",
                [user_id] + date_params,
            )
            res = cursor.fetchone()
            income = float(res["total"]) if res and res["total"] else 0.0

            # Expenses by category (filtered by period)
            cursor.execute(f"""
                SELECT c.name as category_name, SUM(t.amount) as spent
                FROM transactions t
                JOIN categories c ON t.category_id = c.id
                WHERE t.user_id = %s AND t.type = 'EXPENSE' {date_clause}
                GROUP BY c.name
            """, [user_id] + date_params)
            spending = cursor.fetchall()

            # Recent history (filtered by period)
            cursor.execute(f"""
                SELECT t.id, t.amount, t.type, t.occurred_at, c.name as category
                FROM transactions t
                LEFT JOIN categories c ON t.category_id = c.id
                WHERE t.user_id = %s {date_clause}
                ORDER BY t.occurred_at DESC LIMIT 10
            """, [user_id] + date_params)
            history = cursor.fetchall()
        conn.close()

        spending = _convert_decimals(spending)
        history = _convert_decimals(history)

        # Build computed block (matches Mac API response shape)
        total_spent = sum(s["spent"] for s in spending)
        surplus = income - total_spent
        savings_rate = (surplus / income * 100) if income > 0 else 0

        sorted_spending = sorted(spending, key=lambda x: x["spent"], reverse=True)
        top_cat = sorted_spending[0]["category_name"] if sorted_spending else "N/A"
        top_cat_spent = sorted_spending[0]["spent"] if sorted_spending else 0
        top_cat_pct = (top_cat_spent / income * 100) if income > 0 else 0

        over_budget = [
            s["category_name"] for s in spending
            if s["spent"] > income * 0.30
        ]

        return {
            "currency": currency,
            "income": income,
            "spending": spending,
            "history": history,
            "period": period,
            "computed": {
                "total_spent": total_spent,
                "surplus": surplus,
                "savings_rate_pct": round(savings_rate, 1),
                "top_category": top_cat,
                "top_category_spent": top_cat_spent,
                "top_category_pct": round(top_cat_pct, 1),
                "over_budget_categories": over_budget,
                "needs_total": total_spent,  # approximate — DB doesn't classify needs/wants
                "wants_total": 0,
                "budget_50_30_20": {
                    "needs_limit": income * 0.50,
                    "wants_limit": income * 0.30,
                    "savings_target": income * 0.20,
                },
                "budget_status": [],  # not available from raw DB
            },
        }
    except Exception as e:
        print(f"[MacClient DB Fallback] Error: {e}")
        return None


def _db_fetch_daily_spending(user_id: str) -> list[dict]:
    try:
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT DATE(t.occurred_at) AS day,
                       c.name              AS category,
                       SUM(t.amount)       AS amount
                FROM transactions t
                JOIN categories c ON t.category_id = c.id
                WHERE t.user_id = %s AND t.type = 'EXPENSE'
                GROUP BY DATE(t.occurred_at), c.name
                ORDER BY day ASC
            """, (user_id,))
            rows = cursor.fetchall()
        conn.close()
        return _convert_decimals(rows)
    except Exception as e:
        print(f"[MacClient DB Fallback] Error: {e}")
        return []


def _db_fetch_goals(user_id: str) -> list[dict]:
    try:
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT name, target_amount, current_saved, target_date, priority
                FROM financial_goals
                WHERE user_id = %s AND is_active = 1
                ORDER BY priority ASC
            """, (user_id,))
            rows = cursor.fetchall()
        conn.close()
        return _convert_decimals(rows)
    except Exception as e:
        print(f"[MacClient DB Fallback] goals: {e}")
        return []


    # _db_fetch_balances removed — get_account_balances() delegates to
    # get_user_accounts() which already has its own DB fallback path.


def _db_fetch_budget_preferences(user_id: str) -> Optional[dict]:
    try:
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT needs_pct, wants_pct, savings_pct FROM budget_preferences WHERE user_id = %s",
                (user_id,)
            )
            row = cursor.fetchone()
        conn.close()
        return row if row else None
    except Exception as e:
        print(f"[MacClient DB Fallback] budget prefs: {e}")
        return None


def _db_fetch_category_budgets(user_id: str) -> list[dict]:
    try:
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT c.name AS categoryName, cb.monthly_limit AS monthlyLimit
                FROM category_budgets cb
                JOIN categories c ON cb.category_id = c.id
                WHERE cb.user_id = %s
                ORDER BY cb.monthly_limit DESC
            """, (user_id,))
            rows = cursor.fetchall()
        conn.close()
        return _convert_decimals(rows)
    except Exception as e:
        print(f"[MacClient DB Fallback] category budgets: {e}")
        return []


def _db_fetch_monthly_history(user_id: str, months: int = 3) -> dict:
    try:
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor() as cursor:
            # Per-category monthly totals for the last N months (excluding current)
            cursor.execute("""
                SELECT DATE_FORMAT(t.occurred_at, '%%Y-%%m') AS month,
                       c.name AS category,
                       SUM(t.amount) AS spent
                FROM transactions t
                JOIN categories c ON t.category_id = c.id
                WHERE t.user_id = %s AND t.type = 'EXPENSE'
                  AND t.occurred_at >= DATE_SUB(DATE_FORMAT(NOW(), '%%Y-%%m-01'), INTERVAL %s MONTH)
                  AND t.occurred_at < DATE_FORMAT(NOW(), '%%Y-%%m-01')
                GROUP BY month, c.name
                ORDER BY month DESC
            """, (user_id, months))
            month_rows = cursor.fetchall()

            # Recurring detection: same description+category in 2+ months
            cursor.execute("""
                SELECT t.description, c.name AS category,
                       AVG(t.amount) AS amount,
                       COUNT(DISTINCT DATE_FORMAT(t.occurred_at, '%%Y-%%m')) AS occurrences
                FROM transactions t
                JOIN categories c ON t.category_id = c.id
                WHERE t.user_id = %s AND t.type = 'EXPENSE'
                  AND t.occurred_at >= DATE_SUB(NOW(), INTERVAL %s MONTH)
                  AND t.description IS NOT NULL AND t.description != ''
                GROUP BY t.description, c.name
                HAVING occurrences >= 2
                   AND MAX(t.amount) / NULLIF(MIN(t.amount), 0) <= 1.20
                ORDER BY amount DESC
            """, (user_id, months))
            recurring_rows = cursor.fetchall()
        conn.close()

        # Build months dict: {"2026-03": {"Food": 2500000, ...}, ...}
        months_dict = {}
        for row in _convert_decimals(month_rows):
            m = row["month"]
            if m not in months_dict:
                months_dict[m] = {}
            months_dict[m][row["category"]] = row["spent"]

        recurring = _convert_decimals(recurring_rows)

        return {"months": months_dict, "recurring": recurring}
    except Exception as e:
        print(f"[MacClient DB Fallback] monthly history: {e}")
        return {"months": {}, "recurring": []}
