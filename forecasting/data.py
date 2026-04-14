import os
import numpy as np
import pandas as pd
import httpx
import pymysql
import pymysql.cursors
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ── Mac API (primary data source) ───────────────────────────────────────────
MAC_API_URL = os.getenv("MAC_API_URL", "http://100.109.225.15:4001/api")

# ── DB config (fallback when Mac API unreachable) ────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "100.109.225.15"),
    "port":     int(os.getenv("DB_PORT", "3308")),
    "user":     os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "financedb"),
    "cursorclass": pymysql.cursors.DictCursor,
}

SEQ_LEN = 30  # days of history used as input


# ── Minimal MinMaxScaler (no scikit-learn dependency) ────────────────────────

class MinMaxScaler:
    """Scales each feature to [0, 1]. Safe against zero-range columns."""

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        self.range_ = np.where(self.max_ - self.min_ == 0, 1.0, self.max_ - self.min_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.min_) / self.range_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.range_ + self.min_

    def to_dict(self) -> dict:
        return {
            "min_":   self.min_.tolist(),
            "max_":   self.max_.tolist(),
            "range_": self.range_.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MinMaxScaler":
        s = cls()
        s.min_   = np.array(d["min_"])
        s.max_   = np.array(d["max_"])
        s.range_ = np.array(d["range_"])
        return s


# ── Data fetch (Mac API → DB fallback) ───────────────────────────────────────

def fetch_user_daily_spending(user_id: str) -> pd.DataFrame:
    """
    Returns a long-format DataFrame with columns [day, category, amount]
    for all EXPENSE transactions of the given user, ordered by day asc.

    Tries Mac API first, falls back to direct DB.
    """
    rows = _mac_fetch_daily_spending(user_id)
    if rows is None:
        rows = _db_fetch_daily_spending(user_id)

    if not rows:
        return pd.DataFrame(columns=["day", "category", "amount"])

    df = pd.DataFrame(rows)
    df["day"]    = pd.to_datetime(df["day"])
    df["amount"] = df["amount"].astype(float)
    return df


def _mac_fetch_daily_spending(user_id: str) -> Optional[list[dict]]:
    """Fetch daily spending from Mac API (sync)."""
    try:
        r = httpx.get(
            f"{MAC_API_URL}/fina/users/{user_id}/spending/daily",
            timeout=10.0,
        )
        r.raise_for_status()
        data = r.json()
        rows = data.get("rows", data) if isinstance(data, dict) else data
        print(f"[Data] User {user_id} daily spending via Mac API: {len(rows)} rows")
        return rows
    except Exception as e:
        print(f"[Data] Mac API unreachable ({e}), falling back to DB")
        return None


def _db_fetch_daily_spending(user_id: str) -> list[dict]:
    """Fetch daily spending directly from DB (fallback)."""
    try:
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT
                    DATE(t.occurred_at) AS day,
                    c.name              AS category,
                    SUM(t.amount)       AS amount
                FROM transactions t
                JOIN categories c ON t.category_id = c.id
                WHERE t.user_id = %s
                  AND t.type    = 'EXPENSE'
                GROUP BY DATE(t.occurred_at), c.name
                ORDER BY day ASC
            """, (user_id,))
            rows = cursor.fetchall()
        conn.close()
        return _convert_decimals(rows)
    except Exception as e:
        print(f"[Data DB Fallback] Error: {e}")
        return []


def _convert_decimals(obj):
    from decimal import Decimal
    if isinstance(obj, list):
        return [_convert_decimals(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _convert_decimals(v) for k, v in obj.items()}
    if isinstance(obj, Decimal):
        return float(obj)
    return obj


# ── Pivot & align ─────────────────────────────────────────────────────────────

def pivot_to_daily_matrix(
    df: pd.DataFrame,
    known_columns: Optional[list] = None
) -> tuple[pd.DataFrame, list]:
    """
    Converts long-format df → wide daily matrix (dates × features).
    Appends a '_total' column.

    If known_columns is provided (inference mode), the result is aligned
    to that exact column order — missing categories become 0, new ones
    are dropped. This keeps training/inference feature spaces identical.

    Returns (wide_df, column_list).
    """
    if df.empty:
        if known_columns:
            empty = pd.DataFrame(columns=known_columns)
            return empty, known_columns
        return pd.DataFrame(), []

    wide = df.pivot_table(
        index="day", columns="category", values="amount",
        aggfunc="sum", fill_value=0.0
    )
    wide.index = pd.to_datetime(wide.index)

    # Fill calendar gaps so every day has a row
    full_range = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
    wide = wide.reindex(full_range, fill_value=0.0)

    if known_columns is not None:
        # Align to training columns (exclude _total — we recompute it)
        train_cats = [c for c in known_columns if c != "_total"]
        wide = wide.reindex(columns=train_cats, fill_value=0.0)

    wide["_total"] = wide.sum(axis=1)
    columns = list(wide.columns)
    return wide, columns


# ── Sequence builder ──────────────────────────────────────────────────────────

def build_sequences(matrix: np.ndarray, seq_len: int = SEQ_LEN) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates sliding-window (X, y) pairs.
    X[i] : window of seq_len days   → shape (seq_len, F)
    y[i] : mean daily spending of the single next day → shape (F,)

    For sparse users (fewer than seq_len+1 rows), the matrix is padded
    with leading zeros and one sequence is returned.
    """
    T, F = matrix.shape

    if T < seq_len + 1:
        pad = np.zeros((seq_len + 1 - T, F), dtype=np.float32)
        matrix = np.vstack([pad, matrix])
        T = matrix.shape[0]

    X, y = [], []
    for i in range(T - seq_len):
        X.append(matrix[i : i + seq_len])
        y.append(matrix[i + seq_len])   # next single day (mean daily target)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
