"""
LSTM Forecasting Engine — Inference
=====================================
Public API: forecast_user(user_id) -> dict

Returns weekly and monthly spending predictions per category.
Falls back gracefully if no trained model exists for the user.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pymysql
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from forecasting.data import (
    DB_CONFIG, SEQ_LEN, MinMaxScaler,
    fetch_user_daily_spending, pivot_to_daily_matrix
)
from forecasting.model import SpendingLSTM

MODELS_DIR = Path(__file__).parent.parent / "models"

# Module-level cache: user_id → (model, metadata, mtime)
_cache: dict = {}


def _load_model(user_id: int):
    """Load (and cache) model + metadata for a user. Returns (model, meta) or (None, None)."""
    user_dir   = MODELS_DIR / f"user_{user_id}"
    model_path = user_dir / "lstm.pt"
    meta_path  = user_dir / "metadata.json"

    if not model_path.exists() or not meta_path.exists():
        return None, None

    mtime = model_path.stat().st_mtime
    if user_id in _cache and _cache[user_id]["mtime"] == mtime:
        return _cache[user_id]["model"], _cache[user_id]["meta"]

    meta = json.loads(meta_path.read_text())
    lstm = SpendingLSTM(
        input_size=meta["input_size"],
        hidden_size=meta["hidden_size"],
        num_layers=meta["num_layers"]
    )
    lstm.load_state_dict(torch.load(model_path, map_location="cpu"))
    lstm.eval()

    _cache[user_id] = {"model": lstm, "meta": meta, "mtime": mtime}
    return lstm, meta


def _cold_start_fallback(user_id: int) -> dict:
    """
    Returns a naive forecast when no trained model exists.
    Uses the user's total historical expense * 1.05 as monthly estimate.
    """
    try:
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT SUM(amount) as total FROM transactions WHERE user_id = %s AND type = 'EXPENSE'",
                (user_id,)
            )
            row = cursor.fetchone()
        conn.close()
        total = float(row["total"]) if row and row["total"] else 0.0
    except Exception as e:
        print(f"[LSTM Fallback] DB error: {e}")
        total = 0.0

    monthly_estimate = total * 1.05
    return {
        "weekly":  {"total": round(monthly_estimate / 4.3, 2), "by_category": {}},
        "monthly": {"total": round(monthly_estimate, 2),        "by_category": {}},
        "confidence": 50,
        "source": "fallback"
    }


def forecast_user(user_id: int) -> dict:
    """
    Predicts next-week and next-month spending for a user.

    Returns:
        {
            "weekly":  {"total": float, "by_category": {str: float}},
            "monthly": {"total": float, "by_category": {str: float}},
            "confidence": int,   # 0-100
            "source": "lstm" | "fallback"
        }
    """
    lstm, meta = _load_model(user_id)
    if lstm is None:
        print(f"[LSTM Predict] No model for user {user_id} — using fallback.")
        return _cold_start_fallback(user_id)

    columns  = meta["columns"]
    seq_len  = meta["seq_len"]
    scaler   = MinMaxScaler.from_dict(meta["scaler"])
    input_size = meta["input_size"]

    # ── Fetch user's recent spending ──────────────────────────────────────────
    df = fetch_user_daily_spending(user_id)
    wide, _ = pivot_to_daily_matrix(df, known_columns=columns)

    if wide.empty:
        print(f"[LSTM Predict] No recent data for user {user_id} — using fallback.")
        return _cold_start_fallback(user_id)

    matrix = wide.values.astype(np.float32)

    # Ensure exactly seq_len rows (pad with zeros if needed)
    if len(matrix) < seq_len:
        pad    = np.zeros((seq_len - len(matrix), input_size), dtype=np.float32)
        matrix = np.vstack([pad, matrix])
    else:
        matrix = matrix[-seq_len:]

    # ── Inference ─────────────────────────────────────────────────────────────
    matrix_norm = scaler.transform(matrix)
    x = torch.tensor(matrix_norm[np.newaxis, :, :], dtype=torch.float32)  # (1, seq_len, F)

    with torch.no_grad():
        pred_norm = lstm(x).squeeze(0).numpy()  # (F,) — predicted mean daily scaled values

    pred_daily = scaler.inverse_transform(pred_norm.reshape(1, -1)).flatten()
    pred_daily = np.clip(pred_daily, 0, None)   # no negative spending

    # ── Build result dict ─────────────────────────────────────────────────────
    total_idx    = columns.index("_total")
    by_category  = {
        col: round(float(pred_daily[i]) * 7, 2)
        for i, col in enumerate(columns) if col != "_total"
    }
    weekly_total = round(float(pred_daily[total_idx]) * 7, 2)
    monthly_by_category = {k: round(v / 7 * 30, 2) for k, v in by_category.items()}
    monthly_total = round(float(pred_daily[total_idx]) * 30, 2)

    return {
        "weekly": {
            "total":       weekly_total,
            "by_category": by_category
        },
        "monthly": {
            "total":       monthly_total,
            "by_category": monthly_by_category
        },
        "confidence": 80,
        "source": "lstm"
    }
