"""
LSTM Forecasting Engine — Training Script
=========================================
Train a per-user spending LSTM and save artifacts to models/user_{id}/.

Usage:
    python -m forecasting.train_lstm --user_id 1
    python -m forecasting.train_lstm --all_users
    python -m forecasting.train_lstm --user_id 1 --epochs 100 --seq_len 30

NOTE: Run this script out-of-band (not by api.py). Retrain whenever a user
accumulates significantly more transaction history.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pymysql
import torch
import torch.nn as nn

# Allow running as: python -m forecasting.train_lstm
sys.path.insert(0, str(Path(__file__).parent.parent))

from forecasting.data import (
    DB_CONFIG, SEQ_LEN, MinMaxScaler,
    fetch_user_daily_spending, pivot_to_daily_matrix, build_sequences
)
from forecasting.model import SpendingLSTM

MODELS_DIR = Path(__file__).parent.parent / "models"
MIN_REAL_DAYS = 7   # skip training if user has fewer real data days
HIDDEN_SIZE   = 64
NUM_LAYERS    = 2


def get_all_user_ids() -> list[int]:
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT DISTINCT user_id FROM transactions WHERE type = 'EXPENSE'")
            rows = cursor.fetchall()
    finally:
        conn.close()
    return [r["user_id"] for r in rows]


def train_for_user(user_id: int, epochs: int, seq_len: int):
    print(f"\n[LSTM Train] User {user_id}")

    # ── 1. Fetch & build matrix ──────────────────────────────────────────────
    df = fetch_user_daily_spending(user_id)
    if df.empty:
        print(f"  Skipped — no expense data.")
        return

    wide, columns = pivot_to_daily_matrix(df)
    matrix = wide.values.astype(np.float32)

    real_days = (wide != 0).any(axis=1).sum()
    if real_days < MIN_REAL_DAYS:
        print(f"  Skipped — only {real_days} days with real data (minimum {MIN_REAL_DAYS}).")
        return

    print(f"  Days: {len(matrix)} | Features: {len(columns)} | Columns: {columns}")

    # ── 2. Normalize ─────────────────────────────────────────────────────────
    scaler = MinMaxScaler().fit(matrix)
    matrix_norm = scaler.transform(matrix)

    # ── 3. Build sequences ───────────────────────────────────────────────────
    X, y = build_sequences(matrix_norm, seq_len=seq_len)
    print(f"  Sequences — X: {X.shape}, y: {y.shape}")

    # ── 4. Train/val split ───────────────────────────────────────────────────
    if len(X) >= 10:
        split = max(1, int(0.8 * len(X)))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        do_validate = True
    else:
        X_train, y_train = X, y
        do_validate = False
        print("  Not enough sequences for validation split — training on all data.")

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    if do_validate:
        X_val_t = torch.tensor(X_val)
        y_val_t  = torch.tensor(y_val)

    # ── 5. Model & optimizer ─────────────────────────────────────────────────
    input_size = X.shape[2]
    model     = SpendingLSTM(input_size=input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn   = nn.MSELoss()

    best_val_loss = float("inf")
    best_state    = None

    # ── 6. Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred  = model(X_train_t)
        loss  = loss_fn(pred, y_train_t)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            if do_validate:
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val_t)
                    val_loss = loss_fn(val_pred, y_val_t).item()
                print(f"  Epoch {epoch:>4}/{epochs} | train={loss.item():.6f} | val={val_loss:.6f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                print(f"  Epoch {epoch:>4}/{epochs} | train={loss.item():.6f}")

    # ── 7. Save artifacts ────────────────────────────────────────────────────
    user_dir = MODELS_DIR / f"user_{user_id}"
    user_dir.mkdir(parents=True, exist_ok=True)

    weights_to_save = best_state if best_state is not None else model.state_dict()
    torch.save(weights_to_save, user_dir / "lstm.pt")

    metadata = {
        "input_size":  input_size,
        "hidden_size": HIDDEN_SIZE,
        "num_layers":  NUM_LAYERS,
        "seq_len":     seq_len,
        "columns":     columns,
        "scaler":      scaler.to_dict(),
    }
    (user_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"  Saved → {user_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Train LSTM forecasting model(s).")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--user_id",    type=int, help="Train for a specific user")
    group.add_argument("--all_users",  action="store_true", help="Train for every user in DB")
    parser.add_argument("--epochs",    type=int, default=50,    help="Training epochs (default: 50)")
    parser.add_argument("--seq_len",   type=int, default=SEQ_LEN, help=f"Input sequence length in days (default: {SEQ_LEN})")
    args = parser.parse_args()

    if args.all_users:
        user_ids = get_all_user_ids()
        print(f"[LSTM Train] Found {len(user_ids)} user(s): {user_ids}")
    else:
        user_ids = [args.user_id]

    for uid in user_ids:
        train_for_user(uid, epochs=args.epochs, seq_len=args.seq_len)

    print("\n[LSTM Train] Done.")


if __name__ == "__main__":
    main()
