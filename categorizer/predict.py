"""
Hybrid Categorizer — Inference
================================
Public API: categorize(description) -> (category, confidence, method)

Strategy:
  1. Run NLP pipeline to clean the description
  2. Try rule-based matching first (confidence = 1.0 on match)
  3. Fall back to ML classifier if no rule matches
  4. Final fallback: "General"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nlp.pipeline import normalize_transaction
from categorizer.rules import rule_predict
from categorizer.classifier import TransactionClassifier

# Lazy-loaded ML classifier
_clf: TransactionClassifier | None = None

def _get_classifier() -> TransactionClassifier:
    global _clf
    if _clf is None:
        _clf = TransactionClassifier.load()
    return _clf


def categorize(description: str) -> dict:
    """
    Hybrid categorization pipeline.

    Returns:
        {
            category   : str    — predicted category name
            confidence : float  — 0.0 to 1.0
            method     : str    — "rule" | "ml" | "fallback"
            merchant   : str    — normalized merchant name
            cleaned    : str    — cleaned description
        }
    """
    normalized = normalize_transaction(description)
    cleaned    = normalized["cleaned"]
    merchant   = normalized["merchant"]

    # ── Step 1: Rule-based ────────────────────────────────────────────────────
    rule_cat, rule_conf = rule_predict(cleaned)
    if rule_cat:
        return {
            "category":   rule_cat,
            "confidence": rule_conf,
            "method":     "rule",
            "merchant":   merchant,
            "cleaned":    cleaned,
        }

    # ── Step 2: ML classifier ─────────────────────────────────────────────────
    clf = _get_classifier()
    if clf.is_trained:
        ml_cat, ml_conf = clf.predict(cleaned)
        if ml_conf >= 0.4:
            return {
                "category":   ml_cat,
                "confidence": ml_conf,
                "method":     "ml",
                "merchant":   merchant,
                "cleaned":    cleaned,
            }

    # ── Step 3: Fallback ──────────────────────────────────────────────────────
    return {
        "category":   "General",
        "confidence": 0.0,
        "method":     "fallback",
        "merchant":   merchant,
        "cleaned":    cleaned,
    }
