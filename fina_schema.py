"""
FINA Schema - shared constants, SYSTEM_PROMPT, and Pydantic models.
Imported by generate_hybrid.py, api.py, chat.py, and benchmark.py.
"""

import json
import logging
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger("fina")


class Kind(str, Enum):
    ACTION = "action"
    ANALYSIS = "analysis"
    CLARIFICATION = "clarification"


class ActionType(str, Enum):
    LOG_EXPENSE = "LOG_EXPENSE"
    LOG_INCOME = "LOG_INCOME"
    UPDATE_TRANSACTION = "UPDATE_TRANSACTION"
    DELETE_TRANSACTION = "DELETE_TRANSACTION"


VALID_CATEGORIES = {
    "Food",
    "Transport",
    "Shopping",
    "Entertainment",
    "Bills",
    "Health",
    "Education",
}
VALID_SIGNALS = {
    "anomaly_detected",
    "over_budget",
    "goal_at_risk",
    "below_savings_target",
    "above_savings_target",
    "category_budget_exceeded",
    "spending_up_mom",
    "spending_down_mom",
    "high_fixed_costs",
    "no_category_budgets",
    "deficit",
    "on_track",
}

SYSTEM_PROMPT = """\
You are FINA, a financial AI assistant. Output ONLY one valid JSON object.

Schema:
{
  "kind": "action" | "analysis" | "clarification",
  "message": "<natural language>",
  "action": null | {
    "type": "LOG_EXPENSE" | "LOG_INCOME" | "UPDATE_TRANSACTION" | "DELETE_TRANSACTION",
    "arguments": {
      "transaction_ref": null | "<string>",
      "amount": null | <int>,
      "currency": "VND",
      "category": null | "Food" | "Transport" | "Shopping" | "Entertainment" | "Bills" | "Health" | "Education",
      "item": null | "<string>",
      "datetime": null | "<string>",
      "account": null | "<string>",
      "confidence": <float 0.0-1.0>
    }
  },
  "signals": [],
  "needs_clarification": false
}

Rules:
- Use "action" only when the user clearly wants to log, edit, or delete a transaction.
- Use "analysis" for advice, summaries, affordability, trends, goals, forecasts, and status questions.
- Use "clarification" when intent is ambiguous or required fields are missing.
- Use null for unknown fields. Never invent amounts, categories, dates, accounts, or transaction refs.
- If amount or category is required for an action and missing, return kind="clarification".
- Keep all natural language inside "message".
- Output no markdown, no prose outside JSON, and no extra keys.

Context rules:
- FINANCIAL CONTEXT contains pre-computed totals, verdicts, and analysis.
- Copy amounts, percentages, verdicts, and timelines from context. Do not recalculate them.
- Use the supplied context sections when they exist: budgets, category budgets, month-over-month, recurring expenses, anomalies, goals, balances, and forecasts.
- Mention anomalies proactively when present.
- Prioritize the most important risks first in both "message" and "signals": anomaly_detected, goal_at_risk, category_budget_exceeded, deficit, below_savings_target, then lower-priority signals.

USER ROLE rules:
- Student: focus on affordability, debt avoidance, semester planning, and part-time income.
- Worker: focus on salary splitting, emergency funds, BHXH/retirement, and investing basics.
- Freelancer: focus on tax reserve, income buffer, quarterly planning, and business/personal separation.

Style:
- "message" must be concise, specific, and grounded in the provided numbers.
- On follow-ups, be brief and avoid repeating the full overview.
"""


class ActionArguments(BaseModel):
    transaction_ref: Optional[str] = None
    amount: Optional[int] = None
    currency: Optional[str] = "VND"
    category: Optional[str] = None
    item: Optional[str] = None
    datetime: Optional[str] = None
    account: Optional[str] = None
    confidence: float = 0.0

    @model_validator(mode="after")
    def validate_fields(self):
        if self.currency is not None and self.currency != "VND":
            raise ValueError("currency must be VND when provided")
        if self.category is not None and self.category not in VALID_CATEGORIES:
            raise ValueError(f"invalid category: {self.category}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        return self


class AssistantAction(BaseModel):
    type: ActionType
    arguments: ActionArguments = Field(default_factory=ActionArguments)

    @model_validator(mode="after")
    def validate_action(self):
        args = self.arguments
        if self.type in (ActionType.LOG_EXPENSE, ActionType.LOG_INCOME):
            if args.amount is None or args.item is None:
                raise ValueError("log actions require amount and item")
        if self.type == ActionType.LOG_EXPENSE and args.category is None:
            raise ValueError("LOG_EXPENSE requires category")
        if self.type == ActionType.UPDATE_TRANSACTION:
            if args.amount is None and args.category is None and args.item is None:
                raise ValueError("UPDATE_TRANSACTION requires at least one changed field")
        if self.type == ActionType.DELETE_TRANSACTION:
            if args.transaction_ref is None and args.item is None:
                raise ValueError("DELETE_TRANSACTION requires a reference or item")
        return self


class ModelOutput(BaseModel):
    kind: Kind
    message: str
    action: Optional[AssistantAction] = None
    signals: list[str] = Field(default_factory=list)
    needs_clarification: bool = False

    @model_validator(mode="after")
    def validate_output(self):
        unknown_signals = [signal for signal in self.signals if signal not in VALID_SIGNALS]
        if unknown_signals:
            raise ValueError(f"invalid signals: {unknown_signals}")
        if not self.message or not self.message.strip():
            raise ValueError("message must be non-empty")
        if self.kind == Kind.CLARIFICATION:
            if not self.needs_clarification:
                raise ValueError("clarification outputs must set needs_clarification=true")
            if self.action is not None:
                raise ValueError("clarification outputs must not include action")
        if self.kind == Kind.ACTION:
            if self.action is None:
                raise ValueError("action outputs must include action payload")
        if self.kind == Kind.ANALYSIS and self.needs_clarification:
            raise ValueError("analysis outputs must not require clarification")
        return self


def build_action(
    action_type: str,
    *,
    amount=None,
    currency="VND",
    category=None,
    item=None,
    datetime_str=None,
    account=None,
    confidence=0.95,
    transaction_ref=None,
) -> dict:
    """Build a validated action dict for inclusion in a model response."""
    action = AssistantAction(
        type=action_type,
        arguments=ActionArguments(
            transaction_ref=transaction_ref,
            amount=int(amount) if amount is not None else None,
            currency=currency,
            category=category,
            item=item,
            datetime=datetime_str,
            account=account,
            confidence=round(confidence, 2),
        ),
    )
    return action.model_dump(mode="json")


def build_response(kind: str, message: str, *, action=None, signals=None, needs_clarification=False) -> str:
    """Build a validated JSON string for an assistant response."""
    payload = ModelOutput(
        kind=kind,
        message=message,
        action=action,
        signals=signals or [],
        needs_clarification=needs_clarification,
    )
    return json.dumps(payload.model_dump(mode="json"), ensure_ascii=False)


def _strip_code_fences(text: str) -> str:
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def parse_model_output(raw: str) -> Optional[ModelOutput]:
    """Parse raw model text into a validated ModelOutput, or None on failure."""
    text = _strip_code_fences(raw.strip())
    try:
        data = json.loads(text)
        return ModelOutput(**data)
    except Exception as e:
        logger.warning("Failed to parse model output: %s - raw: %.200s", e, raw)
        return None


def fallback_output(raw: str) -> ModelOutput:
    """Return a safe fallback ModelOutput when parsing fails."""
    return ModelOutput(
        kind=Kind.ANALYSIS,
        message=raw.strip()[:500] if raw.strip() else "I encountered an issue processing your request.",
        action=None,
        signals=[],
        needs_clarification=False,
    )
