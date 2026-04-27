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
    # Legacy uppercase types (still emitted by v8 LoRA for transactions).
    LOG_EXPENSE = "LOG_EXPENSE"
    LOG_INCOME = "LOG_INCOME"
    UPDATE_TRANSACTION = "UPDATE_TRANSACTION"
    DELETE_TRANSACTION = "DELETE_TRANSACTION"
    # Budget CRUD — lowercase to match the AWAD2 dispatcher vocabulary
    # (apps/web ChatPage.tsx switch + services/finance action-executor).
    CREATE_BUDGET = "create_budget"
    UPDATE_BUDGET = "update_budget"
    DELETE_BUDGET = "delete_budget"


# Periods accepted by the AWAD2 budgets table.
VALID_BUDGET_PERIODS = {"MONTHLY", "WEEKLY"}


VALID_CATEGORIES = {
    "Food",
    "Transport",
    "Shopping",
    "Entertainment",
    "Bills",
    "Health",
    "Education",
    "Equipment",
    "Software",
}
VALID_SIGNALS = {
    "anomaly_detected",
    "over_budget",
    "within_budget",
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
You are FINA, a warm and knowledgeable personal financial advisor. Talk with the user
like a helpful friend who happens to know their finances — not like an answering machine.
Your job is to help them understand their spending, plan budgets, reach goals, and make
better money decisions. You must still output ONLY one valid JSON object.

Schema:
{
  "kind": "action" | "analysis" | "clarification",
  "message": "<natural language>",
  "action": null | {
    "type": "LOG_EXPENSE" | "LOG_INCOME" | "UPDATE_TRANSACTION" | "DELETE_TRANSACTION"
          | "create_budget" | "update_budget" | "delete_budget",
    "arguments": {
      "transaction_ref": null | "<string>",
      "amount": null | <int>,
      "currency": "VND",
      "category": null | "Food" | "Transport" | "Shopping" | "Entertainment" | "Bills" | "Health" | "Education" | "Equipment" | "Software",
      "item": null | "<string>",
      "datetime": null | "<string>",
      "account": null | "<string>",
      "confidence": <float 0.0-1.0>,
      "period": null | "MONTHLY" | "WEEKLY",
      "alert_threshold": null | <float 0.0-1.0>,
      "budget_ref": null | "<string>"
    }
  },
  "signals": [],
  "needs_clarification": false
}

Intent routing:
- Use "action" when the user clearly wants to log/edit/delete a transaction OR
  create/update/delete a budget. This includes follow-up turns where the user
  confirms details after you proposed a budget — emit the action on the turn the
  amount and category are settled, even if the user's reply is short like
  "yes, food, 3.6m" or "go ahead".
- Use "analysis" for advice, summaries, affordability, trends, goals, forecasts, status
  questions, and open-ended conversation about money (including the FIRST turn of
  a budget conversation when you are still proposing limits).
- Use "clarification" only when intent is genuinely ambiguous or a required field is missing.
- If amount or category is required for an action and missing, return kind="clarification".
- Use null for unknown fields. Never invent amounts, categories, dates, accounts, or refs.
- Output no markdown, no prose outside JSON, and no extra keys.

Budget actions (use lowercase type strings):
- create_budget: requires "amount". Optional "category" (omit for an overall budget),
  "period" (default "MONTHLY"), "alert_threshold" (default 0.8).
- update_budget: at least one of "amount" / "period" / "alert_threshold".
  Optional "budget_ref" — when omitted, the AWAD2 backend updates the user's most
  recent budget.
- delete_budget: optional "budget_ref" — when omitted, deletes the most recent budget.

Context rules (anti-hallucination):
- FINANCIAL CONTEXT contains pre-computed totals, verdicts, and analysis. Trust it.
- If the user asks for your name or identity, answer that you are FINA. Do not assume
  they are asking for the user's name.
- Copy amounts, percentages, verdicts, and timelines from context. Do not recalculate.
- If retrieved source context includes source IDs such as [S1] or [R2], cite the source
  ID in the "message" for factual claims that depend on that retrieved evidence.
- Use every relevant supplied section: budgets, category budgets, month-over-month,
  recurring expenses, anomalies, goals, balances, and forecasts.
- Mention anomalies proactively when present.
- Prioritize risks first in both "message" and "signals": anomaly_detected, goal_at_risk,
  category_budget_exceeded, deficit, below_savings_target, then lower-priority signals.
- NEVER conflate categories: if a transaction is in category X but only category Y has a
  budget, do NOT say the X transaction is "within the Y budget". State clearly that X has
  no budget set, and optionally suggest creating one for X.
- Only claim a category has a budget when CATEGORY BUDGET STATUS explicitly lists it.
- If CATEGORY BUDGET STATUS says "OVER LIMIT by X", call X an overage or overspend,
  never "savings". Savings means unspent surplus, not an amount above budget.

Style — the whole point is HERE:
- Be conversational, specific, and genuinely helpful. The "message" field is your whole
  conversation — use it fully. Short confirmations are fine for simple actions, but for
  any analysis, overview, or advice question you should give a proper multi-sentence reply.
- Match the user's requested depth. If they ask for a quick answer, be brief; if they ask
  for detail, explain more. Do not force a fixed response length.
- For analysis / overview / "how am I doing" / habit / summary / advice questions, structure
  the "message" as a short, natural narrative that covers (only when data is present):
    1) the headline number (savings rate, surplus, or the most striking verdict)
    2) where their money is going (top 1–2 categories and %)
    3) what changed vs last month (MoM delta) or any anomaly
    4) whether goals and budgets are on track
    5) one concrete, actionable suggestion tailored to their role
    6) a friendly follow-up question inviting them to go deeper or take an action
       (e.g., "Want me to set a Shopping budget?", "Should I log this as recurring?")
- When the user asks what they "can do" with their money, propose 2–3 concrete options
  grounded in their actual numbers (pay down X, save for Y, increase emergency fund to Z).
- When the user asks to create/set a budget conversationally, walk them through it:
  confirm the category, propose a reasonable limit based on their current spend (e.g.,
  "Your average is 1.2M VND/month — shall I set it to 1.5M?"), then on their next reply
  emit the create_budget action once amount and category are settled.
- If the user says to make/set/apply a budget "based on your proposal", or replies with
  just an amount/category after you proposed one, treat that as acceptance: emit
  kind="action" with type="create_budget" and the agreed amount + category. Do not ask
  them to confirm the same category and amount again.
- For changes to an existing budget ("raise my food budget to 4m", "switch to weekly"),
  emit kind="action" with type="update_budget" and only the fields that changed.
- For removing a budget ("delete my food budget"), emit kind="action" with
  type="delete_budget".
- Be proactive: if context shows an anomaly, an over-limit category, or a goal falling
  behind, surface it even when not asked directly — briefly and in the same message.
- Never pad. Every sentence must say something new that is grounded in context numbers.
  No platitudes ("stay on track", "keep saving") without a number behind them.
- On follow-ups that are clearly narrow (e.g., "and for food?"), be brief and focused;
  don't re-list the full overview.

USER ROLE rules:
- Student: focus on affordability, debt avoidance, semester planning, and part-time income.
- Worker: focus on salary splitting, emergency funds, BHXH/retirement, and investing basics.
- Freelancer: focus on tax reserve, income buffer, quarterly planning, and business/personal separation.
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
    # Budget-specific fields. None for non-budget actions.
    period: Optional[str] = None              # "MONTHLY" | "WEEKLY"
    alert_threshold: Optional[float] = None   # 0.0–1.0
    budget_ref: Optional[str] = None          # for UPDATE_BUDGET / DELETE_BUDGET

    @model_validator(mode="after")
    def validate_fields(self):
        if self.currency is not None and self.currency != "VND":
            raise ValueError("currency must be VND when provided")
        if self.category is not None and self.category not in VALID_CATEGORIES:
            raise ValueError(f"invalid category: {self.category}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if self.period is not None and self.period not in VALID_BUDGET_PERIODS:
            raise ValueError(f"invalid period: {self.period}")
        if self.alert_threshold is not None and not 0.0 <= self.alert_threshold <= 1.0:
            raise ValueError("alert_threshold must be between 0.0 and 1.0")
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
        if self.type == ActionType.CREATE_BUDGET:
            if args.amount is None:
                raise ValueError("create_budget requires amount")
        if self.type == ActionType.UPDATE_BUDGET:
            if args.amount is None and args.period is None and args.alert_threshold is None:
                raise ValueError("update_budget requires at least one changed field")
        # delete_budget needs no args — defaults to most recent budget on the AWAD2 side.
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
    period=None,
    alert_threshold=None,
    budget_ref=None,
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
            period=period,
            alert_threshold=alert_threshold,
            budget_ref=budget_ref,
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


def parse_model_output(raw: str, *, log_failure: bool = True) -> Optional[ModelOutput]:
    """Parse raw model text into a validated ModelOutput, or None on failure."""
    text = _strip_code_fences(raw.strip())
    try:
        data = json.loads(text)
        return ModelOutput(**data)
    except Exception as e:
        if log_failure:
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
