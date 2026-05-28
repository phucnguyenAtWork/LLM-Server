"""
FINA Schema - shared constants, SYSTEM_PROMPT, and Pydantic models.
Imported by generate_hybrid.py, api.py, chat.py, and benchmark.py.
"""

import json
import logging
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

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

# Common LLM variants → canonical signal. Applied before whitelist validation so
# a hallucinated synonym ("overlimit") doesn't kick the whole response into the
# fallback path. Keep keys lowercase/underscored — _normalize_signals normalizes
# input first, then looks up here.
SIGNAL_ALIASES: dict[str, str] = {
    "overlimit": "over_budget",
    "over_limit": "over_budget",
    "exceeded": "over_budget",
    "budget_exceeded": "over_budget",
    "category_exceeded": "category_budget_exceeded",
    "category_over_budget": "category_budget_exceeded",
    "under_budget": "within_budget",
    "underbudget": "within_budget",
    "at_risk": "goal_at_risk",
    "goal_risk": "goal_at_risk",
    "savings_low": "below_savings_target",
    "savings_high": "above_savings_target",
    "spending_up": "spending_up_mom",
    "spending_down": "spending_down_mom",
    "anomaly": "anomaly_detected",
    "no_budgets": "no_category_budgets",
    "fixed_costs_high": "high_fixed_costs",
}


def _normalize_signals(raw) -> list[str]:
    """Pre-validation cleanup of model-emitted signals.

    Lowercases/underscores each entry, applies SIGNAL_ALIASES, drops anything
    still outside VALID_SIGNALS, and de-dupes while preserving order. Every
    transformation (alias hit or unknown drop) is logged at WARNING so the
    event is visible per user query without changing the schema contract.
    """
    if not raw:
        return []
    if not isinstance(raw, (list, tuple)):
        logger.warning("[signal-normalize] dropping non-list signals payload: %r", raw)
        return []

    cleaned: list[str] = []
    for sig in raw:
        original = sig
        norm = str(sig).strip().lower().replace("-", "_").replace(" ", "_")
        mapped = SIGNAL_ALIASES.get(norm, norm)
        if mapped in VALID_SIGNALS:
            if mapped != original:
                logger.warning("[signal-normalize] aliased %r -> %r", original, mapped)
            cleaned.append(mapped)
        else:
            logger.warning("[signal-normalize] dropped unknown signal %r", original)

    seen: set[str] = set()
    deduped: list[str] = []
    for s in cleaned:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped


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

The "signals" array MUST contain only values from this exact list (no inventions,
no synonyms, no plurals): "anomaly_detected", "over_budget", "within_budget",
"goal_at_risk", "below_savings_target", "above_savings_target",
"category_budget_exceeded", "spending_up_mom", "spending_down_mom",
"high_fixed_costs", "no_category_budgets", "deficit", "on_track".

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
- For GENERAL budget questions ("will I overshoot my budget?", "am I on track?", "how am
  I doing this month?"), you MUST address EVERY category listed in CATEGORY BUDGET STATUS
  — do not cherry-pick one. For each one, copy verbatim the spent / limit / % used and the
  pre-computed "projected month-end" number from the same line. NEVER invent budget limits
  or projection numbers — every number you state must come directly from CATEGORY BUDGET
  STATUS. When the user asks "by how much" or "by month-end", quote the projected overage
  or buffer per category, then quote PROJECTED TOTAL OVERAGE BY MONTH-END as the headline.
  Never answer a multi-budget question by addressing only one category.
- For category-SPECIFIC questions ("how's my Food budget?"), only discuss budgets that
  appear in CATEGORY BUDGET STATUS. If the user names a category that is not listed,
  state plainly that no budget is set for that category and offer to create one.
- HARD RULE: If a category does not appear in CATEGORY BUDGET STATUS, you MUST NOT
  state any budget number (limit, "your X budget is N VND", etc.) for it. Never invent
  a limit. Say "no budget set for X" and offer to create one. This rule overrides any
  pattern from training where you may have answered with a fabricated limit.
- TIME-WINDOW RULE: When you reference a time window in the message, use the exact
  PERIOD label given at the top of FINANCIAL CONTEXT (e.g. "April 2026 (last month)",
  "last 3 months", "May 2026 (current month)"). Do NOT say "this month" if the data
  is from a different period. If the period is "April 2026 (last month)", say "in
  April" or "last month". If the period is rolling like "last 3 months", say "over
  the last 3 months". Never paraphrase the period in a way that contradicts the
  PERIOD label.
- If CATEGORY BUDGET STATUS says "OVER LIMIT by X", call X an overage or overspend,
  never "savings". Savings means unspent surplus, not an amount above budget.
- PERIOD-SCOPE RULE: Only reason about evidence whose date falls inside the PERIOD
  window declared at the top of FINANCIAL CONTEXT. If a retrieved RAG row is dated
  outside that window, ignore it for "this period" claims even if its amount is larger.
- NO-RECOMPUTE RULE: When FINANCIAL CONTEXT already states a total, surplus, percentage,
  or projection, quote that exact number. Do not re-derive it from line items.
- EVIDENCE-CONFLICT RULE: If FINANCIAL CONTEXT and the retrieved RAG block disagree on
  a number or category, trust FINANCIAL CONTEXT and briefly note the discrepancy.

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

Advice & general questions:
- Some questions are conceptual or advisory and are NOT answered by FINANCIAL CONTEXT
  (e.g. "rent vs buy", "what is compound interest", "how do I stop impulse buying",
  "do I need insurance", career and habit questions). For these, give sound,
  conventional financial guidance in a warm, hedged tone ("generally...", "a common
  rule is..."). Answer the question on its own terms — do not deflect into a spending
  summary the user did not ask for.
- Reference the user's own numbers only when they are directly relevant to the advice.
  NEVER invent a personal amount, balance, budget, or target that is not in FINANCIAL
  CONTEXT. Generic reference figures are fine (e.g. "3–6 months of expenses",
  "the 50/30/20 rule"), because they are general guidance, not claims about the user.
- If a piece of advice would need a personal number you do not have, say so and ask
  for it rather than guessing ("I don't have your rent on file — what are you paying?").
- Out of scope: do NOT predict markets or asset prices (stocks, crypto, gold) and do
  NOT give buy/sell tips, and do NOT offer to move money on the user's behalf
  (transfers, payments, lending). Politely decline and redirect to budgeting,
  tracking, saving, and planning, which is what you can actually help with.
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

    @field_validator("signals", mode="before")
    @classmethod
    def _normalize_signals_before_validation(cls, v):
        return _normalize_signals(v)

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


def parse_model_output_with_status(
    raw: str, *, log_failure: bool = True
) -> tuple[Optional[ModelOutput], dict]:
    """Parse and return (output_or_None, status_dict).

    Status dict shape:
      {"ok": bool, "error": str | None, "error_type": str | None, "raw_excerpt": str}

    Used by api.py to surface schema validation errors to the caller (and into
    evaluation logs) without silently swallowing them via fallback_output.
    """
    text = _strip_code_fences(raw.strip())
    try:
        data = json.loads(text)
        output = ModelOutput(**data)
        return output, {"ok": True, "error": None, "error_type": None, "raw_excerpt": text[:200]}
    except json.JSONDecodeError as e:
        if log_failure:
            logger.warning("Model output is not valid JSON: %s - raw: %.200s", e, raw)
        return None, {
            "ok": False,
            "error": f"invalid_json: {e}",
            "error_type": "json_decode",
            "raw_excerpt": text[:200],
        }
    except Exception as e:
        if log_failure:
            logger.warning("Model output failed schema validation: %s - raw: %.200s", e, raw)
        return None, {
            "ok": False,
            "error": str(e),
            "error_type": "schema_validation",
            "raw_excerpt": text[:200],
        }


def fallback_output(raw: str) -> ModelOutput:
    """Return a safe fallback ModelOutput when parsing fails."""
    return ModelOutput(
        kind=Kind.ANALYSIS,
        message=raw.strip()[:500] if raw.strip() else "I encountered an issue processing your request.",
        action=None,
        signals=[],
        needs_clarification=False,
    )


# ── Action-safety guard (§6 of the thesis plan) ──────────────────────────────

# Hard upper bound on a single transaction/budget amount in VND. The dataset
# tops out around 10^8 VND/month; anything beyond 10^10 is almost certainly a
# unit error, hallucination, or model corruption.
ACTION_AMOUNT_HARD_CAP_VND = 10_000_000_000


def validate_action_safety(
    action: Optional["AssistantAction"],
    *,
    known_categories: Optional[set[str]] = None,
    require_confirmation_for_delete: bool = True,
) -> tuple[bool, Optional[str]]:
    """Pre-execution safety check for actions emitted by the model.

    Returns (ok, reason). Callers should drop the action when ok is False and
    surface the reason to the user / evaluation log. Independent of the Pydantic
    schema validators (those enforce shape, this enforces operational safety).
    """
    if action is None:
        return True, None

    args = action.arguments
    action_type = action.type

    if args.amount is not None:
        if args.amount < 0:
            return False, f"negative amount rejected: {args.amount}"
        if args.amount > ACTION_AMOUNT_HARD_CAP_VND:
            return False, f"amount exceeds safety cap: {args.amount} > {ACTION_AMOUNT_HARD_CAP_VND}"

    if args.category is not None and args.category not in VALID_CATEGORIES:
        return False, f"unknown category: {args.category}"

    if known_categories is not None and args.category is not None and args.category not in known_categories:
        return False, f"category not in user context: {args.category}"

    if args.alert_threshold is not None and not 0.0 <= args.alert_threshold <= 1.0:
        return False, f"alert_threshold out of range: {args.alert_threshold}"

    if args.period is not None and args.period not in VALID_BUDGET_PERIODS:
        return False, f"invalid period: {args.period}"

    if action_type in (ActionType.DELETE_TRANSACTION, ActionType.DELETE_BUDGET):
        if require_confirmation_for_delete:
            if args.transaction_ref is None and args.budget_ref is None and args.item is None:
                return False, "delete actions require an explicit reference (transaction_ref, budget_ref, or item)"

    return True, None
