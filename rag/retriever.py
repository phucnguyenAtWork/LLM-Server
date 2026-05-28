"""
RAG Retriever
=============

True vector retrieval over a user's transactions, with a lexical fallback
for environments where ChromaDB is unavailable or the vector query fails.

Design notes:
- Auto-indexes lazily on the first query, gated by a (count, max_updated_at)
  signature so steady-state cost is one query embedding + an ANN lookup.
- Output shape (RetrievalResult / RetrievedSource / [Sx] citations) is kept
  identical to the previous lexical implementation so api.py / chat.py do
  not need to change.
- Inline [Sx] markers in the prompt body are gated by RAG_INLINE_CITATIONS
  because the v8 LoRA was not trained on them; the structured `sources`
  list is always returned to the API regardless.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import mac_client
from rag import store

DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
MAX_TRANSACTIONS = int(os.getenv("RAG_MAX_TRANSACTIONS", "300"))
AUTO_INDEX = os.getenv("RAG_AUTO_INDEX", "true").lower() == "true"
INLINE_CITATIONS = os.getenv("RAG_INLINE_CITATIONS", "false").lower() == "true"
RAG_LOG_PREFIX = "[RAG]"

STOPWORDS = {
    "a", "an", "and", "are", "can", "could", "do", "for", "from", "how",
    "i", "is", "it", "me", "my", "of", "on", "or", "should", "the", "this",
    "to", "what", "with", "you",
}

# Per-process cache of (count, max_updated_at) signatures keyed by user_id.
_signature_cache: dict[str, tuple[int, str]] = {}


@dataclass
class RetrievedSource:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    context: str
    sources: list[RetrievedSource]
    status: str
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "context": self.context,
            "status": self.status,
            "error": self.error,
            "retrieved_ids": [s.id for s in self.sources],
            "retrieved_sources": [
                {"id": s.id, "text": s.text, "metadata": s.metadata}
                for s in self.sources
            ],
        }


# ── Formatting helpers (shared by vector + lexical paths) ──────────────────

def _format_amount(amount: Any, currency: str = "VND") -> str:
    try:
        return f"{float(amount):,.0f}".replace(",", ".") + f" {currency}"
    except Exception:
        return f"{amount} {currency}".strip()


def _format_date(value: Any) -> str:
    if value is None:
        return "unknown date"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    text = str(value)
    return text[:10] if len(text) >= 10 else text


def _transaction_text(tx: dict[str, Any]) -> str:
    tx_type = str(tx.get("type", "EXPENSE")).upper()
    amount = _format_amount(tx.get("amount", 0), tx.get("currency", "VND"))
    category = tx.get("category") or tx.get("categoryName") or "Uncategorized"
    description = tx.get("description") or category
    date = _format_date(tx.get("occurred_at") or tx.get("occurredAt"))
    if tx_type == "INCOME":
        return f"Received {amount} from {description} on {date}."
    return f"Spent {amount} on {category} ({description}) on {date}."


def _build_context(sources: list[RetrievedSource]) -> str:
    """Render evidence in a v8-friendly form (close to FINANCIAL CONTEXT)."""
    if not sources:
        return ""
    lines = ["RECENT RELEVANT TRANSACTIONS:"]
    for s in sources:
        if INLINE_CITATIONS:
            lines.append(f"- [{s.id}] {s.text}")
        else:
            lines.append(f"- {s.text}")
    return "\n" + "\n".join(lines) + "\n"


def _to_source(idx: int, text: str, metadata: dict[str, Any], score: float) -> RetrievedSource:
    meta = dict(metadata or {})
    meta["score"] = round(float(score), 4)
    return RetrievedSource(id=f"S{idx}", text=text, metadata=meta)


# ── Lexical fallback (only used when Chroma is unavailable or errors) ─────

def _tokens(text: str) -> set[str]:
    return {
        tok for tok in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(tok) > 2 and tok not in STOPWORDS
    }


def _lexical_score(query_tokens: set[str], tx: dict[str, Any], text: str) -> float:
    haystack = " ".join(
        str(v) for v in [
            text, tx.get("category"), tx.get("categoryName"),
            tx.get("description"), tx.get("type"),
        ] if v is not None
    )
    overlap = len(query_tokens & _tokens(haystack))
    amount = float(tx.get("amount") or 0)
    return overlap + 0.15 + min(amount / 10_000_000, 0.35)


def _lexical_fallback(
    transactions: list[dict[str, Any]], query: str, top_k: int
) -> list[RetrievedSource]:
    query_tokens = _tokens(query)
    scored = []
    for i, tx in enumerate(transactions):
        text = _transaction_text(tx)
        score = _lexical_score(query_tokens, tx, text) if query_tokens else 0.0
        scored.append((score, i, tx, text))
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    selected = scored[: max(1, min(top_k, len(scored)))]
    return [
        _to_source(
            idx,
            text,
            {
                "transaction_id": str(tx.get("id", "")),
                "type": tx.get("type"),
                "category": tx.get("category") or tx.get("categoryName"),
                "amount": tx.get("amount"),
                "occurred_at": _format_date(tx.get("occurred_at") or tx.get("occurredAt")),
            },
            score,
        )
        for idx, (score, _i, tx, text) in enumerate(selected, start=1)
    ]


# ── Vector retrieval ───────────────────────────────────────────────────────

def _maybe_reindex(user_id: str, transactions: list[dict[str, Any]]) -> bool:
    """Upsert into Chroma only when the signature changed. Returns True on success."""
    if not AUTO_INDEX:
        return True
    sig = store.signature(transactions)
    if _signature_cache.get(user_id) == sig:
        return True
    try:
        store.upsert_transactions(user_id, transactions, _transaction_text)
        _signature_cache[user_id] = sig
        return True
    except Exception as exc:
        print(f"{RAG_LOG_PREFIX} reindex.error user_id={user_id} error={exc}")
        return False


def _period_epoch_range(period: Any) -> tuple[int, int] | None:
    """Convert a PeriodSpec to an inclusive-start, exclusive-end epoch range.

    Returns None when period is missing or has no usable date bounds (e.g.
    period="all"), so callers can skip the filter and preserve legacy
    behaviour.
    """
    if period is None:
        return None
    start = getattr(period, "start", None)
    end = getattr(period, "end", None)
    if start is None or end is None:
        return None
    try:
        start_ts = int(datetime(start.year, start.month, start.day).timestamp())
        end_ts = int(datetime(end.year, end.month, end.day).timestamp())
    except (AttributeError, ValueError, TypeError):
        return None
    return start_ts, end_ts


def _vector_query(
    user_id: str,
    query: str,
    top_k: int,
    period: Any = None,
) -> list[RetrievedSource] | None:
    where: dict[str, Any] | None = None
    rng = _period_epoch_range(period)
    if rng is not None:
        start_ts, end_ts = rng
        where = {
            "$and": [
                {"date_epoch": {"$gte": start_ts}},
                {"date_epoch": {"$lt": end_ts}},
            ]
        }
    raw = store.query(user_id, query, top_k, where=where)
    if not raw:
        return None
    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]
    if not docs:
        return []
    sources = []
    for idx, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
        score = 1.0 - float(dist) if dist is not None else 0.0
        sources.append(_to_source(idx, doc, meta or {}, score))
    return sources


# ── Public API ─────────────────────────────────────────────────────────────

def _filter_by_period(
    transactions: list[dict[str, Any]], period: Any
) -> list[dict[str, Any]]:
    """In-process date filter for the lexical fallback. Returns the input
    unchanged when no usable period range is supplied."""
    rng = _period_epoch_range(period)
    if rng is None:
        return transactions
    start_ts, end_ts = rng
    out: list[dict[str, Any]] = []
    for tx in transactions:
        raw_date = (
            tx.get("date_iso")
            or tx.get("occurred_at")
            or tx.get("occurredAt")
            or ""
        )
        if hasattr(raw_date, "strftime"):
            iso = raw_date.strftime("%Y-%m-%d")
        else:
            iso = str(raw_date)[:10]
        try:
            ts = int(datetime.strptime(iso, "%Y-%m-%d").timestamp())
        except (ValueError, TypeError):
            continue
        if start_ts <= ts < end_ts:
            out.append(tx)
    return out


async def retrieve_context_with_sources(
    user_id: str,
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    transactions: list[dict[str, Any]] | None = None,
    period: Any = None,
) -> RetrievalResult:
    """
    Retrieve source-cited transaction evidence for one user query.

    Caller may pass `transactions` to avoid a duplicate Mac API round-trip
    when the structured FINANCIAL CONTEXT path already fetched them.
    Never raises: returns an empty context with a non-ok status instead.
    """
    print(f"{RAG_LOG_PREFIX} stage=retrieve.start user_id={user_id} top_k={top_k} query={query!r}")

    if transactions is None:
        try:
            transactions = await mac_client.get_transactions(user_id, limit=MAX_TRANSACTIONS)
        except Exception as exc:
            print(f"{RAG_LOG_PREFIX} stage=retrieve.error user_id={user_id} error={exc}")
            return RetrievalResult(context="", sources=[], status="error", error=str(exc))

    if not transactions:
        print(f"{RAG_LOG_PREFIX} stage=retrieve.empty user_id={user_id}")
        return RetrievalResult(context="", sources=[], status="indexed_empty")

    transactions = transactions[:MAX_TRANSACTIONS]

    if store.CHROMA_AVAILABLE and _maybe_reindex(user_id, transactions):
        sources = _vector_query(user_id, query, top_k, period=period)
        if sources is not None:
            if not sources:
                return RetrievalResult(context="", sources=[], status="indexed_empty")
            print(f"{RAG_LOG_PREFIX} stage=select.done status=ok_vector selected={len(sources)}")
            return RetrievalResult(
                context=_build_context(sources),
                sources=sources,
                status="ok_vector",
            )

    scoped = _filter_by_period(transactions, period) if period is not None else transactions
    if not scoped:
        scoped = transactions  # period filter wiped everything; fall back to unfiltered
    sources = _lexical_fallback(scoped, query, top_k)
    print(f"{RAG_LOG_PREFIX} stage=select.done status=fallback_lexical selected={len(sources)}")
    return RetrievalResult(
        context=_build_context(sources),
        sources=sources,
        status="fallback_lexical",
    )


async def retrieve_context(user_id: str, query: str) -> str:
    """Backward-compatible helper returning only the context string."""
    return (await retrieve_context_with_sources(user_id, query)).context
