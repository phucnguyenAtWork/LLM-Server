"""
RAG Vector Store — ChromaDB
=============================
Per-user transaction collections backed by ChromaDB's local PersistentClient.
Embeddings use Chroma's DefaultEmbeddingFunction (ONNX MiniLM, CPU) so they
do not contend with the 8-bit Qwen on GPU.

Transaction fetching and formatting live elsewhere (mac_client +
rag.retriever._transaction_text) so this module is purely a vector index.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("[RAG] WARNING: chromadb not installed. Run: pip install chromadb")

CHROMA_PATH = os.getenv(
    "CHROMA_PATH",
    str(Path(__file__).parent.parent / "chroma_db"),
)
LOG_PREFIX = "[RAG.store]"

_client = None
_embedding_function = None
_collection_cache: dict[str, Any] = {}


def _client_singleton():
    global _client, _embedding_function
    if not CHROMA_AVAILABLE:
        return None, None
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        _embedding_function = embedding_functions.DefaultEmbeddingFunction()
    return _client, _embedding_function


def _collection_name(user_id: str) -> str:
    return f"user_{user_id}_transactions"


def get_collection(user_id: str):
    """Return (or create) the user's collection. None if Chroma unavailable."""
    client, ef = _client_singleton()
    if client is None:
        return None
    name = _collection_name(user_id)
    cached = _collection_cache.get(name)
    if cached is not None:
        return cached
    coll = client.get_or_create_collection(name=name, embedding_function=ef)
    _collection_cache[name] = coll
    return coll


def _tx_id(tx: dict[str, Any]) -> str:
    raw = tx.get("id") or tx.get("transaction_id") or tx.get("transactionId")
    return str(raw) if raw is not None else ""


def _updated_at(tx: dict[str, Any]) -> str:
    val = (
        tx.get("updated_at")
        or tx.get("updatedAt")
        or tx.get("occurred_at")
        or tx.get("occurredAt")
        or ""
    )
    return str(val)


def signature(transactions: list[dict[str, Any]]) -> tuple[int, str]:
    """A cheap fingerprint used to skip re-indexing when nothing changed."""
    if not transactions:
        return (0, "")
    max_updated = max((_updated_at(t) for t in transactions), default="")
    return (len(transactions), max_updated)


def _build_metadata(tx: dict[str, Any]) -> dict[str, Any]:
    amount = tx.get("amount")
    try:
        amount_val = float(amount) if amount is not None else 0.0
    except (TypeError, ValueError):
        amount_val = 0.0
    date = (
        tx.get("occurred_at")
        or tx.get("occurredAt")
        or ""
    )
    if hasattr(date, "strftime"):
        date_iso = date.strftime("%Y-%m-%d")
    else:
        date_iso = str(date)[:10]
    return {
        "transaction_id": _tx_id(tx),
        "type": str(tx.get("type", "")),
        "category": str(tx.get("category") or tx.get("categoryName") or ""),
        "amount": amount_val,
        "currency": str(tx.get("currency", "VND")),
        "date_iso": date_iso,
    }


def upsert_transactions(
    user_id: str,
    transactions: Iterable[dict[str, Any]],
    formatter,
) -> int:
    """
    Upsert transactions into the user's collection. `formatter(tx)` produces
    the natural-language document text. Returns the number upserted.
    """
    coll = get_collection(user_id)
    if coll is None:
        return 0

    docs: list[str] = []
    ids: list[str] = []
    metas: list[dict[str, Any]] = []
    for tx in transactions:
        tx_id = _tx_id(tx)
        if not tx_id:
            continue
        docs.append(formatter(tx))
        ids.append(tx_id)
        metas.append(_build_metadata(tx))

    if not docs:
        return 0

    batch = 64
    for i in range(0, len(docs), batch):
        coll.upsert(
            documents=docs[i : i + batch],
            ids=ids[i : i + batch],
            metadatas=metas[i : i + batch],
        )
    print(f"{LOG_PREFIX} upsert user_id={user_id} count={len(docs)}")
    return len(docs)


def query(user_id: str, text: str, top_k: int) -> dict[str, Any] | None:
    """Vector query. Returns the raw Chroma response or None on failure."""
    coll = get_collection(user_id)
    if coll is None:
        return None
    try:
        return coll.query(query_texts=[text], n_results=top_k)
    except Exception as exc:
        print(f"{LOG_PREFIX} query.error user_id={user_id} error={exc}")
        return None
