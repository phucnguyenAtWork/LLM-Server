"""
RAG Vector Store — ChromaDB
=============================
Indexes user transactions as searchable documents.
Each document = one transaction formatted as natural language.
"""

import sys
from pathlib import Path
from datetime import datetime

import os
import httpx
import pymysql
import pymysql.cursors
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("[RAG] WARNING: chromadb not installed. Run: pip install chromadb")

# ── Mac API (primary data source) ───────────────────────────────────────────
MAC_API_URL = os.getenv("MAC_API_URL", "http://100.109.225.15:4001/api")

# ── DB config (fallback) ────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "100.109.225.15"),
    "port":     int(os.getenv("DB_PORT", "3308")),
    "user":     os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "financedb"),
    "cursorclass": pymysql.cursors.DictCursor,
}

CHROMA_PATH = str(Path(__file__).parent.parent / "chroma_db")


def _get_client():
    if not CHROMA_AVAILABLE:
        return None
    return chromadb.PersistentClient(path=CHROMA_PATH)


def _format_transaction(t: dict) -> str:
    """Convert a transaction row into a natural language sentence."""
    amount   = float(t.get("amount", 0))
    category = t.get("category", "General")
    tx_type  = t.get("type", "EXPENSE")
    date     = t.get("occurred_at", "")
    if hasattr(date, "strftime"):
        date = date.strftime("%Y-%m-%d")

    if tx_type == "INCOME":
        return f"Received {amount:,.0f} VND as income on {date}."
    return f"Spent {amount:,.0f} VND on {category} on {date}."


def _fetch_user_transactions(user_id: str) -> list[dict]:
    """Fetch transactions via Mac API, fall back to direct DB."""
    # Try Mac API first
    try:
        r = httpx.get(
            f"{MAC_API_URL}/fina/users/{user_id}/transactions",
            params={"limit": 200},
            timeout=10.0,
        )
        r.raise_for_status()
        data = r.json()
        rows = data.get("transactions", data) if isinstance(data, dict) else data
        print(f"[RAG] Fetched {len(rows)} transactions via Mac API for user {user_id}")
        return rows
    except Exception as e:
        print(f"[RAG] Mac API unreachable ({e}), falling back to DB")

    # DB fallback
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
                LIMIT 200
            """, (user_id,))
            return cursor.fetchall()
        conn.close()
    except Exception as e:
        print(f"[RAG DB Fallback] Error: {e}")
        return []


def index_user(user_id: str) -> int:
    """
    (Re)builds the ChromaDB collection for a user.
    Returns the number of documents indexed.
    """
    client = _get_client()
    if client is None:
        return 0

    transactions = _fetch_user_transactions(user_id)
    if not transactions:
        return 0

    collection_name = f"user_{user_id}_transactions"

    # Delete and recreate for a clean reindex
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    ef = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
    )

    docs, ids, metas = [], [], []
    for t in transactions:
        doc_id = str(t["id"])
        text   = _format_transaction(t)
        meta   = {
            "category": str(t.get("category", "")),
            "type":     str(t.get("type", "")),
            "amount":   float(t.get("amount", 0)),
        }
        docs.append(text)
        ids.append(doc_id)
        metas.append(meta)

    # Upsert in batches of 50
    batch_size = 50
    for i in range(0, len(docs), batch_size):
        collection.upsert(
            documents=docs[i:i+batch_size],
            ids=ids[i:i+batch_size],
            metadatas=metas[i:i+batch_size],
        )

    print(f"[RAG] Indexed {len(docs)} transactions for user {user_id}")
    return len(docs)


def get_collection(user_id: str):
    """Returns the ChromaDB collection for a user, or None."""
    client = _get_client()
    if client is None:
        return None
    collection_name = f"user_{user_id}_transactions"
    try:
        ef = embedding_functions.DefaultEmbeddingFunction()
        return client.get_collection(name=collection_name, embedding_function=ef)
    except Exception:
        return None
