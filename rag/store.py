"""
RAG Vector Store — ChromaDB
=============================
Indexes user transactions as searchable documents.
Each document = one transaction formatted as natural language.
"""

import sys
from pathlib import Path
from datetime import datetime

import pymysql
import pymysql.cursors

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("[RAG] WARNING: chromadb not installed. Run: pip install chromadb")

DB_CONFIG = {
    "host": "100.109.225.15",
    "port": 3308,
    "user": "root",
    "password": "rootpass",
    "database": "financedb",
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


def _fetch_user_transactions(user_id: int) -> list[dict]:
    conn = pymysql.connect(**DB_CONFIG)
    try:
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
    finally:
        conn.close()


def index_user(user_id: int) -> int:
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
            documents=ids[i:i+batch_size],
            ids=ids[i:i+batch_size],
            metadatas=metas[i:i+batch_size],
        )
        # ChromaDB needs documents not IDs as the text
        collection.upsert(
            documents=docs[i:i+batch_size],
            ids=ids[i:i+batch_size],
            metadatas=metas[i:i+batch_size],
        )

    print(f"[RAG] Indexed {len(docs)} transactions for user {user_id}")
    return len(docs)


def get_collection(user_id: int):
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
