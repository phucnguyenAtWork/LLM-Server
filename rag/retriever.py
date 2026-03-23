"""
RAG Retriever
==============
Public API: retrieve_context(user_id, query) -> str

Searches the user's indexed transactions for the most relevant entries
given the user's query, then formats them as a context string for the LLM.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.store import get_collection, index_user, CHROMA_AVAILABLE

TOP_K = 5  # number of relevant documents to retrieve


def retrieve_context(user_id: int, query: str) -> str:
    """
    Returns a formatted string of relevant transactions for the LLM context.
    Falls back to an empty string if ChromaDB is unavailable or not indexed.
    """
    if not CHROMA_AVAILABLE:
        return ""

    collection = get_collection(user_id)

    # Auto-index if collection doesn't exist yet
    if collection is None:
        indexed = index_user(user_id)
        if indexed == 0:
            return ""
        collection = get_collection(user_id)
        if collection is None:
            return ""

    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(TOP_K, collection.count()),
        )

        docs = results.get("documents", [[]])[0]
        if not docs:
            return ""

        lines = "\n".join(f"  - {doc}" for doc in docs)
        return f"\nRELEVANT TRANSACTION HISTORY:\n{lines}\n"

    except Exception as e:
        print(f"[RAG] Retrieval error: {e}")
        return ""
