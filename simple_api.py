"""
Simple RAG API for Streamlit
============================

Just the essential functions you need.
"""

import os
from typing import Dict, List

# Configure for clean operation
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from src.constant import COLLECTION_NAME
from src.rag_retriever.model_loader import load_retrieval_models
from src.rag_retriever.retriever import VietnameseRAG

# Global instances - loaded once
_rag = None
_models_loaded = False


def _ensure_models_loaded():
    """Load models once and keep them in memory."""
    global _rag, _models_loaded

    if not _models_loaded:
        print("Loading RAG models (one time only)...")
        client, embedding_model, reranker_model = load_retrieval_models()
        _rag = VietnameseRAG(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
        )
        _models_loaded = True
        print("✅ Models loaded successfully!")

    return _rag


def ask_question(question: str) -> Dict:
    """
    Ask a question and get an answer.

    Returns:
        {
            "answer": str,
            "sources": list,
            "confidence": float
        }
    """
    try:
        rag = _ensure_models_loaded()
        result = rag.generate_answer(question)
        return result
    except Exception as e:
        return {"answer": f"Error: {e}", "sources": [], "confidence": 0.0}


def search_documents(query: str, top_k: int = 3) -> List[Dict]:
    """
    Search for documents.

    Returns:
        List of {
            "content": str,
            "source": str,
            "score": float,
            "metadata": dict
        }
    """
    try:
        rag = _ensure_models_loaded()
        results = rag.search(query, initial_k=10, final_k=top_k)

        formatted = []
        for doc, score in results:
            formatted.append(
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source_filename", "Unknown"),
                    "score": score,
                    "metadata": doc.metadata,
                }
            )

        return formatted
    except Exception as e:
        return []


# Test if run directly
if __name__ == "__main__":
    print("Testing RAG...")
    result = ask_question("CNN là gì?")
    print(f"Answer: {result['answer'][:100]}...")
