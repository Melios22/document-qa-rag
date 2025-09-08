import json
import pickle
from datetime import datetime
from typing import Any, Dict, List

from langchain.schema import Document

from ..constant import (
    CHARS_PER_TOKEN,
    CHUNKER_MODEL,
    MAX_CHUNK_TOKENS,
    METADATA_FILE,
    OVERLAP_TOKENS,
    PROCESSED_DOCS_FILE,
    PROCESSING_SESSION_ID,
)
from ..utils.logging import get_logger

logger = get_logger("rag.preprocess.storage")


def save_documents(documents: List[Document]) -> Dict[str, Any]:
    """Save documents and return file paths"""
    if not documents:
        logger.error("No documents to save!")
        return {}

    # Define output files
    files = {
        "documents": PROCESSED_DOCS_FILE,
        "metadata": METADATA_FILE,
    }

    try:
        # Save processed documents
        with open(files["documents"], "wb") as f:
            pickle.dump(documents, f)
        logger.info(f"Documents saved: {files['documents'].name}")

        # Generate basic statistics
        token_counts = [doc.metadata.get("estimated_tokens", 0) for doc in documents]

        # Save processing metadata
        processing_metadata = {
            "processing_info": {
                "session_id": PROCESSING_SESSION_ID,
                "timestamp": datetime.now().isoformat(),
                "chunker_model": CHUNKER_MODEL,
                "total_chunks": len(documents),
            },
            "summary_stats": {
                "total_chunks": len(documents),
                "avg_tokens": (
                    round(sum(token_counts) / len(token_counts), 2)
                    if token_counts
                    else 0
                ),
                "total_tokens": sum(token_counts),
            },
            "processing_config": {
                "max_tokens": MAX_CHUNK_TOKENS,
                "overlap_tokens": OVERLAP_TOKENS,
                "chars_per_token": CHARS_PER_TOKEN,
            },
            "file_paths": {
                "documents": str(files["documents"]),
                "metadata": str(files["metadata"]),
            },
        }

        with open(files["metadata"], "w", encoding="utf-8") as f:
            json.dump(processing_metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Metadata saved: {files['metadata'].name}")

        return {
            "success": True,
            "files": files,
            "stats": processing_metadata["summary_stats"],
        }

    except Exception as e:
        logger.error(f"Error saving documents: {e}")
        return {"success": False, "error": str(e)}


def preview_documents(documents: List[Document], num_samples: int = 3):
    """Preview sample documents"""
    if not documents:
        logger.warning("No documents to preview")
        return

    logger.info(f"Document Preview ({len(documents)} total chunks):")

    preview_docs = documents[:num_samples]

    for i, doc in enumerate(preview_docs):
        logger.info(f"Chunk {i+1}:")
        logger.info(f"  ID: {doc.metadata.get('chunk_id')}")
        logger.info(f"  Source: {doc.metadata.get('source_filename')}")
        logger.info(f"  Tokens: {doc.metadata.get('estimated_tokens')}")
        logger.info(f"  Content: {doc.page_content[:120]}...")
