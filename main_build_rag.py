"""
RAG System Builder - Hybrid BGE-M3 System
==========================================

Build a hybrid vector database using BGE-M3 dense+sparse embeddings.
All configuration is loaded from config.json.
"""

import gc
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src import (
    HAS_RAG_BUILDER,
    build_hybrid_vectorstore,
    load_documents,
    require_rag_builder,
)
from src.utils.logging import build_logger as logger

# Ensure we have RAG building components
if not HAS_RAG_BUILDER:
    require_rag_builder()


def main():
    """Main RAG system building pipeline"""
    logger.milestone("Starting RAG System Building")

    try:
        # Stage 1: Load processed documents
        logger.info("Loading processed documents")
        docs = load_documents()

        if not docs:
            logger.error("No processed documents found. Run main_preprocess.py first")
            return

        logger.info(f"Loaded {len(docs)} documents for vectorization")

        # Stage 2: Build hybrid vectorstore
        logger.info("Initializing hybrid vector store construction")
        gc.collect()

        logger.info("Building hybrid vector database with BGE-M3 embeddings")
        success = build_hybrid_vectorstore(docs, batch_size=32)

        # Cleanup
        del docs
        gc.collect()
        logger.info("Memory cleanup completed")

        if success:
            logger.milestone("Hybrid RAG system built successfully - Ready for queries")
        else:
            logger.error("Failed to build hybrid vector store")

    except Exception as e:
        logger.error(f"RAG building failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
