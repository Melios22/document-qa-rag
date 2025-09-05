"""
RAG Retrieval Models Loader
===========================

This module handles loading and initialization of models needed for RAG retrieval:
- BGE-M3 embedding model for vector search
- Milvus vector database connection
- BGE-Reranker-V2-M3 for document reranking

Memory optimization is applied throughout to prevent OOM issues.
"""

import gc
from typing import Any, Tuple

from FlagEmbedding import FlagReranker
from pymilvus import MilvusClient

from ..constant import COLLECTION_NAME  # Milvus collection name
from ..constant import DEVICE  # Computing device
from ..constant import EMBED_MODEL_ID  # BGE-M3 model identifier
from ..constant import FP16  # Use FP16 precision for reranker
from ..constant import MILVUS_URI  # Milvus database URI
from ..constant import RERANKER_MODEL_ID  # BGE reranker model identifier
from ..rag_builder.vector_bge_m3 import BGEM3Encoder
from ..utils.logging import get_logger

logger = get_logger("rag.retriever.model_loader")


def load_retrieval_models() -> Tuple[Any, Any, Any]:
    """
    Load and initialize all models needed for RAG retrieval with PyMilvus.

    Returns:
        Tuple containing: Milvus client, embedding model, and reranker model
    """
    logger.info("Loading retrieval models")

    # Clear memory before loading models
    gc.collect()

    # Load BGE-M3 embedding model
    logger.info("Loading BGE-M3 embedding model")
    embedding_model = BGEM3Encoder(
        model=EMBED_MODEL_ID,
        device="cpu",  # Force CPU to avoid memory conflicts
        normalize_embeddings=True,
        use_fp16=False,  # Set to False to avoid dtype issues
        max_length=512,
        batch_size=16,  # Smaller batch size for memory efficiency
    )

    # Test embedding model with minimal example
    test_result = embedding_model.encode("test")
    logger.milestone(
        f"BGE-M3 model loaded! Dimension: {test_result['dense_vecs'].shape[1]}"
    )

    # Clean up test embedding immediately
    del test_result
    gc.collect()

    # Initialize Milvus client
    logger.info("Connecting to Milvus vector database")
    client = MilvusClient(uri=str(MILVUS_URI))

    # Test connection
    if client.has_collection(COLLECTION_NAME):
        logger.info(f"Milvus connected! Collection '{COLLECTION_NAME}' found")
    else:
        logger.warning(
            f"Collection '{COLLECTION_NAME}' not found - may need to build first"
        )

    # Load BGE reranker model
    logger.info("Loading BGE-Reranker-V2-M3")
    reranker_model = FlagReranker(
        RERANKER_MODEL_ID,
        normalize=True,
        use_fp16=False,  # Disable FP16 to avoid memory issues
        device="cpu",  # Force CPU to avoid CUDA OOM
    )
    logger.milestone("BGE Reranker loaded successfully!")

    # Final memory cleanup
    gc.collect()

    logger.milestone("All retrieval models loaded successfully!")
    return client, embedding_model, reranker_model
