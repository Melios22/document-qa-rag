import datetime
import json
from typing import Any, Dict, List

from langchain.schema import Document

from ..constant import (
    BATCH_SIZE,
    COLLECTION_NAME,
    DEVICE,
    EMBED_MODEL_ID,
    ENCODE_KWARGS,
    MAX_RETRIES,
    MILVUS_URI,
    MODEL_KWARGS,
    PROCESSED_DOCS_FILE,
)
from ..utils.logging import get_logger

logger = get_logger("rag.builder.metadata")


def save_config(documents: List[Document], embedding_model, client) -> Dict[str, Any]:
    """Save enhanced RAG configuration for PyMilvus"""
    try:
        # Get embedding dimension
        test_embedding = embedding_model.embed_query("test")
        embedding_dim = len(test_embedding)

        # Enhanced configuration
        config = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model_config": {
                "embedding_model": EMBED_MODEL_ID,
                "embedding_dimension": embedding_dim,
                "device": DEVICE,
                "model_kwargs": MODEL_KWARGS,
                "encode_kwargs": ENCODE_KWARGS,
            },
            "vectorstore_config": {
                "database_uri": str(MILVUS_URI),
                "collection_name": COLLECTION_NAME,
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "normalization": True,
            },
            "processing_config": {
                "batch_size": BATCH_SIZE,
                "max_retries": MAX_RETRIES,
                "total_documents": len(documents),
            },
            "file_paths": {
                "input_documents": str(PROCESSED_DOCS_FILE),
                "vector_database": str(MILVUS_URI),
            },
        }

        # Just log the completion, no need to save additional metadata files
        logger.info(f"Configuration generated for {len(documents)} documents")
        logger.info(f"Using collection: {COLLECTION_NAME}")
        logger.info(f"Embedding model: {EMBED_MODEL_ID}")

        return config

    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        return {}
