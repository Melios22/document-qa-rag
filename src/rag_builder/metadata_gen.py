import datetime
import json
from typing import Any, Dict, List

from langchain.schema import Document

from .. import (
    BATCH_SIZE,
    COLLECTION_NAME,
    DEVICE,
    EMBED_MODEL_ID,
    ENCODE_KWARGS,
    MAX_RETRIES,
    MILVUS_METADATA_FILE,
    MILVUS_URI,
    MODEL_KWARGS,
    PROCESSED_DOCS_FILE,
    RAG_CONFIG_FILE,
    RAG_METADATA_FILE,
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
                "config_file": str(RAG_CONFIG_FILE),
                "metadata_file": str(RAG_METADATA_FILE),
            },
        }

        # Save main config
        config_path = RAG_CONFIG_FILE
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # Save detailed metadata
        metadata = {
            "vectorstore_metadata": {
                "collection_name": COLLECTION_NAME,
                "embedding_model": EMBED_MODEL_ID,
                "embedding_dimension": embedding_dim,
                "database_uri": str(MILVUS_URI),
                "document_count": len(documents),
                "processing_method": "batch_processing",
                "batch_size": BATCH_SIZE,
                "client_type": "PyMilvus",
            },
            "document_statistics": {
                "total_documents": len(documents),
                "total_characters": sum(len(doc.page_content) for doc in documents),
                "average_length": sum(len(doc.page_content) for doc in documents)
                / len(documents),
            },
            "processing_summary": {
                "status": "completed" if client else "failed",
                "timestamp": datetime.datetime.now().isoformat(),
                "model_loaded": embedding_model is not None,
                "client_created": client is not None,
            },
        }

        metadata_path = MILVUS_METADATA_FILE
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Configuration saved to {config_path} and {metadata_path}")

        return config

    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        return {}
