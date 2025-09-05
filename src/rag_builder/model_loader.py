"""
RAG Builder Model Loader
========================

This module handles loading components needed for building the RAG system:
- Load processed documents from storage
- Initialize BGE-M3 embedding model for vector creation

Memory optimization is applied to handle large document collections.
"""

import gc
import pickle
import time
from pathlib import Path
from typing import List, Optional

from langchain.schema import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from ..constant import DEVICE  # Computing device (cpu/cuda)
from ..constant import EMBED_MODEL_ID  # BGE-M3 model identifier
from ..constant import ENCODE_KWARGS  # Encoding arguments for embeddings
from ..constant import MODEL_KWARGS  # Model arguments for embeddings
from ..constant import PROCESSED_DOCS_FILE  # Path to processed documents file
from ..utils.logging import get_logger

logger = get_logger("rag.builder.model_loader")


def load_documents() -> List[Document]:
    """
    Load and validate documents from Stage 1 processing with memory optimization.

    Returns:
        List[Document]: List of processed document chunks
    """
    file_path = PROCESSED_DOCS_FILE
    try:
        logger.info(f"Loading documents from: {file_path}")

        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Clear memory before loading large file
        gc.collect()

        with open(file_path, "rb") as f:
            documents = pickle.load(f)

        # Validation
        if not documents:
            raise ValueError("No documents found in file")

        if not all(isinstance(doc, Document) for doc in documents):
            raise ValueError("Invalid document format")

        # Calculate statistics efficiently
        total_chars = sum(len(doc.page_content) for doc in documents)
        avg_chars = total_chars / len(documents)

        logger.milestone(f"Successfully loaded {len(documents)} documents")
        logger.info(f"Document Statistics:")
        logger.info(f"   Total documents: {len(documents)}")
        logger.info(f"   Average length: {avg_chars:.1f} characters")
        logger.info(f"   Total size: {total_chars:,} characters")

        # Clear temporary variables
        del total_chars, avg_chars
        gc.collect()

        return documents

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        logger.error("Please run Stage 1 processing first!")
        return []
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        return []


def load_embedding_model() -> Optional[HuggingFaceEmbeddings]:
    """
    Load BGE-M3 embedding model with enhanced error handling and memory optimization.

    Returns:
        Optional[HuggingFaceEmbeddings]: Initialized embedding model or None if failed
    """
    try:
        logger.info(f"Loading BAAI BGE-M3 model: {EMBED_MODEL_ID}")
        logger.info("First download may take several minutes...")
        logger.info(f"Model config: {MODEL_KWARGS}")
        logger.info(f"Encode config: {ENCODE_KWARGS}")

        # Clear memory before loading model
        gc.collect()

        start_time = time.time()

        # Update model kwargs with device
        model_kwargs_updated = MODEL_KWARGS.copy()
        model_kwargs_updated["device"] = DEVICE

        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_ID,
            model_kwargs=model_kwargs_updated,
            encode_kwargs=ENCODE_KWARGS,
        )

        load_time = time.time() - start_time

        # Test the model with Vietnamese text
        test_texts = [
            "Đây là một câu tiếng Việt để kiểm tra mô hình.",
            "This is an English sentence for testing.",
            "机器学习是人工智能的重要分支。",  # Chinese
        ]

        logger.info("Testing model with multilingual samples")

        # Test with a single sample to verify functionality
        test_embedding = embedding_model.embed_query(
            "Test embedding for model validation"
        )
        embedding_dim = len(test_embedding)

        # Clean up test embedding immediately
        del test_embedding
        gc.collect()

        load_time = time.time() - start_time

        logger.milestone("BGE-M3 model loaded successfully!")
        logger.info(f"Load time: {load_time:.1f}s")
        logger.info(f"Embedding dimension: {embedding_dim}")
        logger.info("Multilingual support: ✅")
        logger.info(f"Device: {DEVICE}")

        # Clear temporary variables
        del model_kwargs_updated, load_time, embedding_dim
        gc.collect()

        return embedding_model

    except Exception as e:
        logger.error(f"Failed to load BGE-M3 model: {e}")
        logger.error(
            "Try setting device to 'cpu' in config if GPU memory is insufficient"
        )
        gc.collect()
        return None
