"""
Vietnamese RAG System - Essential Exports
==========================================

Minimal package for direct imports.
"""

# Direct access only - avoid circular imports
pass

# Configuration and constants
from src.constant import (
    BATCH_SIZE,
    CHARS_PER_TOKEN,
    CHUNKER_MODEL,
    COLLECTION_NAME,
    DEFAULT_K,
    DENSE_INDEX_CONFIG,
    DENSE_INDEX_FALLBACK_CONFIG,
    DENSE_SEARCH_FALLBACK_PARAMS,
    DENSE_SEARCH_PARAMS,
    DEVICE,
    EMBED_MODEL_ID,
    EMBEDDING_DIM,
    ENCODE_KWARGS,
    FP16,
    LOG_FILE_BUILD,
    LOG_FILE_PREPROCESS,
    MAX_CONTEXT_LENGTH,
    MAX_RETRIES,
    MAX_TOKENS,
    METADATA_FILE,
    MILVUS_DOCKER_URI,
    MILVUS_METADATA_FILE,
    MILVUS_URI,
    MODEL_KWARGS,
    OUTPUT_DIR,
    OVERLAP_TOKENS,
    PDF_INPUT_DIR,
    PROCESSED_DOCS_FILE,
    PROCESSING_SESSION_ID,
    PROMPT,
    RAG_CONFIG_FILE,
    RAG_METADATA_FILE,
    RERANK_TOP_K,
    RERANKER_MODEL_ID,
    RRF_K,
    SAFETY_MARGIN,
    SIMILARITY_THRESHOLD,
    SPARSE_INDEX_CONFIG,
    SPARSE_SEARCH_PARAMS,
    USE_DOCKER_MILVUS,
)

# Document processing components
from src.preprocess import (
    ChunkMetadataGenerator,
    TextCleaner,
    preview_documents,
    process_pdfs,
    save_documents,
)

# RAG building components
from src.rag_builder import load_documents, load_embedding_model, save_config
from src.rag_builder.builder import build_hybrid_vectorstore
from src.rag_builder.connection_manager import get_index_config, get_milvus_client
from src.rag_builder.vector_bge_m3 import BGEM3Encoder

# RAG retrieval components
from src.rag_retriever import (
    DocumentRetriever,
    HybridSearcher,
    RAGGenerator,
    RAGLLMCaller,
    VietnameseRAG,
    get_rag_llm_caller,
    hybrid_search,
    load_retrieval_models,
)
from src.rag_retriever.models import GeminiLLM, LLMFactory, WatsonxLLM, get_llm

# Utilities
from src.utils.logging import RAGLogger, get_logger

__all__ = [
    # Configuration
    "BATCH_SIZE",
    "CHARS_PER_TOKEN",
    "CHUNKER_MODEL",
    "COLLECTION_NAME",
    "DEFAULT_K",
    "DENSE_INDEX_CONFIG",
    "DENSE_INDEX_FALLBACK_CONFIG",
    "DENSE_SEARCH_PARAMS",
    "DENSE_SEARCH_FALLBACK_PARAMS",
    "DEVICE",
    "EMBED_MODEL_ID",
    "EMBEDDING_DIM",
    "ENCODE_KWARGS",
    "FP16",
    "LOG_FILE_BUILD",
    "LOG_FILE_PREPROCESS",
    "MAX_CONTEXT_LENGTH",
    "MAX_RETRIES",
    "MAX_TOKENS",
    "METADATA_FILE",
    "MILVUS_DOCKER_URI",
    "MILVUS_METADATA_FILE",
    "MILVUS_URI",
    "MODEL_KWARGS",
    "OUTPUT_DIR",
    "OVERLAP_TOKENS",
    "PDF_INPUT_DIR",
    "PROCESSED_DOCS_FILE",
    "PROCESSING_SESSION_ID",
    "PROMPT",
    "RAG_CONFIG_FILE",
    "RAG_METADATA_FILE",
    "RERANK_TOP_K",
    "RERANKER_MODEL_ID",
    "RRF_K",
    "SAFETY_MARGIN",
    "SIMILARITY_THRESHOLD",
    "SPARSE_INDEX_CONFIG",
    "SPARSE_SEARCH_PARAMS",
    "USE_DOCKER_MILVUS",
    # Document processing
    "ChunkMetadataGenerator",
    "TextCleaner",
    "preview_documents",
    "process_pdfs",
    "save_documents",
    # RAG building
    "build_hybrid_vectorstore",
    "BGEM3Encoder",
    "get_index_config",
    "get_milvus_client",
    "load_documents",
    "load_embedding_model",
    "save_config",
    # RAG retrieval
    "HybridSearcher",
    "hybrid_search",
    "DocumentRetriever",
    "RAGGenerator",
    "VietnameseRAG",
    "load_retrieval_models",
    "RAGLLMCaller",
    "get_rag_llm_caller",
    # LLM Models
    "LLMFactory",
    "get_llm",
    "GeminiLLM",
    "WatsonxLLM",
    # Utilities
    "get_logger",
    "RAGLogger",
]
