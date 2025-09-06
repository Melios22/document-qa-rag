"""
Vietnamese RAG System - Essential Exports
==========================================

Modular package with conditional imports for better IDE support.
Each stage can be imported independently based on available dependencies.
"""

# Core constants (no external dependencies) - always available
from src.constant import *

# Conditional imports with graceful degradation
# This allows IDE IntelliSense while preventing import errors

# Document Processing Components
try:
    from src.preprocess import (
        ChunkMetadataGenerator,
        TextCleaner,
        preview_documents,
        process_pdfs,
        save_documents,
    )

    HAS_PREPROCESS = True
except ImportError:
    HAS_PREPROCESS = False
    # Create placeholder classes for IDE support
    ChunkMetadataGenerator = None
    TextCleaner = None
    preview_documents = None
    process_pdfs = None
    save_documents = None

# RAG Building Components
try:
    from src.rag_builder import load_documents, load_embedding_model, save_config
    from src.rag_builder.builder import build_hybrid_vectorstore
    from src.rag_builder.connection_manager import get_index_config, get_milvus_client
    from src.rag_builder.vector_bge_m3 import BGEM3Encoder

    HAS_RAG_BUILDER = True
except ImportError:
    HAS_RAG_BUILDER = False
    # Create placeholder functions for IDE support
    load_documents = None
    load_embedding_model = None
    save_config = None
    build_hybrid_vectorstore = None
    get_index_config = None
    get_milvus_client = None
    BGEM3Encoder = None

# RAG Retrieval Components
try:
    from src.rag_retriever import (
        HybridSearcher,
        RAGLLMCaller,
        VietnameseRAG,
        get_rag_llm_caller,
        hybrid_search,
        load_retrieval_models,
    )
    from src.rag_retriever.models import GeminiLLM, LLMFactory, WatsonxLLM

    HAS_RAG_RETRIEVER = True
except ImportError:
    HAS_RAG_RETRIEVER = False
    # Create placeholder classes for IDE support
    HybridSearcher = None
    RAGLLMCaller = None
    VietnameseRAG = None
    get_rag_llm_caller = None
    hybrid_search = None
    load_retrieval_models = None
    GeminiLLM = None
    LLMFactory = None
    WatsonxLLM = None

# Utilities (lightweight, should always work)
try:
    from src.utils.logging import RAGLogger, get_logger

    HAS_LOGGING = True
except ImportError:
    HAS_LOGGING = False
    RAGLogger = None
    get_logger = None


# Helper functions to check what's available
def check_dependencies():
    """Check which components are available"""
    return {
        "preprocess": HAS_PREPROCESS,
        "rag_builder": HAS_RAG_BUILDER,
        "rag_retriever": HAS_RAG_RETRIEVER,
        "logging": HAS_LOGGING,
    }


def require_preprocess():
    """Ensure preprocessing components are available"""
    if not HAS_PREPROCESS:
        raise ImportError(
            "Preprocessing dependencies not installed. Run: pip install -r requirements-preprocess.txt"
        )


def require_rag_builder():
    """Ensure RAG building components are available"""
    if not HAS_RAG_BUILDER:
        raise ImportError(
            "RAG building dependencies not installed. Run: pip install -r requirements-build.txt"
        )


def require_rag_retriever():
    """Ensure RAG retrieval components are available"""
    if not HAS_RAG_RETRIEVER:
        raise ImportError(
            "RAG retrieval dependencies not installed. Run: pip install -r requirements-retrieval.txt"
        )


__all__ = [
    # Constants (always available)
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
    "MILVUS_URI",
    "MODEL_KWARGS",
    "OUTPUT_DIR",
    "OVERLAP_TOKENS",
    "PDF_INPUT_DIR",
    "PROCESSED_DOCS_FILE",
    "PROCESSING_SESSION_ID",
    "PROMPT",
    "RERANK_TOP_K",
    "RERANKER_MODEL_ID",
    "RRF_K",
    "SAFETY_MARGIN",
    "SIMILARITY_THRESHOLD",
    "SPARSE_INDEX_CONFIG",
    "SPARSE_SEARCH_PARAMS",
    "USE_DOCKER_MILVUS",
    # Conditional components (may be None if dependencies not installed)
    "ChunkMetadataGenerator",
    "TextCleaner",
    "preview_documents",
    "process_pdfs",
    "save_documents",
    "load_documents",
    "load_embedding_model",
    "save_config",
    "build_hybrid_vectorstore",
    "get_index_config",
    "get_milvus_client",
    "BGEM3Encoder",
    "HybridSearcher",
    "RAGLLMCaller",
    "VietnameseRAG",
    "get_rag_llm_caller",
    "hybrid_search",
    "load_retrieval_models",
    "GeminiLLM",
    "LLMFactory",
    "WatsonxLLM",
    "RAGLogger",
    "get_logger",
    # Capability flags
    "HAS_PREPROCESS",
    "HAS_RAG_BUILDER",
    "HAS_RAG_RETRIEVER",
    "HAS_LOGGING",
    # Helper functions
    "check_dependencies",
    "require_preprocess",
    "require_rag_builder",
    "require_rag_retriever",
]
