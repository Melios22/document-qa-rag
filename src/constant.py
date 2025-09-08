"""
Configuration constants for the RAG system.
This module loads and provides access to configuration settings.
"""

import json
import uuid
from pathlib import Path
from typing import Any, Dict

# Get the project root directory (parent of src)
PROJECT_ROOT = Path(__file__).parent.parent


def load_config() -> Dict[str, Any]:
    """Load configuration from config.json"""
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Load configuration
config = load_config()

# import torch
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"


# Project paths (resolved relative to project root)
PDF_INPUT_DIR = PROJECT_ROOT / config["data_paths"]["input"]["pdf_documents"]
OUTPUT_DIR = PROJECT_ROOT / config["data_paths"]["output"]["processed_documents"]
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

PROCESSED_DOCS_FILE = (
    PROJECT_ROOT / config["data_paths"]["generated_files"]["processed_docs"]
)
METADATA_FILE = PROJECT_ROOT / config["data_paths"]["generated_files"]["metadata"]

# Vector database path
MILVUS_URI = PROJECT_ROOT / config["vector_database"]["connection"]["uri"]
MILVUS_DOCKER_URI = config["vector_database"]["connection"]["docker_uri"]
USE_DOCKER_MILVUS = config["vector_database"]["connection"]["use_docker"]
COLLECTION_NAME = config["vector_database"]["connection"]["collection_name"]

# Logging paths
LOG_DIR = PROJECT_ROOT / config["data_paths"]["output"]["logs"]
LOG_DIR.mkdir(exist_ok=True, parents=True)

LOG_FILE_PREPROCESS = PROJECT_ROOT / config["logging"]["files"]["preprocessing"]
LOG_FILE_BUILD = PROJECT_ROOT / config["logging"]["files"]["rag_building"]
LOG_FILE_RETRIEVAL = PROJECT_ROOT / config["logging"]["files"]["retrieval"]
LOG_FILE_ERRORS = PROJECT_ROOT / config["logging"]["files"]["errors"]

# Model configuration
EMBED_MODEL_ID = config["embedding_model"]["model_id"]
CHUNKER_MODEL = config["embedding_model"][
    "model_id"
]  # Alias for backward compatibility
MODEL_KWARGS = config["embedding_model"]["model_kwargs"]
MODEL_KWARGS["device"] = DEVICE

ENCODE_KWARGS = config["embedding_model"]["encode_kwargs"]
EMBEDDING_DIM = config["embedding_model"]["embedding_dimension"]
USE_HYBRID_SEARCH = config["embedding_model"]["use_hybrid_search"]

# Reranker model configuration
RERANKER_MODEL_ID = config["reranker_model"]["model_id"]
FP16 = config["reranker_model"]["use_fp16"]

# Document processing configuration
MAX_CHUNK_TOKENS = config["document_processing"]["chunking"]["max_tokens_per_chunk"]
OVERLAP_TOKENS = config["document_processing"]["chunking"]["overlap_tokens"]
CHARS_PER_TOKEN = config["document_processing"]["chunking"]["chars_per_token"]
SAFETY_MARGIN = config["document_processing"]["chunking"]["safety_margin"]
BATCH_SIZE = config["document_processing"]["batch_processing"]["batch_size"]
MAX_RETRIES = config["document_processing"]["batch_processing"]["max_retries"]

# Vector database configuration
DENSE_WEIGHT = config["vector_database"]["hybrid_search"]["dense_weight"]
SPARSE_WEIGHT = config["vector_database"]["hybrid_search"]["sparse_weight"]
RRF_K = config["vector_database"]["hybrid_search"]["rrf_k"]

# Index configurations
DENSE_INDEX_CONFIG = config["vector_database"]["indexing"]["dense_index"]
DENSE_INDEX_FALLBACK_CONFIG = config["vector_database"]["indexing"][
    "dense_index_fallback"
]
SPARSE_INDEX_CONFIG = config["vector_database"]["indexing"]["sparse_index"]

# Search parameters
DENSE_SEARCH_PARAMS = config["vector_database"]["search_params"]["dense_search_params"]
DENSE_SEARCH_FALLBACK_PARAMS = config["vector_database"]["search_params"][
    "dense_search_params_fallback"
]
SPARSE_SEARCH_PARAMS = config["vector_database"]["search_params"][
    "sparse_search_params"
]

# Search & retrieval configuration
DEFAULT_K = config["search_retrieval"]["vector_search"]["default_k"]
RERANK_TOP_K = config["search_retrieval"]["reranking"]["rerank_top_k"]
SIMILARITY_THRESHOLD = config["search_retrieval"]["vector_search"][
    "similarity_threshold"
]

# LLM configuration
MAX_CONTEXT_LENGTH = config["prompts"]["llm_config"]["max_context_length"]
MAX_OUTPUT_TOKENS = config["prompts"]["llm_config"]["max_tokens"]
TEMPERATURE = config["prompts"]["llm_config"]["temperature"]

# Session ID for logging
PROCESSING_SESSION_ID = str(uuid.uuid4())[:8]

# Prompt templates
PROMPT = config["prompts"]["default_vietnamese"]
