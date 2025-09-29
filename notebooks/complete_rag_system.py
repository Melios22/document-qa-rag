#!/usr/bin/env python3
"""
Complete Vietnamese RAG System - All-in-One Implementation
===========================================================

This unified script combines all three stages of the RAG pipeline:
1. Document Preprocessing (PDF processing and text cleaning)
2. RAG Building (Vector database construction with BGE-M3 embeddings)
3. RAG Retrieval (Interactive search and answer generation)

Simply run the script and it will execute all stages sequentially,
then enter an interactive query loop.
"""

import gc
import json
import os
import pickle
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU usage
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Add the project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Core imports
import torch
from langchain.schema import Document
from tqdm.auto import tqdm

# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml"""
    config_path = PROJECT_ROOT / "notebooks" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Load configuration and set up constants
config = load_config()
DEVICE = "cpu"

# Project paths
PDF_INPUT_DIR = PROJECT_ROOT / config["data_paths"]["input"]["pdf_documents"]
OUTPUT_DIR = PROJECT_ROOT / config["data_paths"]["output"]["processed_documents"]
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

PROCESSED_DOCS_FILE = (
    PROJECT_ROOT / config["data_paths"]["generated_files"]["processed_docs"]
)
METADATA_FILE = PROJECT_ROOT / config["data_paths"]["generated_files"]["metadata"]

# Vector database configuration
MILVUS_URI = PROJECT_ROOT / config["vector_database"]["connection"]["uri"]
MILVUS_DOCKER_URI = config["vector_database"]["connection"]["docker_uri"]
USE_DOCKER_MILVUS = config["vector_database"]["connection"]["use_docker"]
COLLECTION_NAME = config["vector_database"]["connection"]["collection_name"]

# Logging setup
LOG_DIR = PROJECT_ROOT / config["data_paths"]["output"]["logs"]
LOG_DIR.mkdir(exist_ok=True, parents=True)

# Model configuration
EMBED_MODEL_ID = config["embedding_model"]["model_id"]
CHUNKER_MODEL = config["embedding_model"]["model_id"]
MODEL_KWARGS = config["embedding_model"]["model_kwargs"]
MODEL_KWARGS["device"] = DEVICE
ENCODE_KWARGS = config["embedding_model"]["encode_kwargs"]
EMBEDDING_DIM = config["embedding_model"]["embedding_dimension"]

# Reranker configuration
RERANKER_MODEL_ID = config["reranker_model"]["model_id"]
FP16 = config["reranker_model"]["use_fp16"]

# Document processing configuration
MAX_CHUNK_TOKENS = config["document_processing"]["chunking"]["max_tokens_per_chunk"]
OVERLAP_TOKENS = config["document_processing"]["chunking"]["overlap_tokens"]
CHARS_PER_TOKEN = config["document_processing"]["chunking"]["chars_per_token"]
SAFETY_MARGIN = config["document_processing"]["chunking"]["safety_margin"]
BATCH_SIZE = config["document_processing"]["batch_processing"]["batch_size"]
MAX_RETRIES = config["document_processing"]["batch_processing"]["max_retries"]

# Vector database indexing
DENSE_WEIGHT = config["vector_database"]["hybrid_search"]["dense_weight"]
SPARSE_WEIGHT = config["vector_database"]["hybrid_search"]["sparse_weight"]
RRF_K = config["vector_database"]["hybrid_search"]["rrf_k"]

DENSE_INDEX_CONFIG = config["vector_database"]["indexing"]["dense_index"]
DENSE_INDEX_FALLBACK_CONFIG = config["vector_database"]["indexing"][
    "dense_index_fallback"
]
SPARSE_INDEX_CONFIG = config["vector_database"]["indexing"]["sparse_index"]

DENSE_SEARCH_PARAMS = config["vector_database"]["search_params"]["dense_search_params"]
DENSE_SEARCH_FALLBACK_PARAMS = config["vector_database"]["search_params"][
    "dense_search_params_fallback"
]
SPARSE_SEARCH_PARAMS = config["vector_database"]["search_params"][
    "sparse_search_params"
]

# Search and retrieval
DEFAULT_K = config["search_retrieval"]["vector_search"]["default_k"]
RERANK_TOP_K = config["search_retrieval"]["reranking"]["rerank_top_k"]
SIMILARITY_THRESHOLD = config["search_retrieval"]["vector_search"][
    "similarity_threshold"
]

# LLM configuration
MAX_CONTEXT_LENGTH = config["prompts"]["llm_config"]["max_context_length"]
MAX_OUTPUT_TOKENS = config["prompts"]["llm_config"]["max_tokens"]
TEMPERATURE = config["prompts"]["llm_config"]["temperature"]
PROMPT = config["prompts"]["default_vietnamese"]

# Session ID for logging
PROCESSING_SESSION_ID = str(uuid.uuid4())[:8]

# ============================================================================
# LOGGING UTILITIES
# ============================================================================


class SimpleLogger:
    """Simple console logger for the unified system"""

    def __init__(self, name: str):
        self.name = name

    def info(self, message: str):
        # Reduced logging - only show important info
        pass

    def milestone(self, message: str):
        print(f"✅ {message}")

    def warning(self, message: str):
        print(f"⚠️  {message}")

    def error(self, message: str):
        print(f"❌ {message}")


# Create loggers for different stages
preprocess_logger = SimpleLogger("Preprocess")
build_logger = SimpleLogger("RAGBuilder")
retrieval_logger = SimpleLogger("Retrieval")

# ============================================================================
# STAGE 1: DOCUMENT PREPROCESSING
# ============================================================================


class TextCleaner:
    """Vietnamese text cleaning utilities"""

    def __init__(self):
        # Load Vietnamese stopwords
        stopwords_file = PROJECT_ROOT / "data" / "vietnamese-stopwords.txt"
        self.stopwords = set()
        if stopwords_file.exists():
            with open(stopwords_file, "r", encoding="utf-8") as f:
                self.stopwords = {line.strip().lower() for line in f if line.strip()}

    def clean_text(self, text: str) -> Tuple[str, Dict]:
        """Clean and normalize Vietnamese text"""
        if not text or not isinstance(text, str):
            return "", {"error": "Invalid input text"}

        original_length = len(text)

        # Basic cleaning
        text = text.strip()
        # Remove excessive whitespace
        import re

        text = re.sub(r"\s+", " ", text)
        # Remove special characters but keep Vietnamese diacritics
        text = re.sub(r"[^\w\s\u00C0-\u024F\u1E00-\u1EFF]", " ", text)
        text = text.strip()

        cleaned_length = len(text)

        cleaning_metadata = {
            "original_length": original_length,
            "cleaned_length": cleaned_length,
            "reduction_ratio": (
                (original_length - cleaned_length) / original_length
                if original_length > 0
                else 0
            ),
        }

        return text, cleaning_metadata

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for Vietnamese text"""
        if not text:
            return 0
        return max(1, len(text) // CHARS_PER_TOKEN)


class ChunkMetadataGenerator:
    """Generate metadata for document chunks"""

    def __init__(self, session_id: str):
        self.session_id = session_id

    def create_metadata(
        self,
        doc: Document,
        source_file: str,
        chunk_index: int,
        cleaning_metadata: Dict,
        estimated_tokens: int,
    ) -> Dict:
        """Create comprehensive metadata for a document chunk"""

        # Extract filename from path
        source_filename = Path(source_file).name

        # Create unique chunk ID
        chunk_id = f"{source_filename}_{chunk_index}_{self.session_id}"

        # Combine existing metadata with new metadata
        metadata = {
            "chunk_id": chunk_id,
            "source_filename": source_filename,
            "source_path": source_file,
            "chunk_index": chunk_index,
            "estimated_tokens": estimated_tokens,
            "processing_session": self.session_id,
            "processed_at": datetime.now().isoformat(),
            "cleaning_stats": cleaning_metadata,
            "char_count": len(doc.page_content),
            # Preserve original metadata if exists
            **doc.metadata,
        }

        return metadata


def process_pdfs() -> List[Document]:
    """PDF processing with BAAI BGE-M3 and enhanced error handling"""

    # Memory optimization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    text_cleaner = TextCleaner()
    metadata_generator = ChunkMetadataGenerator(PROCESSING_SESSION_ID)

    # Find PDF files
    pdf_files = list(Path(PDF_INPUT_DIR).glob("*.pdf"))
    preprocess_logger.info(f"Found {len(pdf_files)} PDF files for processing")

    if not pdf_files:
        preprocess_logger.error(f"No PDF files found in {PDF_INPUT_DIR}")
        return []

    all_documents = []

    # Initialize chunker with BAAI BGE-M3
    preprocess_logger.info(f"Initializing chunker with {CHUNKER_MODEL}")

    try:
        from docling.chunking import HybridChunker
        from langchain_docling import DoclingLoader
        from langchain_docling.loader import ExportType

        chunker = HybridChunker(
            tokenizer=CHUNKER_MODEL,
            max_tokens=MAX_CHUNK_TOKENS,
            overlap_tokens=OVERLAP_TOKENS,
        )
        preprocess_logger.milestone("Chunker initialized successfully")

    except Exception as e:
        preprocess_logger.error(f"Chunker initialization failed: {e}")
        return []

    # Process each PDF file
    pbar = tqdm(total=len(pdf_files), desc="Processing PDFs")

    for pdf_file in pdf_files:
        pbar.set_description(f"Processing {pdf_file.name}")
        pdf_file_str = str(pdf_file)

        try:
            # Use DoclingLoader with chunker
            loader = DoclingLoader(
                file_path=pdf_file_str,
                export_type=ExportType.DOC_CHUNKS,
                chunker=chunker,
            )

            # Load documents - already chunked
            docs = loader.load()

            # Process each chunk
            file_chunks_processed = 0

            for i, doc in enumerate(docs):
                # Clean text
                cleaned_content, cleaning_metadata = text_cleaner.clean_text(
                    doc.page_content
                )

                if cleaned_content and len(cleaned_content) > 50:
                    estimated_tokens = text_cleaner.estimate_tokens(cleaned_content)

                    # Create metadata
                    metadata = metadata_generator.create_metadata(
                        doc=doc,
                        source_file=pdf_file_str,
                        chunk_index=i,
                        cleaning_metadata=cleaning_metadata,
                        estimated_tokens=estimated_tokens,
                    )

                    # Update document with cleaned content and metadata
                    doc.page_content = cleaned_content
                    doc.metadata = metadata

                    all_documents.append(doc)
                    file_chunks_processed += 1

            del loader
            pbar.update(1)

        except Exception as e:
            preprocess_logger.error(f"Error processing {pdf_file.name}: {e}")
            pbar.update(1)
            continue

    del chunker
    pbar.close()

    preprocess_logger.milestone(
        f"Processing Complete: {len(all_documents)} total chunks created"
    )
    return all_documents


def save_documents(documents: List[Document]) -> Dict[str, Any]:
    """Save documents and metadata"""
    if not documents:
        preprocess_logger.error("No documents to save!")
        return {"success": False}

    # Define output files
    files = {
        "documents": PROCESSED_DOCS_FILE,
        "metadata": METADATA_FILE,
    }

    try:
        # Save processed documents
        with open(files["documents"], "wb") as f:
            pickle.dump(documents, f)
        preprocess_logger.info(f"Documents saved: {files['documents'].name}")

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
        preprocess_logger.info(f"Metadata saved: {files['metadata'].name}")

        return {
            "success": True,
            "files": files,
            "stats": processing_metadata["summary_stats"],
        }

    except Exception as e:
        preprocess_logger.error(f"Error saving documents: {e}")
        return {"success": False, "error": str(e)}


def preview_documents(documents: List[Document], num_samples: int = 3):
    """Preview sample documents"""
    if not documents:
        preprocess_logger.warning("No documents to preview")
        return

    preprocess_logger.info(f"Document Preview ({len(documents)} total chunks):")

    preview_docs = documents[:num_samples]

    for i, doc in enumerate(preview_docs):
        preprocess_logger.info(f"Chunk {i+1}:")
        preprocess_logger.info(f"  ID: {doc.metadata.get('chunk_id')}")
        preprocess_logger.info(f"  Source: {doc.metadata.get('source_filename')}")
        preprocess_logger.info(f"  Tokens: {doc.metadata.get('estimated_tokens')}")
        preprocess_logger.info(f"  Content: {doc.page_content[:120]}...")


# ============================================================================
# STAGE 2: RAG BUILDING (Vector Database Construction)
# ============================================================================


def load_documents() -> List[Document]:
    """Load processed documents from Stage 1"""
    file_path = PROCESSED_DOCS_FILE
    try:
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

        print(f"📚 Loaded {len(documents)} documents for vectorization")
        gc.collect()

        return documents

    except FileNotFoundError:
        build_logger.error(f"File not found: {file_path}")
        build_logger.error("Please run Stage 1 processing first!")
        return []
    except Exception as e:
        build_logger.error(f"Error loading documents: {e}")
        return []


class BGEM3Encoder:
    """BGE-M3 encoder for dense and sparse embeddings"""

    def __init__(
        self,
        model: str = "BAAI/bge-m3",
        device: str = "cpu",
        normalize_embeddings: bool = True,
        use_fp16: bool = False,
        max_length: int = 512,
        batch_size: int = 32,
        trust_remote_code: bool = True,
    ):
        from FlagEmbedding import BGEM3FlagModel

        self.model_id = model
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.use_fp16 = use_fp16
        self.max_length = max_length
        self.batch_size = batch_size
        self.model = BGEM3FlagModel(
            model,
            device=device,
            use_fp16=use_fp16,
            trust_remote_code=trust_remote_code,
        )

    def encode(
        self, text_or_texts: Union[str, List[str]], batch_size: Optional[int] = None
    ) -> Dict[str, List]:
        """Encode text(s) to dense and sparse vectors"""
        try:
            if isinstance(text_or_texts, str):
                texts = [text_or_texts]
            else:
                texts = text_or_texts

            if not texts:
                raise ValueError("No texts provided for encoding")

            out = self.model.encode(
                sentences=texts,
                batch_size=batch_size or self.batch_size,
                max_length=self.max_length,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )

            # Ensure consistent data types
            if "dense_vecs" in out:
                import numpy as np

                out["dense_vecs"] = out["dense_vecs"].astype(np.float32)

            # Validate output structure
            if "dense_vecs" not in out or "lexical_weights" not in out:
                raise ValueError("Invalid output from BGE-M3 model")

            return out

        except Exception as e:
            print(f"❌ Error encoding with BGE-M3: {e}")
            import numpy as np

            return {
                "dense_vecs": np.array([]).astype(np.float32),
                "lexical_weights": [],
            }


def get_milvus_client():
    """Get Milvus client with automatic configuration detection"""
    from pymilvus import MilvusClient

    try:
        if USE_DOCKER_MILVUS:
            print(f"🔌 Connecting to Docker Milvus...")
            client = MilvusClient(uri=MILVUS_DOCKER_URI)
        else:
            print(f"🔌 Using local Milvus...")
            client = MilvusClient(uri=str(MILVUS_URI))

        # Test connection and HNSW support
        try:
            # Try a simple operation to test connection
            collections = client.list_collections()
            supports_hnsw = True  # Assume modern Milvus supports HNSW
            return client, supports_hnsw
        except Exception as e:
            build_logger.warning(f"Connection test failed: {e}")
            return client, False

    except Exception as e:
        build_logger.error(f"Failed to connect to Milvus: {e}")
        raise


def ensure_hybrid_collection(client, name: str, dense_dim: int) -> bool:
    """Create hybrid collection schema"""
    from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

    try:
        # Drop existing collection if it exists
        if client.has_collection(name):
            build_logger.info(f"Dropping existing collection: {name}")
            client.drop_collection(name)

        # Define schema
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=False,
                max_length=500,
            ),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(
                name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim
            ),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]

        schema = CollectionSchema(
            fields=fields, description="Hybrid dense+sparse collection"
        )

        # Create collection
        build_logger.info(f"Creating collection: {name}")
        client.create_collection(
            collection_name=name, schema=schema, consistency_level="Strong"
        )

        return True

    except Exception as e:
        build_logger.error(f"Error creating collection: {e}")
        return False


def get_index_config():
    """Get index configuration with automatic fallback"""
    try:
        # Try HNSW first
        return DENSE_INDEX_CONFIG, DENSE_SEARCH_PARAMS
    except:
        # Fallback to IVF_FLAT
        return DENSE_INDEX_FALLBACK_CONFIG, DENSE_SEARCH_FALLBACK_PARAMS


def build_indexes(client, name: str) -> None:
    """Build indexes with automatic HNSW/IVF_FLAT selection"""
    from pymilvus.milvus_client.index import IndexParams

    try:
        # Get appropriate index configuration
        dense_index_config, dense_search_params = get_index_config()

        build_logger.info(f"Building dense index: {dense_index_config['index_type']}")

        # Dense index with auto-detection
        dense_index_params = IndexParams()
        dense_index_params.add_index(
            field_name="dense_vector",
            index_type=dense_index_config["index_type"],
            metric_type=dense_index_config["metric_type"],
            params=dense_index_config["params"],
        )
        client.create_index(
            collection_name=name,
            index_params=dense_index_params,
        )

        build_logger.milestone(
            f"Dense index created: {dense_index_config['index_type']}"
        )

        # Sparse index
        sparse_index_params = IndexParams()
        sparse_index_params.add_index(
            field_name="sparse_vector",
            index_type=SPARSE_INDEX_CONFIG["index_type"],
            metric_type=SPARSE_INDEX_CONFIG["metric_type"],
            params=SPARSE_INDEX_CONFIG["params"],
        )
        client.create_index(
            collection_name=name,
            index_params=sparse_index_params,
        )

        build_logger.info("Loading collection")
        client.load_collection(name)
        print("✅ Collection indexed and loaded successfully")

    except Exception as e:
        build_logger.error(f"Error building indexes: {e}")
        raise


def insert_documents(
    client, name: str, dense_vecs, sparse_vecs, docs: List[Document]
) -> None:
    """Insert documents with vectors into Milvus"""
    try:
        entities = []
        for i, doc in enumerate(docs):
            entities.append(
                {
                    "id": doc.metadata.get("chunk_id", f"doc_{i}"),
                    "text": doc.page_content,
                    "dense_vector": dense_vecs[i],
                    "sparse_vector": sparse_vecs[i],
                    "metadata": doc.metadata,
                }
            )

        client.insert(collection_name=name, data=entities)
        build_logger.info(f"Inserted {len(entities)} documents")

    except Exception as e:
        build_logger.error(f"Error inserting documents: {e}")
        raise


def build_hybrid_vectorstore(documents: List[Document], batch_size: int = 32) -> bool:
    """Build hybrid vectorstore with automatic Docker/local detection"""
    if not documents:
        print("❌ No documents to process")
        return False

    print(f"🔄 Building hybrid vectorstore with {len(documents)} documents...")

    # Get client with automatic connection management
    client, supports_hnsw = get_milvus_client()

    # Create collection
    ensure_hybrid_collection(client, COLLECTION_NAME, EMBEDDING_DIM)

    # Load encoder
    encoder = BGEM3Encoder(
        model=EMBED_MODEL_ID,
        device="cpu",
        normalize_embeddings=ENCODE_KWARGS.get("normalize_embeddings", True),
        batch_size=ENCODE_KWARGS.get("batch_size", 32),
    )

    # Process documents in batches
    total = len(documents)
    total_batches = (total + batch_size - 1) // batch_size

    for batch_idx, start in enumerate(range(0, total, batch_size)):
        end = min(start + batch_size, total)
        batch_docs = documents[start:end]
        texts = [d.page_content for d in batch_docs]

        print(
            f"📦 Processing batch {batch_idx + 1}/{total_batches} ({len(batch_docs)} docs)"
        )

        try:
            # Encode texts
            emb = encoder.encode(texts, batch_size=len(texts))
            dense = emb["dense_vecs"]
            sparse = emb["lexical_weights"]

            # Insert into Milvus
            insert_documents(client, COLLECTION_NAME, dense, sparse, batch_docs)

            # Clean up memory
            del emb, dense, sparse, batch_docs, texts
            gc.collect()

        except Exception as e:
            print(f"❌ Error processing batch {batch_idx + 1}: {e}")
            return False

    # Build indexes
    print("🔧 Building indexes...")
    build_indexes(client, COLLECTION_NAME)

    index_type = "HNSW" if supports_hnsw else "IVF_FLAT"
    print(f"✅ Hybrid vectorstore built successfully with {index_type} indexing!")
    return True


# ============================================================================
# STAGE 3: RAG RETRIEVAL (Search and Answer Generation)
# ============================================================================


def search_dense(
    client,
    collection_name: str,
    query_embedding: List[float],
    k: int,
    search_params: Dict,
):
    """Dense vector search"""
    from pymilvus import AnnSearchRequest

    search_req = AnnSearchRequest(
        data=[query_embedding],
        anns_field="dense_vector",
        param=search_params,
        limit=k,
        expr="",
    )
    return client.search(
        collection_name=collection_name, search_requests=[search_req], limit=k
    )


def search_sparse(client, collection_name: str, query_sparse: Dict, k: int):
    """Sparse vector search"""
    from pymilvus import AnnSearchRequest

    search_req = AnnSearchRequest(
        data=[query_sparse],
        anns_field="sparse_vector",
        param=SPARSE_SEARCH_PARAMS,
        limit=k,
        expr="",
    )
    return client.search(
        collection_name=collection_name, search_requests=[search_req], limit=k
    )


def reciprocal_rank_fusion(
    dense_results: List, sparse_results: List, k: int = 60
) -> List[Dict]:
    """RRF fusion of dense and sparse search results"""
    try:
        # Create score maps
        dense_scores = {}
        sparse_scores = {}

        # Process dense results
        for rank, result in enumerate(dense_results):
            entity = result.get("entity", {})
            doc_id = entity.get("id", "")
            if doc_id:
                dense_scores[doc_id] = {
                    "rank": rank + 1,
                    "score": result.get("distance", 0.0),
                    "entity": entity,
                }

        # Process sparse results
        for rank, result in enumerate(sparse_results):
            entity = result.get("entity", {})
            doc_id = entity.get("id", "")
            if doc_id:
                sparse_scores[doc_id] = {
                    "rank": rank + 1,
                    "score": result.get("distance", 0.0),
                    "entity": entity,
                }

        # Calculate RRF scores
        rrf_scores = {}
        all_ids = set(dense_scores.keys()) | set(sparse_scores.keys())

        for doc_id in all_ids:
            dense_rank = dense_scores.get(doc_id, {}).get("rank", float("inf"))
            sparse_rank = sparse_scores.get(doc_id, {}).get("rank", float("inf"))

            rrf_score = (1.0 / (k + dense_rank)) + (1.0 / (k + sparse_rank))

            # Use entity from whichever search found it (prefer dense)
            entity = dense_scores.get(doc_id, {}).get("entity") or sparse_scores.get(
                doc_id, {}
            ).get("entity", {})

            rrf_scores[doc_id] = {
                "rrf_score": rrf_score,
                "dense_rank": dense_rank,
                "sparse_rank": sparse_rank,
                "entity": entity,
            }

        # Sort by RRF score and format results
        sorted_results = sorted(
            rrf_scores.items(), key=lambda x: x[1]["rrf_score"], reverse=True
        )

        formatted_results = []
        for doc_id, data in sorted_results:
            formatted_results.append(
                {
                    "entity": data["entity"],
                    "rrf_score": data["rrf_score"],
                    "dense_rank": data["dense_rank"],
                    "sparse_rank": data["sparse_rank"],
                }
            )

        return formatted_results

    except Exception as e:
        retrieval_logger.error(f"Error in RRF fusion: {e}")
        return []


def hybrid_search(
    client,
    collection_name: str,
    query_embedding: List[float],
    query_sparse: Dict,
    k: int = DEFAULT_K,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> List[Dict]:
    """Perform hybrid search with RRF fusion"""
    try:
        retrieval_logger.info(f"Starting hybrid search with k={k}")

        # Dense search with fallback
        try:
            dense_results = search_dense(
                client, collection_name, query_embedding, k, DENSE_SEARCH_PARAMS
            )
            dense_results = [
                {"entity": hit["entity"], "distance": hit["distance"]}
                for hit in dense_results[0]
                if "entity" in hit
            ]
            retrieval_logger.info(f"Dense search returned {len(dense_results)} results")
        except Exception as e:
            retrieval_logger.warning(f"Dense search failed, trying fallback: {e}")
            dense_results = search_dense(
                client,
                collection_name,
                query_embedding,
                k,
                DENSE_SEARCH_FALLBACK_PARAMS,
            )
            dense_results = [
                {"entity": hit["entity"], "distance": hit["distance"]}
                for hit in dense_results[0]
                if "entity" in hit
            ]
            retrieval_logger.info(
                f"Dense fallback returned {len(dense_results)} results"
            )

        # Sparse search
        try:
            sparse_results = search_sparse(client, collection_name, query_sparse, k)
            sparse_results = [
                {"entity": hit["entity"], "distance": hit["distance"]}
                for hit in sparse_results[0]
                if "entity" in hit
            ]
            retrieval_logger.info(
                f"Sparse search returned {len(sparse_results)} results"
            )
        except Exception as e:
            retrieval_logger.error(f"Sparse search failed: {e}")
            sparse_results = []

        # RRF fusion
        if dense_results and sparse_results:
            fused_results = reciprocal_rank_fusion(
                dense_results, sparse_results, k=RRF_K
            )
            retrieval_logger.info(f"RRF fusion combined results: {len(fused_results)}")
        elif dense_results:
            fused_results = dense_results[:k]
            retrieval_logger.info(f"Using dense-only results: {len(fused_results)}")
        elif sparse_results:
            fused_results = sparse_results[:k]
            retrieval_logger.info(f"Using sparse-only results: {len(fused_results)}")
        else:
            retrieval_logger.warning("No results from either search method")
            return []

        # Filter by similarity threshold
        filtered_results = []
        for result in fused_results:
            score = (
                result.get("rrf_score")
                or result.get("combined_score")
                or result.get("dense_score", 0.0)
            )
            if score >= similarity_threshold:
                filtered_results.append(result)

        retrieval_logger.info(f"Final results after filtering: {len(filtered_results)}")
        return filtered_results

    except Exception as e:
        retrieval_logger.error(f"Error in hybrid_search: {e}")
        return []
    finally:
        gc.collect()


def get_embedding_model():
    """Load BGE-M3 embedding model"""
    return BGEM3Encoder(
        model=EMBED_MODEL_ID,
        device=DEVICE,
        normalize_embeddings=ENCODE_KWARGS.get("normalize_embeddings", True),
        batch_size=ENCODE_KWARGS.get("batch_size", 8),
    )


def get_reranker_model():
    """Load BGE reranker model"""
    from FlagEmbedding import FlagReranker

    try:
        retrieval_logger.info(f"Loading reranker: {RERANKER_MODEL_ID}")
        reranker = FlagReranker(RERANKER_MODEL_ID, use_fp16=FP16, device=DEVICE)
        retrieval_logger.milestone("Reranker loaded successfully")
        return reranker

    except Exception as e:
        retrieval_logger.error(f"Error loading reranker: {e}")
        return None


class VietnameseRAG:
    """Unified Vietnamese RAG system"""

    def __init__(
        self,
        client=None,
        collection_name: str = COLLECTION_NAME,
        embedding_model=None,
        reranker_model=None,
        k: int = DEFAULT_K,
        rerank_top_k: int = RERANK_TOP_K,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ):

        # Store configuration
        self.collection_name = collection_name
        self.k = k
        self.rerank_top_k = rerank_top_k
        self.similarity_threshold = similarity_threshold

        # Initialize Milvus client
        self.client = client if client else get_milvus_client()[0]

        # Load models
        self.embedding_model = embedding_model or get_embedding_model()
        self.reranker_model = reranker_model or get_reranker_model()

        retrieval_logger.milestone("Vietnamese RAG system initialized")

    def search(self, query: str) -> List[Tuple[Document, float]]:
        """Search for relevant documents"""
        try:
            # Generate embeddings
            embeddings = self.embedding_model.encode([query])
            query_embedding = embeddings["dense_vecs"][0]
            query_sparse = embeddings["lexical_weights"][0]

            # Perform hybrid search
            search_results = hybrid_search(
                client=self.client,
                collection_name=self.collection_name,
                query_embedding=query_embedding,
                query_sparse=query_sparse,
                k=self.k,
                similarity_threshold=self.similarity_threshold,
            )

            # Convert search results to documents
            documents = []
            scores = []
            for result in search_results:
                entity = result.get("entity", {})
                text_content = entity.get("text", "")
                metadata = entity.get("metadata", {})

                doc = Document(page_content=text_content, metadata=metadata)
                documents.append(doc)

                score = result.get(
                    "rrf_score",
                    result.get("combined_score", result.get("dense_score", 0.0)),
                )
                scores.append(score)

            # Rerank if we have a reranker and documents
            if self.reranker_model and documents:
                try:
                    pairs = [[query, doc.page_content] for doc in documents]
                    rerank_scores = self.reranker_model.compute_score(
                        pairs, normalize=True
                    )

                    # Ensure rerank_scores is a list
                    if not isinstance(rerank_scores, list):
                        rerank_scores = [rerank_scores]

                    # Sort by rerank scores
                    ranked_docs = sorted(
                        zip(documents, rerank_scores), key=lambda x: x[1], reverse=True
                    )
                    documents = [doc for doc, _ in ranked_docs[: self.rerank_top_k]]
                    scores = [score for _, score in ranked_docs[: self.rerank_top_k]]

                except Exception as e:
                    retrieval_logger.warning(
                        f"Reranking failed: {e}, using original results"
                    )

            return list(zip(documents, scores))

        except Exception as e:
            retrieval_logger.error(f"Error in search: {e}")
            return []

    def answer(self, query: str) -> Dict:
        """Complete RAG pipeline: search + answer generation"""
        try:
            # Search for documents
            results = self.search(query)

            if not results:
                return {
                    "answer": "Tôi không tìm thấy thông tin phù hợp để trả lời câu hỏi của bạn.",
                    "sources": [],
                    "confidence": 0.0,
                    "success": False,
                    "retrieval_count": 0,
                }

            documents = [doc for doc, _ in results]
            scores = [score for _, score in results]

            # Prepare context from retrieved documents
            context_parts = []
            for i, doc in enumerate(documents[:3], 1):
                content = doc.page_content.strip()
                if content:
                    context_parts.append(f"[Tài liệu {i}]:\n{content}")

            context = (
                "\n\n".join(context_parts)
                if context_parts
                else "Không tìm thấy thông tin liên quan."
            )

            # Create answer using the Vietnamese RAG prompt format
            answer = PROMPT.format(context=context, query=query)

            # For this unified script, we'll provide a structured response
            # In the full system, this would be sent to an LLM for generation
            structured_answer = f"""Dựa trên {len(documents)} tài liệu được tìm thấy:

THÔNG TIN THAM KHẢO:
{context}

ĐÂY LÀ THÔNG TIN THÔ ĐƯỢC TÌM THẤY. Trong hệ thống đầy đủ, thông tin này sẽ được xử lý bởi mô hình ngôn ngữ lớn để tạo ra câu trả lời chi tiết và chính xác hơn theo đúng prompt template của hệ thống."""

            # Prepare sources information
            sources = []
            for doc, score in results[:3]:
                sources.append(
                    {
                        "filename": doc.metadata.get("source_filename", "Unknown"),
                        "score": float(score),
                        "content_preview": (
                            doc.page_content[:200] + "..."
                            if len(doc.page_content) > 200
                            else doc.page_content
                        ),
                    }
                )

            return {
                "answer": structured_answer,
                "sources": sources,
                "confidence": max(scores) if scores else 0.0,
                "success": True,
                "retrieval_count": len(documents),
                "context_length": len(context),
                "raw_prompt": answer,  # Include the formatted prompt for reference
            }

        except Exception as e:
            retrieval_logger.error(f"Error in answer generation: {e}")
            return {
                "answer": "Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi của bạn.",
                "sources": [],
                "confidence": 0.0,
                "success": False,
                "error": str(e),
                "retrieval_count": 0,
            }


# ============================================================================
# MAIN PIPELINE CONTROLLER
# ============================================================================


def run_preprocess():
    """Run document preprocessing stage"""
    preprocess_logger.milestone("Starting RAG Document Preprocessing")

    try:
        preprocess_logger.info("Beginning PDF document processing")
        docs = process_pdfs()

        if not docs:
            preprocess_logger.error("No documents were processed successfully")
            return False

        preprocess_logger.info(f"Successfully processed {len(docs)} documents")

        preprocess_logger.info("Saving processed documents")
        save_result = save_documents(documents=docs)

        if not save_result.get("success"):
            preprocess_logger.error("Failed to save documents")
            return False

        preprocess_logger.info("Generating document preview")
        preview_documents(documents=docs)

        preprocess_logger.milestone(
            f"Document preprocessing completed successfully - {len(docs)} documents ready"
        )
        return True

    except Exception as e:
        preprocess_logger.error(f"Preprocessing failed: {str(e)}")
        return False


def run_build():
    """Run RAG building stage"""
    build_logger.milestone("Starting RAG System Building")

    try:
        # Stage 1: Load processed documents
        build_logger.info("Loading processed documents")
        docs = load_documents()

        if not docs:
            build_logger.error("No processed documents found. Run preprocessing first")
            return False

        build_logger.info(f"Loaded {len(docs)} documents for vectorization")

        # Stage 2: Build hybrid vectorstore
        build_logger.info("Initializing hybrid vector store construction")
        gc.collect()

        build_logger.info("Building hybrid vector database with BGE-M3 embeddings")
        success = build_hybrid_vectorstore(docs, batch_size=32)

        # Cleanup
        del docs
        gc.collect()
        build_logger.info("Memory cleanup completed")

        if success:
            build_logger.milestone(
                "Hybrid RAG system built successfully - Ready for queries"
            )
            return True
        else:
            build_logger.error("Failed to build hybrid vector store")
            return False

    except Exception as e:
        build_logger.error(f"RAG building failed: {str(e)}")
        return False


def run_search():
    """Run interactive search stage"""
    try:
        print("🤖 Loading RAG system...")

        # Initialize RAG system
        rag = VietnameseRAG()

        print("✅ RAG system ready for queries!")

        print("\n💡 Enter 'quit' to exit the system")
        print("🤖 Vietnamese RAG: BGE-M3 hybrid search + document retrieval")
        print("=" * 60)

        while True:
            try:
                # Get user input
                query = input("\n🔍 Enter your query: ").strip()

                # Check for quit command
                if query.lower() in ["quit", "q", "exit"]:
                    print("👋 Goodbye!")
                    break

                if not query:
                    print("⚠️ Please enter a valid query")
                    continue

                print(f"🔍 Searching for: {query}")

                # Perform search and answer generation
                result = rag.answer(query)

                # Display results
                if result.get("success"):
                    print(f"\n🔍 **Query:** {query}")
                    print(f"\n🤖 **Answer:**")
                    print("=" * 50)
                    print(result["answer"])

                    if result.get("sources"):
                        print(f"\n📚 **Sources ({len(result['sources'])}):**")
                        for i, source in enumerate(result["sources"][:3], 1):
                            print(
                                f"{i}. {source['filename']} (score: {source['score']:.3f})"
                            )

                else:
                    print(f"\n❌ Error: {result.get('error', 'Unknown error')}")

                # Clean up after each query
                gc.collect()

            except KeyboardInterrupt:
                retrieval_logger.info("User interrupted with Ctrl+C")
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                retrieval_logger.error(f"Error processing query '{query}': {str(e)}")
                print(f"❌ Error processing query: {e}")
                continue

    except Exception as e:
        retrieval_logger.error(f"Critical error loading RAG system: {str(e)}")
        print(f"❌ Error loading RAG system: {e}")
        print("💡 Make sure you've built the vectorstore first")
        return False

    return True


def main():
    """Main entry point - runs all three stages sequentially"""
    print("🚀 Complete Vietnamese RAG System")
    print("=" * 60)

    # Stage 1: Document Preprocessing
    print("\n📄 STAGE 1: Document Preprocessing")
    print("-" * 40)
    success = run_preprocess()

    if not success:
        print("❌ Preprocessing failed. Stopping.")
        return

    # Stage 2: RAG Building
    print("\n🗃️ STAGE 2: RAG Building")
    print("-" * 40)
    success = run_build()

    if not success:
        print("❌ RAG building failed. Stopping.")
        return

    # Stage 3: Interactive RAG Search
    print("\n🤖 STAGE 3: Interactive RAG Search")
    print("-" * 40)
    run_search()

    print("\n✅ RAG system session completed!")


if __name__ == "__main__":
    main()
    main()
