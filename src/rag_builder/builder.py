"""
Hybrid RAG Builder (BGE-M3 dense+sparse to Milvus)
==================================================

Build a single Milvus collection that stores both dense and sparse vectors
for each chunk using configuration from config.json.
"""

import gc
from typing import List

from langchain.schema import Document

from ..constant import COLLECTION_NAME, EMBED_MODEL_ID, EMBEDDING_DIM, ENCODE_KWARGS
from .connection_manager import get_milvus_client
from .milvus import build_indexes, ensure_hybrid_collection, insert_documents
from .encoder import BGEM3Encoder


def load_encoder() -> BGEM3Encoder:
    """Load BGE-M3 encoder"""
    return BGEM3Encoder(
        model=EMBED_MODEL_ID,
        device="cpu",
        normalize_embeddings=ENCODE_KWARGS.get("normalize_embeddings", True),
        batch_size=ENCODE_KWARGS.get("batch_size", 32),
    )


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
    encoder = load_encoder()

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
