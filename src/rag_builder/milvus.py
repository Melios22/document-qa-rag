"""
Milvus Hybrid (dense + sparse) collection utilities
===================================================

Implements hybrid search with configurable parameters from config.json:
1) Schema with id (VARCHAR), text, dense_vector (FLOAT_VECTOR), sparse_vector (SPARSE_FLOAT_VECTOR), metadata (JSON)
2) Insert both         return formatted_results

    except Exception as e:
        log        return rrf_results

    except Exception as e:
        logger.error(f"Error in RRF fusion: {e}")
        return []rror(f"Error in hybrid search: {e}")
        return [] and sparse vectors
3) Query with collection.hybrid_search using RRF fusion
4) Support for both Docker and file-based Milvus with automatic HNSW detection
"""

import json
import uuid
from typing import Dict, List, Optional, Tuple

from langchain.schema import Document
from pymilvus import (
    AnnSearchRequest,
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    RRFRanker,
    connections,
)
from pymilvus.milvus_client.index import IndexParams

from .. import (
    COLLECTION_NAME,
    DENSE_INDEX_CONFIG,
    DENSE_INDEX_FALLBACK_CONFIG,
    DENSE_SEARCH_FALLBACK_PARAMS,
    DENSE_SEARCH_PARAMS,
    EMBEDDING_DIM,
    RRF_K,
    SPARSE_INDEX_CONFIG,
    SPARSE_SEARCH_PARAMS,
)
from ..utils.logging import get_logger
from .connection_manager import get_index_config, get_milvus_client

logger = get_logger("rag.builder.milvus")


def ensure_hybrid_collection(client: MilvusClient, name: str, dense_dim: int) -> bool:
    """
    Ensure hybrid collection exists with proper schema.

    Returns:
        True if collection was created/recreated, False if it already existed
    """
    collection_created = False
    if client.has_collection(name):
        logger.info(f"Dropping existing collection: {name}")
        client.drop_collection(name)
        collection_created = True

    schema = CollectionSchema(
        fields=[
            FieldSchema(
                name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64
            ),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(
                name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim
            ),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ],
        description="Hybrid dense+sparse collection for BGE-M3",
    )

    logger.info(f"Creating collection: {name}")
    client.create_collection(
        collection_name=name, schema=schema, consistency_level="Strong"
    )

    return True


def build_indexes(client: MilvusClient, name: str) -> None:
    """Build indexes with automatic HNSW/IVF_FLAT selection"""
    try:
        # Get appropriate index configuration
        dense_index_config, dense_search_params = get_index_config()

        logger.info(f"Building dense index: {dense_index_config['index_type']}")

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

        logger.milestone(f"Dense index created: {dense_index_config['index_type']}")

        # Sparse index using config
        logger.info("Building sparse index")
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
        logger.milestone("Sparse index created")

        logger.info("Loading collection")
        client.load_collection(name)
        logger.milestone("Collection loaded and ready for search")

    except Exception as e:
        logger.error(f"Error building indexes: {e}")
        raise


def insert_documents(
    client: MilvusClient,
    name: str,
    dense_vecs,
    sparse_vecs,
    docs: List[Document],
) -> None:
    """Insert documents with better error handling"""
    try:
        rows = []
        for i, doc in enumerate(docs):
            rows.append(
                {
                    "id": uuid.uuid4().hex,
                    "text": doc.page_content,
                    "dense_vector": dense_vecs[i].tolist(),
                    "sparse_vector": sparse_vecs[i],  # dict {token_id: weight}
                    "metadata": doc.metadata or {},
                }
            )

        result = client.insert(collection_name=name, data=rows)
        logger.info(f"Inserted {len(rows)} documents")

    except Exception as e:
        logger.error(f"Error inserting documents: {e}")
        raise


def hybrid_search(
    uri: str,
    name: str,
    dense_q,
    sparse_q,
    k: int = 20,
) -> List[Tuple[Document, float]]:
    """
    Perform hybrid search with automatic parameter selection
    """
    try:
        # Ensure connection for Collection API
        connection_alias = "default"
        try:
            connections.get_connection(alias=connection_alias)
        except Exception:
            connections.connect(alias=connection_alias, uri=uri)

        coll = Collection(name)

        # Get appropriate search parameters
        _, dense_search_params = get_index_config()

        # Create search requests using appropriate parameters
        dense_req = AnnSearchRequest(
            data=[dense_q],
            anns_field="dense_vector",
            param=dense_search_params,
            limit=k,
        )

        sparse_req = AnnSearchRequest(
            data=[sparse_q],
            anns_field="sparse_vector",
            param=SPARSE_SEARCH_PARAMS,
            limit=k,
        )

        # Create RRF ranker using config
        ranker = RRFRanker(k=RRF_K)

        # Perform hybrid search
        res = coll.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=ranker,
            limit=k,
            output_fields=["text", "metadata"],
        )

        # Convert to (Document, score)
        out: List[Tuple[Document, float]] = []
        for hit in res[0]:
            # PyMilvus hit object structure
            meta = hit.entity.get("metadata", {})
            text = hit.entity.get("text", "")
            score = float(hit.distance)
            out.append((Document(page_content=text, metadata=meta), score))

        return out

    except Exception as e:
        print(f"❌ Error in hybrid search: {e}")
        return []


def search_dense(
    client: MilvusClient,
    collection_name: str,
    query_embedding: List[float],
    k: int = 20,
    search_params: Optional[Dict] = None,
) -> List[Dict]:
    """
    Perform dense vector search.

    Args:
        client: Milvus client
        collection_name: Name of the collection
        query_embedding: Dense query embedding
        k: Number of results to return
        search_params: Search parameters

    Returns:
        List of search results
    """
    try:
        search_params = search_params or DENSE_SEARCH_PARAMS

        results = client.search(
            collection_name=collection_name,
            data=[query_embedding],
            anns_field="dense_vector",
            search_params=search_params,
            limit=k,
            output_fields=["text", "metadata"],
        )

        # Convert to standard format
        formatted_results = []
        for i, hit in enumerate(results[0]):
            formatted_results.append(
                {
                    "id": hit.id,  # Use attribute, not dict access
                    "dense_score": float(hit.distance),  # Use attribute directly
                    "rank": i + 1,
                    "entity": hit.entity,  # hit.entity is already a dict
                }
            )

        return formatted_results

    except Exception as e:
        logger.error(f"Error in dense search: {e}")
        return []


def search_sparse(
    client: MilvusClient,
    collection_name: str,
    query_sparse: Dict,
    k: int = 20,
) -> List[Dict]:
    """
    Perform sparse vector search.

    Args:
        client: Milvus client
        collection_name: Name of the collection
        query_sparse: Sparse query embedding (dict of token_id -> weight)
        k: Number of results to return

    Returns:
        List of search results
    """
    try:
        results = client.search(
            collection_name=collection_name,
            data=[query_sparse],
            anns_field="sparse_vector",
            search_params=SPARSE_SEARCH_PARAMS,
            limit=k,
            output_fields=["text", "metadata"],
        )

        # Convert to standard format
        formatted_results = []
        for i, hit in enumerate(results[0]):
            formatted_results.append(
                {
                    "id": hit.id,  # Use attribute, not dict access
                    "sparse_score": float(hit.distance),  # Use attribute directly
                    "rank": i + 1,
                    "entity": hit.entity,  # hit.entity is already a dict
                }
            )

        return formatted_results

    except Exception as e:
        logger.error(f"Error in sparse search: {e}")
        return []


def reciprocal_rank_fusion(
    dense_results: List[Dict],
    sparse_results: List[Dict],
    k: int = 10,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> List[Dict]:
    """
    Simplified RRF fusion with better scoring.
    """
    try:
        doc_scores = {}
        all_docs = {}

        # Process dense results
        for rank, result in enumerate(dense_results, 1):
            doc_id = result.get("id")
            if doc_id:
                rrf_score = dense_weight / (k + rank)
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
                all_docs[doc_id] = result
                all_docs[doc_id]["dense_score"] = result.get("dense_score", 0)

        # Process sparse results
        for rank, result in enumerate(sparse_results, 1):
            doc_id = result.get("id")
            if doc_id:
                rrf_score = sparse_weight / (k + rank)
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

                if doc_id in all_docs:
                    all_docs[doc_id]["sparse_score"] = result.get("sparse_score", 0)
                else:
                    all_docs[doc_id] = result
                    all_docs[doc_id]["sparse_score"] = result.get("sparse_score", 0)
                    all_docs[doc_id]["dense_score"] = 0

        # Normalize scores to 0-1 range
        if doc_scores:
            max_score = max(doc_scores.values())
            min_score = min(doc_scores.values())
            score_range = max_score - min_score

            if score_range > 0:
                for doc_id in doc_scores:
                    normalized = (doc_scores[doc_id] - min_score) / score_range
                    doc_scores[doc_id] = 0.1 + 0.9 * normalized
            else:
                for doc_id in doc_scores:
                    doc_scores[doc_id] = 0.5

        # Sort and format results
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        fused_results = []
        for doc_id, rrf_score in sorted_docs:
            doc = all_docs[doc_id].copy()
            doc["rrf_score"] = rrf_score
            doc["combined_score"] = rrf_score
            fused_results.append(doc)

        return fused_results

    except Exception as e:
        print(f"❌ Error in RRF fusion: {e}")
        return []
