"""
Core Hybrid Search Components
============================

Essential hybrid search functionality for RAG system.
Contains only the core hybrid_search function and HybridSearcher class.
"""

import gc
from typing import Dict, List

from ..constant import (
    COLLECTION_NAME,
    DEFAULT_K,
    DENSE_SEARCH_FALLBACK_PARAMS,
    DENSE_SEARCH_PARAMS,
    RRF_K,
    SIMILARITY_THRESHOLD,
)
from ..rag_builder.connection_manager import get_milvus_client
from ..rag_builder.milvus import reciprocal_rank_fusion, search_dense, search_sparse
from ..utils.logging import get_logger

logger = get_logger("rag.retriever.hybrid_search")


def hybrid_search(
    client,
    collection_name: str,
    query_embedding: List[float],
    query_sparse: Dict,
    k: int = DEFAULT_K,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> List[Dict]:
    """
    Perform hybrid search combining dense and sparse retrieval with RRF fusion.

    Args:
        client: Milvus client
        collection_name: Name of the collection
        query_embedding: Dense embedding vector
        query_sparse: Sparse embedding dictionary
        k: Number of results to retrieve
        similarity_threshold: Minimum similarity threshold

    Returns:
        List of search results with RRF scores
    """
    try:
        logger.info(
            f"Starting hybrid search with k={k}, threshold={similarity_threshold}"
        )

        # Dense search
        try:
            dense_results = search_dense(
                client, collection_name, query_embedding, k, DENSE_SEARCH_PARAMS
            )
            logger.info(f"Dense search returned {len(dense_results)} results")
        except Exception as e:
            logger.warning(f"Dense search failed, trying fallback: {e}")
            dense_results = search_dense(
                client,
                collection_name,
                query_embedding,
                k,
                DENSE_SEARCH_FALLBACK_PARAMS,
            )
            logger.info(f"Dense fallback returned {len(dense_results)} results")

        # Sparse search
        try:
            sparse_results = search_sparse(client, collection_name, query_sparse, k)
            logger.info(f"Sparse search returned {len(sparse_results)} results")
        except Exception as e:
            logger.error(f"Sparse search failed: {e}")
            sparse_results = []

        # RRF fusion
        if dense_results and sparse_results:
            fused_results = reciprocal_rank_fusion(
                dense_results, sparse_results, k=RRF_K
            )
            logger.info(f"RRF fusion combined results: {len(fused_results)}")
        elif dense_results:
            fused_results = dense_results[:k]
            logger.info(f"Using dense-only results: {len(fused_results)}")
        elif sparse_results:
            fused_results = sparse_results[:k]
            logger.info(f"Using sparse-only results: {len(fused_results)}")
        else:
            logger.warning("No results from either search method")
            return []

        # Filter by similarity threshold (use appropriate score field)
        filtered_results = []
        for result in fused_results:
            # Get the best available score (RRF score, combined score, or dense score)
            score = (
                result.get("rrf_score")
                or result.get("combined_score")
                or result.get("dense_score", 0.0)
            )
            if score >= similarity_threshold:
                filtered_results.append(result)

        logger.info(f"Final results after filtering: {len(filtered_results)}")
        return filtered_results

    except Exception as e:
        logger.error(f"Error in hybrid_search: {e}")
        return []
    finally:
        gc.collect()


class HybridSearcher:
    """Hybrid searcher with BGE-M3 embeddings."""

    def __init__(
        self,
        client=None,
        collection_name: str = COLLECTION_NAME,
        embedding_model=None,
        k: int = DEFAULT_K,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ):
        """
        Initialize hybrid searcher.

        Args:
            client: Milvus client
            collection_name: Collection name
            embedding_model: BGE-M3 model
            k: Number of results
            similarity_threshold: Minimum similarity
        """
        if client is None:
            client, _ = get_milvus_client()
        self.client = client
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.k = k
        self.similarity_threshold = similarity_threshold

    def search(self, query: str) -> List[Dict]:
        """
        Perform hybrid search for query.

        Args:
            query: Search query

        Returns:
            List of search results
        """
        try:
            # Get embedding model if not provided
            if self.embedding_model is None:
                from .model_loader import get_embedding_model

                self.embedding_model = get_embedding_model()

            # Generate embeddings
            embeddings = self.embedding_model.encode([query])

            query_embedding = embeddings["dense_vecs"][0]
            query_sparse = embeddings["lexical_weights"][0]

            # Perform hybrid search
            results = hybrid_search(
                self.client,
                self.collection_name,
                query_embedding,
                query_sparse,
                self.k,
                self.similarity_threshold,
            )

            return results

        except Exception as e:
            logger.error(f"Error in HybridSearcher.search: {e}")
            return []
