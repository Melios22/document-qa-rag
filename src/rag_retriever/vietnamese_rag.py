"""
Unified Vietnamese RAG System
=============================

A unified RAG system with modular subclasses for clean architecture.
Main class handles the complete flow while subclasses provide modularity.
"""

from typing import Any, Dict, List, Optional, Tuple

from langchain.schema import Document

from ..constant import COLLECTION_NAME, DEFAULT_K, RERANK_TOP_K, SIMILARITY_THRESHOLD
from ..rag_builder.connection_manager import get_milvus_client
from ..utils.logging import get_logger
from .hybrid_search import hybrid_search
from .llm_caller import RAGLLMCaller, get_rag_llm_caller
from .model_loader import get_embedding_model, get_reranker_model

logger = get_logger("rag.unified")


class DocumentRetriever:
    """Internal subclass for document retrieval operations."""

    def __init__(self, parent_rag):
        """Initialize retriever with reference to parent RAG system."""
        self.parent = parent_rag

    def retrieve_and_rerank(self, query: str) -> Tuple[List[Document], List[float]]:
        """
        Retrieve and rerank documents for a query.

        Args:
            query: Search query

        Returns:
            Tuple of (documents, rerank_scores)
        """
        try:
            logger.milestone(f"Starting retrieval for query: {query[:50]}...")

            # Generate embeddings first
            embeddings = self.parent.embedding_model.encode([query])
            query_embedding = embeddings["dense_vecs"][0]
            query_sparse = embeddings["lexical_weights"][0]

            # Perform hybrid search with embeddings
            search_results = hybrid_search(
                client=self.parent.client,
                collection_name=self.parent.collection_name,
                query_embedding=query_embedding,
                query_sparse=query_sparse,
                k=self.parent.k,
                similarity_threshold=self.parent.similarity_threshold,
            )

            # Convert search results to documents and perform reranking
            documents = []
            scores = []
            for result in search_results:
                # Extract text and metadata from entity field
                entity = result.get("entity", {})
                text_content = entity.get("text", "")
                metadata = entity.get("metadata", {})

                doc = Document(
                    page_content=text_content,
                    metadata=metadata,
                )
                documents.append(doc)
                # Use the combined score from RRF or the available score
                score = result.get(
                    "rrf_score",
                    result.get("combined_score", result.get("dense_score", 0.0)),
                )
                scores.append(score)

            # Perform reranking if we have documents and a reranker model
            if documents and self.parent.reranker_model:
                try:
                    # Prepare texts for reranking
                    texts = [doc.page_content for doc in documents]

                    # Rerank documents
                    rerank_results = self.parent.reranker_model.compute_score(
                        [[query, text] for text in texts]
                    )

                    # Get rerank scores and sort
                    if isinstance(rerank_results, list):
                        rerank_scores = rerank_results
                    else:
                        rerank_scores = rerank_results.tolist()

                    # Sort by rerank scores and take top k
                    scored_docs = list(zip(documents, rerank_scores))
                    scored_docs.sort(key=lambda x: x[1], reverse=True)

                    # Take top rerank_top_k results
                    top_k = min(len(scored_docs), self.parent.rerank_top_k)
                    top_docs = scored_docs[:top_k]

                    documents = [doc for doc, _ in top_docs]
                    rerank_scores = [score for _, score in top_docs]

                except Exception as e:
                    logger.warning(f"Reranking failed, using original scores: {e}")
                    rerank_scores = scores[: self.parent.rerank_top_k]
                    documents = documents[: self.parent.rerank_top_k]
            else:
                # No reranking, just take top results
                top_k = min(len(documents), self.parent.rerank_top_k)
                documents = documents[:top_k]
                rerank_scores = scores[:top_k]

            logger.info(
                f"Retrieved {len(documents)} documents, reranked to top {len(rerank_scores)}"
            )
            return documents, rerank_scores

        except Exception as e:
            logger.error(f"Error in retrieve_and_rerank: {e}")
            return [], []

    def search_only(
        self,
        query: str,
        initial_k: Optional[int] = None,
        final_k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Search for documents without answer generation.

        Args:
            query: Search query
            initial_k: Override initial retrieval count
            final_k: Override final count after reranking

        Returns:
            List of (Document, score) tuples
        """
        # Use provided k values or defaults
        k_to_use = initial_k or self.parent.k
        rerank_k_to_use = final_k or self.parent.rerank_top_k

        try:
            # Generate embeddings first
            embeddings = self.parent.embedding_model.encode([query])
            query_embedding = embeddings["dense_vecs"][0]
            query_sparse = embeddings["lexical_weights"][0]

            # Perform hybrid search with embeddings
            search_results = hybrid_search(
                client=self.parent.client,
                collection_name=self.parent.collection_name,
                query_embedding=query_embedding,
                query_sparse=query_sparse,
                k=k_to_use,
                similarity_threshold=self.parent.similarity_threshold,
            )

            # Convert search results to documents
            documents = []
            scores = []
            for result in search_results:
                # Extract text and metadata from entity field
                entity = result.get("entity", {})
                text_content = entity.get("text", "")
                metadata = entity.get("metadata", {})

                doc = Document(
                    page_content=text_content,
                    metadata=metadata,
                )
                documents.append(doc)
                # Use the combined score from RRF or the available score
                score = result.get(
                    "rrf_score",
                    result.get("combined_score", result.get("dense_score", 0.0)),
                )
                scores.append(score)

            # Return top results according to final_k
            top_k = min(len(documents), rerank_k_to_use)
            return list(zip(documents[:top_k], scores[:top_k]))

        except Exception as e:
            logger.error(f"Error in search_only: {e}")
            return []


class AnswerGenerator:
    """Internal subclass for answer generation operations."""

    def __init__(self, parent_rag):
        """Initialize generator with reference to parent RAG system."""
        self.parent = parent_rag

    def generate_answer(
        self,
        query: str,
        documents: List[Document],
        rerank_scores: Optional[List[float]] = None,
        **llm_kwargs,
    ) -> Dict:
        """
        Generate answer from query and retrieved documents.

        Args:
            query: User question
            documents: Retrieved documents
            rerank_scores: Reranking scores
            **llm_kwargs: Additional LLM parameters

        Returns:
            Dict with answer, sources, confidence, etc.
        """
        if not documents:
            logger.warning("No documents provided for answer generation")
            return {
                "answer": "Tôi không tìm thấy thông tin phù hợp để trả lời câu hỏi của bạn.",
                "sources": [],
                "confidence": 0.0,
                "success": False,
                "retrieval_count": 0,
            }

        try:
            logger.milestone(
                f"Generating answer using {self.parent.llm_caller.model_type} model"
            )

            # Generate answer using LLM
            result = self.parent.llm_caller.generate_answer(
                query=query,
                documents=documents,
                rerank_scores=rerank_scores,
                **llm_kwargs,
            )

            # Add retrieval metadata
            result["retrieval_count"] = len(documents)
            result["success"] = True

            logger.info(
                f"Answer generated successfully with confidence: {result.get('confidence', 'N/A')}"
            )
            return result

        except Exception as e:
            logger.error(f"Error in generate_answer: {e}")
            return {
                "answer": "Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi của bạn.",
                "sources": [],
                "confidence": 0.0,
                "success": False,
                "error": str(e),
                "retrieval_count": len(documents),
            }

    def switch_model(self, model_type: str, **model_kwargs):
        """
        Switch LLM model.

        Args:
            model_type: New model type
            **model_kwargs: Model parameters
        """
        try:
            logger.milestone(
                f"Switching from {self.parent.llm_caller.model_type} to {model_type}"
            )

            # Create new LLM caller with specified model
            self.parent.llm_caller = get_rag_llm_caller(
                model_type=model_type, **model_kwargs
            )

            # Update parent's model type tracking
            self.parent._current_model_type = model_type

            logger.info(f"Successfully switched to {model_type} model")

        except Exception as e:
            logger.error(f"Error switching to {model_type}: {e}")
            raise


class VietnameseRAG:
    """
    Unified Vietnamese RAG system with modular subclasses.

    This class provides the complete RAG flow while using internal
    subclasses for clean modular architecture.
    """

    def __init__(
        self,
        # Connection parameters
        client=None,
        collection_name: str = COLLECTION_NAME,
        # Model parameters
        embedding_model=None,
        reranker_model=None,
        # Retrieval parameters
        k: int = DEFAULT_K,
        rerank_top_k: int = RERANK_TOP_K,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        # LLM parameters
        model_type: str = "gemini",
        llm_caller: Optional[RAGLLMCaller] = None,
        **llm_kwargs,
    ):
        """
        Initialize unified Vietnamese RAG system.

        Args:
            client: Milvus client
            collection_name: Collection name in Milvus
            embedding_model: BGE-M3 embedding model
            reranker_model: Reranker model
            k: Initial retrieval count
            rerank_top_k: Final count after reranking
            similarity_threshold: Minimum similarity threshold
            model_type: LLM model type (gemini, watsonx, etc.)
            llm_caller: Custom LLM caller instance
            **llm_kwargs: Additional LLM parameters
        """
        # Store configuration
        self.collection_name = collection_name
        self.k = k
        self.rerank_top_k = rerank_top_k
        self.similarity_threshold = similarity_threshold
        self._current_model_type = model_type

        # Initialize Milvus client
        self.client = client or get_milvus_client()

        # Load models
        self.embedding_model = embedding_model or get_embedding_model()
        self.reranker_model = reranker_model or get_reranker_model()

        # Initialize LLM caller
        self.llm_caller = llm_caller or get_rag_llm_caller(
            model_type=model_type, **llm_kwargs
        )

        # Initialize modular subclasses
        self.retriever = DocumentRetriever(self)
        self.generator = AnswerGenerator(self)

        logger.milestone(f"Vietnamese RAG system initialized")
        logger.info(
            f"Configuration: {model_type} model, {k}->{rerank_top_k} retrieval, threshold={similarity_threshold}"
        )

    def answer(self, query: str, **llm_kwargs) -> Dict:
        """
        Main method: Complete RAG flow from query to answer.

        Args:
            query: User question
            **llm_kwargs: Additional LLM parameters

        Returns:
            Complete result with answer, sources, confidence, etc.
        """
        try:
            logger.milestone(f"Processing query: {query[:100]}...")

            # Step 1: Retrieve and rerank documents
            documents, rerank_scores = self.retriever.retrieve_and_rerank(query)

            if not documents:
                logger.warning("No documents retrieved for query")
                return {
                    "answer": "Tôi không tìm thấy thông tin phù hợp để trả lời câu hỏi của bạn.",
                    "sources": [],
                    "confidence": 0.0,
                    "success": False,
                    "retrieval_count": 0,
                }

            # Step 2: Generate answer
            result = self.generator.generate_answer(
                query=query,
                documents=documents,
                rerank_scores=rerank_scores,
                **llm_kwargs,
            )

            logger.milestone("Query processing completed successfully")
            return result

        except Exception as e:
            logger.error(f"Error in complete RAG flow: {e}")
            return {
                "answer": "Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi của bạn.",
                "sources": [],
                "confidence": 0.0,
                "success": False,
                "error": str(e),
                "retrieval_count": 0,
            }

    def search(
        self, query: str, initial_k: Optional[int] = None, final_k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for documents only (no answer generation).

        Args:
            query: Search query
            initial_k: Override initial retrieval count
            final_k: Override final count after reranking

        Returns:
            List of (Document, score) tuples
        """
        return self.retriever.search_only(query, initial_k, final_k)

    def switch_model(self, model_type: str, **model_kwargs):
        """
        Switch LLM model.

        Args:
            model_type: New model type (gemini, watsonx, etc.)
            **model_kwargs: Model-specific parameters
        """
        self.generator.switch_model(model_type, **model_kwargs)

    @property
    def model_type(self) -> str:
        """Get current LLM model type."""
        return self._current_model_type

    @property
    def status(self) -> Dict[str, Any]:
        """Get system status and configuration."""
        return {
            "model_type": self.model_type,
            "collection_name": self.collection_name,
            "retrieval_config": {
                "k": self.k,
                "rerank_top_k": self.rerank_top_k,
                "similarity_threshold": self.similarity_threshold,
            },
            "models_loaded": {
                "embedding": self.embedding_model is not None,
                "reranker": self.reranker_model is not None,
                "llm": self.llm_caller is not None,
            },
            "milvus_connected": self.client is not None,
        }

    def update_config(
        self,
        k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ):
        """
        Update retrieval configuration.

        Args:
            k: New initial retrieval count
            rerank_top_k: New reranking count
            similarity_threshold: New similarity threshold
        """
        if k is not None:
            self.k = k
            logger.info(f"Updated k to {k}")

        if rerank_top_k is not None:
            self.rerank_top_k = rerank_top_k
            logger.info(f"Updated rerank_top_k to {rerank_top_k}")

        if similarity_threshold is not None:
            self.similarity_threshold = similarity_threshold
            logger.info(f"Updated similarity_threshold to {similarity_threshold}")


# Convenience function for backward compatibility
def get_vietnamese_rag(**kwargs) -> VietnameseRAG:
    """
    Get Vietnamese RAG instance with default configuration.

    Args:
        **kwargs: Configuration parameters

    Returns:
        VietnameseRAG instance
    """
    return VietnameseRAG(**kwargs)
    return VietnameseRAG(**kwargs)
    return VietnameseRAG(**kwargs)
    return VietnameseRAG(**kwargs)
