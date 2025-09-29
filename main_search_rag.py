"""
RAG System Search Interface - Complete Answer Generation
========================================================

Complete RAG system with hybrid BGE-M3 search and answer generation
using unified Vietnamese RAG class with LLM integration.
"""

import gc
import os
from typing import Any, Dict, List, Tuple

from langchain.schema import Document

from src import (
    COLLECTION_NAME,
    HAS_RAG_RETRIEVER,
    VietnameseRAG,
    load_retrieval_models,
    require_rag_retriever,
)
from src.rag_builder.connection_manager import get_milvus_client
from src.utils.logging import RAGLogger
from src.utils.logging import retrieval_logger as logger

# Ensure we have RAG retrieval components
if not HAS_RAG_RETRIEVER:
    require_rag_retriever()

# Configure environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU usage
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def perform_retrieval(rag, query: str) -> Tuple[List[Document], List[float]]:
    """
    Perform document retrieval and reranking only.

    Args:
        rag: The Vietnamese RAG system instance
        query: The search query

    Returns:
        Tuple of (documents, rerank_scores)
    """
    logger.milestone(f"Starting retrieval for query: {query[:50]}...")

    try:
        # Use the retriever directly for cleaner separation
        documents, scores = rag.retriever.retrieve_and_rerank(query)

        logger.info(f"Retrieved {len(documents)} documents with scores")
        return documents, scores

    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        return [], []


def generate_answer(rag, query: str, documents: List[Document]) -> Dict[str, Any]:
    """
    Generate an answer using LLM based on retrieved documents.

    Args:
        rag: The Vietnamese RAG system instance
        query: The search query
        documents: Retrieved documents

    Returns:
        Dictionary containing answer and metadata
    """
    if not documents:
        return {
            "success": False,
            "error": "No documents provided for answer generation",
            "answer": None,
        }

    try:
        logger.milestone(f"Generating answer for query: {query[:50]}...")

        # Use the generator on the RAG instance
        answer_result = rag.generator.generate_answer(query, documents)

        logger.info("Answer generation completed")
        return {
            "success": True,
            "answer": answer_result.get("answer", ""),
            "confidence": answer_result.get("confidence", 0.0),
            "error": None,
        }

    except Exception as e:
        logger.error(f"Error during answer generation: {e}")
        return {"success": False, "error": str(e), "answer": None}


def display_results(
    query: str,
    documents: List[Document],
    scores: List[float],
    answer_result: Dict[str, Any],
) -> None:
    """
    Display the retrieval and answer generation results.

    Args:
        query: The search query
        documents: Retrieved documents
        scores: Document scores
        answer_result: LLM answer generation result
    """
    if not documents:
        print("❌ No relevant documents found.")
        return

    # If answer generation succeeded, show only query and answer
    if answer_result.get("success") and answer_result.get("answer"):
        print(f"\n🔍 **Query:** {query}")
        print(f"\n🤖 **Answer:**")
        print("=" * 50)
        print(answer_result["answer"])

        # Log the successful Q&A pair
        retrieval_info = {
            "num_results": len(documents),
            "sources": [
                {
                    "filename": doc.metadata.get("source_filename", "Unknown"),
                    "score": score,
                    "content_length": len(doc.page_content),
                }
                for doc, score in zip(documents, scores)
            ],
            "confidence": answer_result.get("confidence", 0.0),
        }
        RAGLogger.log_qa_pair(query, answer_result["answer"], retrieval_info)

    # If answer generation failed, show LLM calling prompt
    else:
        print(f"\n⚠️ **LLM calling failed or returned no answer**")
        if answer_result.get("error"):
            print(f"Error: {answer_result['error']}")

        print(f"\n🔍 Query: {query}")
        print(
            f"📊 Retrieved {len(documents)} documents but LLM answer generation failed."
        )

        # Log the failed attempt
        error_msg = answer_result.get("error", "LLM answer generation failed")
        retrieval_info = {"num_results": len(documents)}
        RAGLogger.log_qa_pair(query, f"Failed: {error_msg}", retrieval_info)


def main():
    """Main execution function with interactive loop"""
    logger.milestone("Starting Vietnamese Hybrid RAG Search System")

    try:
        logger.info("Loading RAG system components")

        # Load individual components using new unified structure
        client, _ = get_milvus_client()
        # load_retrieval_models returns (client, embedding_model, reranker_model)
        _, embedding_model, reranker_model = load_retrieval_models()

        rag = VietnameseRAG(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
        )

        logger.milestone("RAG system loaded successfully - Ready for queries")

        print("\n💡 Enter 'quit' to exit the system")
        print("🤖 Complete RAG: BGE-M3 search → LLM answer generation")
        print("=" * 50)

        while True:
            try:
                # Get user input
                query = input("\n🔍 Enter your query: ").strip()

                # Check for quit command
                if query.lower() in ["quit", "q", "exit"]:
                    logger.info("User requested system exit")
                    print("👋 Goodbye!")
                    break

                if not query:
                    print("⚠️ Please enter a valid query")
                    continue

                logger.info(f"Processing query: '{query}'")

                # Step 1: Perform retrieval
                documents, scores = perform_retrieval(rag, query)

                # Step 2: Generate answer (if documents found)
                answer_result = generate_answer(rag, query, documents)

                # Step 3: Display results
                display_results(query, documents, scores, answer_result)

                # Clean up after each query
                gc.collect()
                logger.info("Memory cleanup completed after query processing")

            except KeyboardInterrupt:
                logger.info("User interrupted with Ctrl+C")
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing query '{query}': {str(e)}")
                print(f"❌ Error processing query: {e}")
                continue

    except Exception as e:
        logger.error(f"Critical error loading RAG system: {str(e)}")
        print(f"❌ Error loading RAG system: {e}")
        print("💡 Make sure you've built the vectorstore first with main_build_rag.py")


if __name__ == "__main__":
    main()
