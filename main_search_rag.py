"""
RAG System Search Interface - Complete Answer Generation
========================================================

Complete RAG system with hybrid BGE-M3 search and answer generation
using unified Vietnamese RAG class with LLM integration.
"""

import gc
import os

from src import COLLECTION_NAME
from src.rag_builder.connection_manager import get_milvus_client
from src.rag_retriever import VietnameseRAG, get_embedding_model, get_reranker_model
from src.utils.logging import RAGLogger
from src.utils.logging import retrieval_logger as logger

# Configure environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU usage
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def main():
    """Main execution function with interactive loop"""
    logger.milestone("Starting Vietnamese Hybrid RAG Search System")

    try:
        logger.info("Loading RAG system components")

        # Load individual components using new unified structure
        client = get_milvus_client()
        embedding_model = get_embedding_model()
        reranker_model = get_reranker_model()

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

                # Use new unified answer method for complete RAG flow
                result = rag.answer(query)

                if result.get("success", False) and result.get("sources"):
                    sources = result["sources"]
                    logger.info(
                        f"Generated answer with {len(sources)} source documents"
                    )

                    print(f"\n🤖 **Answer:**")
                    print("=" * 50)
                    print(result["answer"])

                    print(f"\n📋 **Sources** ({len(sources)} documents):")
                    print("=" * 50)

                    # Prepare answer for logging
                    retrieval_info = {
                        "num_results": len(sources),
                        "sources": [],
                        "confidence": result.get("confidence", 0.0),
                    }

                    for i, source in enumerate(sources, 1):
                        print(f"\n📄 Document {i}")
                        print("-" * 30)

                        # Show content
                        content = source.get("content", "")
                        content_preview = (
                            content[:500] + "..." if len(content) > 500 else content
                        )
                        print(content_preview)

                        filename = source.get("filename", "Unknown")
                        score = source.get("score", 0.0)
                        print(f"📎 Source: {filename} (Score: {score:.4f})")

                        retrieval_info["sources"].append(
                            {
                                "filename": filename,
                                "score": score,
                                "content_length": len(content),
                            }
                        )

                    # Log the Q&A pair
                    RAGLogger.log_qa_pair(query, result["answer"], retrieval_info)

                else:
                    logger.warning(f"No relevant documents found for query: '{query}'")
                    print("❌ No relevant documents found or answer generation failed.")

                    error_msg = result.get("error", "No relevant documents found.")
                    print(f"🔍 Details: {error_msg}")

                    # Log the unsuccessful query
                    RAGLogger.log_qa_pair(
                        query, f"Failed: {error_msg}", {"num_results": 0}
                    )

                # Clean up after each query
                if "result" in locals():
                    del result
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
    main()
