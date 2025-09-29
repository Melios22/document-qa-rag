"""
Simple RAG API for Streamlit
============================

Just the essential functions you need.
"""

import os
from typing import Dict, List

# Configure for clean operation
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from src import (
    HAS_RAG_RETRIEVER,
    VietnameseRAG,
    load_retrieval_models,
    require_rag_retriever,
)
from src.constant import COLLECTION_NAME, PROMPT

# Ensure we have the retrieval components
if not HAS_RAG_RETRIEVER:
    require_rag_retriever()

from src.rag_builder.connection_manager import get_milvus_client

# Global instances - loaded once
_rag = None
_models_loaded = False


def _ensure_models_loaded():
    """Load models once and keep them in memory."""
    global _rag, _models_loaded

    if not _models_loaded:
        print("Loading RAG models (one time only)...")
        # Load models and client
        client, _ = get_milvus_client()
        _, embedding_model, reranker_model = load_retrieval_models()
        _rag = VietnameseRAG(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
        )
        _models_loaded = True
        print("✅ Models loaded successfully!")

    return _rag


def ask_question_with_stages(question: str, status_callback=None) -> Dict:
    """
    Ask a question and get an answer with stage-by-stage status updates.

    Args:
        question: The question to ask
        status_callback: Function to call with status updates (text)

    Returns:
        {
            "answer": str,
            "sources": list,
            "confidence": float
        }
    """
    try:
        rag = _ensure_models_loaded()

        # Stage 1: Retrieval
        if status_callback:
            status_callback("🔍 Retrieving relevant documents...")

        documents, scores = rag.retriever.retrieve_and_rerank(question)

        if not documents:
            return {
                "answer": "No relevant documents found for your question.",
                "sources": [],
                "confidence": 0.0,
            }

        # Stage 2: Reranking (already done in retrieve_and_rerank, but show status)
        if status_callback:
            status_callback("📊 Reranking documents by relevance...")

        # Prepare sources for return
        sources = []
        for doc, score in zip(documents, scores):
            sources.append(
                {
                    "content": doc.page_content,
                    "filename": doc.metadata.get("source_filename", "Unknown"),
                    "score": score,
                    "metadata": doc.metadata,
                }
            )

        # Stage 3: LLM Generation
        if status_callback:
            status_callback("🤖 Generating answer with LLM...")

        llm_result = None
        # Step 2: LLM Call - Comment this line to return prompt instead of answer
        llm_result = rag.generator.generate_answer(question, documents)

        # If LLM call is active (uncommented above), return the answer
        if llm_result and llm_result.get("answer"):
            return {
                "answer": llm_result["answer"],
                "sources": sources,
                "confidence": llm_result.get("confidence", 0.0),
            }
        else:
            # If LLM call failed, return prompt using config template
            context = "\n\n".join([doc.page_content for doc in documents[:3]])
            formatted_prompt = PROMPT.format(query=question, context=context)
            return {
                "answer": f"LLM Call Failed. Here's the prompt that would be sent:\n\n{formatted_prompt}",
                "sources": sources,
                "confidence": 0.0,
            }

    except Exception as e:
        return {"answer": f"Error: {e}", "sources": [], "confidence": 0.0}


def search_documents(query: str, top_k: int = 3) -> List[Dict]:
    """
    Search for documents.

    Returns:
        List of {
            "content": str,
            "source": str,
            "score": float,
            "metadata": dict
        }
    """
    try:
        rag = _ensure_models_loaded()
        results = rag.search(query, initial_k=10, final_k=top_k)

        formatted = []
        for doc, score in results:
            formatted.append(
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source_filename", "Unknown"),
                    "score": score,
                    "metadata": doc.metadata,
                }
            )

        return formatted
    except Exception as e:
        return []


# ==========================================
# TEMPORARY TEST VARIABLES FOR LLM TESTING
# ==========================================

# Test question
TEST_QUESTION = "CNN là gì và nó hoạt động như thế nào?"

# Mock documents for testing (simulating retrieved documents)
TEST_DOCUMENTS = [
    {
        "page_content": """CNN (Convolutional Neural Network) là một loại mạng nền tảng sâu được thiết kế đặc biệt để xử lý dữ liệu có cấu trúc dạng lưới như hình ảnh. CNN sử dụng các phép tích chập (convolution) để trích xuất đặc trưng từ dữ liệu đầu vào. Kiến trúc CNN bao gồm các lớp tích chập, lớp pooling và lớp fully connected. CNN đặc biệt hiệu quả trong các tác vụ thị giác máy tính như phân loại hình ảnh, phát hiện đối tượng.""",
        "metadata": {
            "source_filename": "deep_learning_guide.pdf",
            "page_number": 15,
            "chunk_id": "chunk_123",
        },
    },
    {
        "page_content": """Mạng CNN hoạt động bằng cách áp dụng các bộ lọc (filter) lên dữ liệu đầu vào để tạo ra feature map. Quá trình này giúp mạng học các đặc trưng cục bộ như cạnh, góc, và texture. Lớp pooling giúp giảm kích thước không gian và tăng tính bất biến cho phép dịch chuyển. Cuối cùng, các lớp fully connected sẽ thực hiện phân loại dựa trên các đặc trưng đã được trích xuất.""",
        "metadata": {
            "source_filename": "neural_networks.pdf",
            "page_number": 42,
            "chunk_id": "chunk_456",
        },
    },
    {
        "page_content": """CNN được ứng dụng rộng rãi trong nhiều lĩnh vực: nhận dạng hình ảnh, xử lý video, y học (phân tích hình ảnh y tế), xe tự lái (nhận dạng biển báo, phát hiện vật cản). Các kiến trúc CNN nổi tiếng bao gồm LeNet, AlexNet, VGG, ResNet, và Inception. Mỗi kiến trúc có những cải tiến riêng để tăng hiệu suất và độ chính xác.""",
        "metadata": {
            "source_filename": "computer_vision_applications.pdf",
            "page_number": 8,
            "chunk_id": "chunk_789",
        },
    },
]


def test_llm_call_only():
    """
    Test function to call LLM directly with mock data - no embedding/retrieval needed.
    This bypasses all the retrieval steps and tests just the LLM generation.
    """
    print("=== Testing LLM Call Only ===")
    print(f"Question: {TEST_QUESTION}")
    print()

    try:
        # Convert mock documents to LangChain Document format
        from langchain.schema import Document

        mock_documents = [
            Document(page_content=doc["page_content"], metadata=doc["metadata"])
            for doc in TEST_DOCUMENTS
        ]

        try:
            if not HAS_RAG_RETRIEVER:
                print("❌ RAG retriever not available, skipping real LLM test")
                return

            # Try to load just the LLM components
            from src.rag_retriever.llm_caller import RAGLLMCaller

            # Create LLM caller with model type string (not object)
            llm_caller = RAGLLMCaller(model_type="gemini")  # Use string, not object
            result = llm_caller.generate_answer(TEST_QUESTION, mock_documents)

            print(result)

        except Exception as e:
            print(f"❌ Real LLM test failed: {e}")
            print("This is expected if LLM dependencies are not configured")

    except Exception as e:
        print(f"❌ Test failed: {e}")


# Test if run directly
if __name__ == "__main__":
    # test_llm_call_only()
    # print("\n" + "=" * 60 + "\n")

    result = ask_question_with_stages("OTA là gì?")
    print(f"Answer: {result['answer']}...")
