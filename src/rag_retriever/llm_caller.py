"""
Simple LLM Caller for RAG System
===============================

Main LLM caller that works with different model types through the factory.
"""

from typing import Any, Dict, List, Optional

from langchain.schema import Document

from ..constant import PROMPT
from ..utils.logging import get_logger
from .models.base_llm import BaseLLM
from .models.factory import LLMFactory

logger = get_logger("rag.llm.caller")


class RAGLLMCaller:
    """
    All-in-one LLM caller: prepare prompt, init model, call LLM, return structured output.

    Now supports multiple model types through the factory system.
    """

    def __init__(self, model_type: str = "gemini", **model_kwargs):
        """
        Initialize RAG LLM caller.

        Args:
            model_type: Type of model to use ("gemini", etc.)
            **model_kwargs: Additional parameters for the model
        """
        self.model_type = model_type
        self.model_kwargs = model_kwargs
        self.llm: Optional[BaseLLM] = None

        logger.info(f"Initialized RAG LLM caller with model type: {model_type}")

    def _init_llm(self, **kwargs):
        """Initialize LLM if not already done"""
        if self.llm is None:
            # Combine init kwargs with runtime kwargs
            combined_kwargs = {**self.model_kwargs, **kwargs}
            self.llm = LLMFactory.create_llm(self.model_type, **combined_kwargs)

    def _prepare_context(self, documents: List[Document]) -> str:
        """Format documents into context string"""
        if not documents:
            return "Không tìm thấy thông tin liên quan."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.page_content.strip()
            if content:
                context_parts.append(f"[Tài liệu {i}]:\n{content}")

        return "\n\n".join(context_parts)

    def _prepare_prompt(self, query: str, context: str) -> str:
        """Create final prompt from query and context"""
        return PROMPT.format(context=context, query=query)

    def generate_answer(
        self, query: str, documents: List[Document], **llm_kwargs
    ) -> Dict[str, Any]:
        """
        Complete pipeline: documents → context → prompt → LLM → structured output

        Args:
            query: User question
            documents: Retrieved documents
            **llm_kwargs: Optional LLM parameters (model_id, max_tokens, temperature)

        Returns:
            Dict with answer, context_length, success status
        """
        try:
            # Step 1: Prepare context
            context = self._prepare_context(documents)

            # Step 2: Prepare prompt
            prompt = self._prepare_prompt(query, context)

            # Step 3: Initialize LLM
            self._init_llm(**llm_kwargs)

            # Step 4: Call model
            answer = self.llm.generate(prompt)

            # Step 5: Return structured output
            return {
                "answer": answer,
                "context_length": len(context),
                "model_type": self.model_type,
                "success": True,
                "error": None,
            }

        except Exception as e:
            logger.error(f"RAG LLM call failed: {e}")
            return {
                "answer": f"Xin lỗi, có lỗi xảy ra: {str(e)}",
                "context_length": 0,
                "model_type": self.model_type,
                "success": False,
                "error": str(e),
            }


# Global instance with default model
_rag_llm_caller = None


def get_rag_llm_caller(model_type: str = "gemini", **kwargs) -> RAGLLMCaller:
    """
    Get global RAG LLM caller instance.

    Args:
        model_type: Type of model to use
        **kwargs: Model parameters

    Returns:
        RAGLLMCaller instance
    """
    global _rag_llm_caller
    if _rag_llm_caller is None or _rag_llm_caller.model_type != model_type:
        _rag_llm_caller = RAGLLMCaller(model_type=model_type, **kwargs)
    return _rag_llm_caller
