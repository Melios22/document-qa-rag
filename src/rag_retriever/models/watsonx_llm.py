"""
IBM Watsonx LLM Model
====================

IBM Watsonx model implementation for RAG system.
"""

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_ibm import WatsonxLLM as Watsonx

from ...constant import MAX_OUTPUT_TOKENS, TEMPERATURE
from ...utils.logging import get_logger
from .base_llm import BaseLLM

logger = get_logger("rag.llm.watsonx")


class WatsonxLLM(BaseLLM):
    """IBM Watsonx LLM implementation"""

    def __init__(
        self,
        model_id: str = "ibm/granite-13b-chat-v2",
        api_key: Optional[str] = None,
        max_tokens: int = MAX_OUTPUT_TOKENS,
        temperature: float = TEMPERATURE,
        project_id: Optional[str] = None,
        url: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Watsonx LLM.

        Args:
            model_id: Watsonx model identifier
            api_key: API key (gets from env if None)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            project_id: Watsonx project ID
            url: Watsonx service URL
        """
        super().__init__(model_id, api_key, max_tokens, temperature)

        load_dotenv()
        # Standardize env var names
        self.project_id = project_id or os.getenv("WATSONX_PROJECT_ID")
        self.api_key = self.api_key or os.getenv("WATSONX_API_KEY")
        self.url = url or os.getenv("WATSONX_URL")

        if not all([self.project_id, self.api_key, self.url]):
            missing = []
            if not self.project_id:
                missing.append("WATSONX_PROJECT_ID")
            if not self.api_key:
                missing.append("WATSONX_API_KEY")
            if not self.url:
                missing.append("WATSONX_URL")
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")

        self.llm = Watsonx(
            model_id=self.model_id,
            url=self.url,
            apikey=self.api_key,
            project_id=self.project_id,
            params={
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "top_k": 50,
            },
        )

        logger.info(f"Initialized Watsonx LLM: {model_id}")

    def generate(self, prompt: str) -> str:
        """
        Generate response from prompt.

        Args:
            prompt: Input prompt

        Returns:
            Generated text response
        """
        try:
            response = self.llm.invoke(prompt)
            result = response.strip()
            logger.info(f"Generated {len(result)} characters from Watsonx")
            return result

        except Exception as e:
            logger.error(f"Watsonx generation failed: {e}")
            return f"Error generating response: {e}"
