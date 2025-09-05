"""
Base LLM Interface
=================

Common interface for all LLM models in the RAG system.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseLLM(ABC):
    """Base class for all LLM implementations"""

    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        """
        Initialize LLM with common parameters.

        Args:
            model_id: Model identifier
            api_key: API key for the service
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        self.model_id = model_id
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate response from prompt.

        Args:
            prompt: Input prompt

        Returns:
            Generated text response
        """
        pass
