"""
Gemini LLM Model
===============

Google Gemini model implementation for RAG system.
"""

import os
from typing import Optional

from dotenv import load_dotenv
from google.genai import Client, types

from ...utils.logging import get_logger

logger = get_logger("rag.llm.gemini")


class GeminiLLM:
    """Google Gemini LLM implementation"""

    def __init__(
        self,
        model_id: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        max_tokens: int = 4096,  # Increased from 1024 to 4096
        temperature: float = 0.0,
        **kwargs,
    ):
        """
        Initialize Gemini LLM.

        Args:
            model_id: Gemini model identifier
            api_key: API key (gets from env if None)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        load_dotenv()
        self.model_id = model_id
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required")

        self.llm = Client(api_key=self.api_key)
        self.config = types.GenerateContentConfig(
            system_instruction="You are a helpful assistant.",
            max_output_tokens=max_tokens,
            temperature=temperature,
            # thinking_config=types.ThinkingConfig(
            #     max_steps=5,
            #     stop_sequences=["\n"],
            # ),
        )

        logger.info(f"Initialized Gemini LLM: {model_id}")

    def generate(self, prompt: str) -> str:
        """
        Generate response from prompt.

        Args:
            prompt: Input prompt

        Returns:
            Generated text response
        """
        try:
            response = self.llm.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=self.config,
            )
            result = response.text.strip()
            logger.info(f"Generated {len(result)} characters from Gemini")
            return result

        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return f"Error generating response: {e}"
