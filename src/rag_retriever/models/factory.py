"""
LLM Model Factory
================

Factory for creating different LLM model instances.
"""

from typing import Any, Dict, Optional

from ...utils.logging import get_logger
from .base_llm import BaseLLM
from .gemini_llm import GeminiLLM
from .watsonx_llm import WatsonxLLM

# Import other models as you add them
# from .your_model import YourModelLLM


logger = get_logger("rag.llm.factory")


class LLMFactory:
    """Factory for creating LLM model instances"""

    # Register available models here
    MODELS = {
        "gemini": GeminiLLM,
        "watsonx": WatsonxLLM,
        # "your_model": YourModelLLM,
        # Add more models here
    }

    @classmethod
    def create_llm(
        self, model_type: str = "gemini", model_id: Optional[str] = None, **kwargs
    ) -> BaseLLM:
        """
        Create an LLM instance.

        Args:
            model_type: Type of model ("gemini", "your_model", etc.)
            model_id: Specific model ID (uses default if None)
            **kwargs: Additional parameters for the model

        Returns:
            Initialized LLM instance

        Raises:
            ValueError: If model_type is not supported
        """
        if model_type not in self.MODELS:
            available = ", ".join(self.MODELS.keys())
            raise ValueError(
                f"Model '{model_type}' not supported. Available: {available}"
            )

        model_class = self.MODELS[model_type]

        # Set default model_id based on type
        if model_id is None:
            defaults = {
                "gemini": "gemini-2.5-flash",
                "watsonx": "ibm/granite-13b-chat-v2",
                # Add defaults for other models
            }
            model_id = defaults.get(model_type, "default-model")

        logger.info(f"Creating {model_type} LLM with model_id: {model_id}")
        return model_class(model_id=model_id, **kwargs)

    @classmethod
    def list_available_models(cls) -> list:
        """Get list of available model types"""
        return list(cls.MODELS.keys())
