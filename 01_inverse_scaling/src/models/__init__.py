"""
Models module for inverse scaling experiments.

Provides unified interfaces for accessing models from different providers
(OpenAI, Anthropic, HuggingFace) with consistent APIs for evaluation.
"""

from .base import ModelInterface, ModelResponse, ModelConfig
from .openai_models import OpenAIModel
from .anthropic_models import AnthropicModel
from .huggingface_models import HuggingFaceModel
from .model_registry import ModelRegistry, get_model

__all__ = [
    "ModelInterface",
    "ModelResponse",
    "ModelConfig",
    "OpenAIModel",
    "AnthropicModel",
    "HuggingFaceModel",
    "ModelRegistry",
    "get_model",
]
