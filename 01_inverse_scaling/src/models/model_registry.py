"""
Model registry for inverse scaling experiments.

Provides centralized management and access to models from different providers.
"""

from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass
import yaml
from pathlib import Path

from .base import ModelInterface, ModelConfig
from .openai_models import OpenAIModel
from .anthropic_models import AnthropicModel
from .huggingface_models import HuggingFaceModel


@dataclass
class RegisteredModel:
    """Information about a registered model."""
    name: str
    provider: str
    estimated_params: str
    model_class: Type[ModelInterface]
    enabled: bool = True
    extra_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}


class ModelRegistry:
    """
    Central registry for all models used in experiments.

    Handles model instantiation, configuration, and lifecycle.
    """

    # Provider to model class mapping
    PROVIDER_CLASSES = {
        "openai": OpenAIModel,
        "anthropic": AnthropicModel,
        "huggingface": HuggingFaceModel,
    }

    def __init__(self):
        self._registered: Dict[str, RegisteredModel] = {}
        self._instances: Dict[str, ModelInterface] = {}

    def register(
        self,
        name: str,
        provider: str,
        estimated_params: str,
        enabled: bool = True,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a model.

        Args:
            name: Model name/identifier
            provider: Provider name (openai, anthropic, huggingface)
            estimated_params: Estimated parameter count (e.g., "7B", "70B")
            enabled: Whether model is enabled for experiments
            extra_params: Additional provider-specific parameters
        """
        if provider not in self.PROVIDER_CLASSES:
            raise ValueError(f"Unknown provider: {provider}. "
                           f"Available: {list(self.PROVIDER_CLASSES.keys())}")

        model_class = self.PROVIDER_CLASSES[provider]
        key = f"{provider}/{name}"

        self._registered[key] = RegisteredModel(
            name=name,
            provider=provider,
            estimated_params=estimated_params,
            model_class=model_class,
            enabled=enabled,
            extra_params=extra_params or {},
        )

    def register_from_config(self, config_path: Path) -> None:
        """
        Register models from a configuration file.

        Args:
            config_path: Path to YAML configuration file
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        models_config = config.get("models", {})

        for provider, models in models_config.items():
            if not isinstance(models, list):
                continue

            for model_info in models:
                self.register(
                    name=model_info["name"],
                    provider=model_info.get("provider", provider),
                    estimated_params=model_info.get("estimated_params", "unknown"),
                    enabled=model_info.get("enabled", True),
                    extra_params=model_info.get("extra_params"),
                )

    def get_model(
        self,
        name: str,
        provider: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        api_key: Optional[str] = None,
    ) -> ModelInterface:
        """
        Get a model instance.

        Args:
            name: Model name
            provider: Provider (inferred if not specified)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            api_key: API key (uses environment variable if not provided)

        Returns:
            Model interface instance
        """
        # Determine key
        if provider:
            key = f"{provider}/{name}"
        else:
            # Try to find model by name alone
            matching = [k for k in self._registered if k.endswith(f"/{name}")]
            if len(matching) == 1:
                key = matching[0]
            elif len(matching) > 1:
                raise ValueError(f"Ambiguous model name '{name}'. "
                               f"Matches: {matching}. Specify provider.")
            else:
                raise ValueError(f"Model not found: {name}")

        if key not in self._registered:
            raise ValueError(f"Model not registered: {key}")

        registered = self._registered[key]

        if not registered.enabled:
            raise ValueError(f"Model is disabled: {key}")

        # Check if instance already exists with same config
        instance_key = f"{key}_{temperature}_{max_tokens}"
        if instance_key in self._instances:
            return self._instances[instance_key]

        # Create model config
        config = ModelConfig(
            name=registered.name,
            provider=registered.provider,
            estimated_params=registered.estimated_params,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            extra_params=registered.extra_params,
        )

        # Instantiate model
        model = registered.model_class(config)
        self._instances[instance_key] = model

        return model

    def get_all_enabled(self) -> List[str]:
        """Get list of all enabled model keys."""
        return [k for k, v in self._registered.items() if v.enabled]

    def get_by_provider(self, provider: str) -> List[str]:
        """Get all models for a specific provider."""
        return [k for k, v in self._registered.items() if v.provider == provider]

    def get_by_scale(self, min_params: Optional[str] = None, max_params: Optional[str] = None) -> List[str]:
        """
        Get models within a parameter range.

        Args:
            min_params: Minimum parameters (e.g., "7B")
            max_params: Maximum parameters (e.g., "70B")

        Returns:
            List of model keys within range
        """
        def parse_params(s: str) -> float:
            """Parse parameter string to float."""
            s = s.upper().strip()
            multipliers = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
            for suffix, mult in multipliers.items():
                if s.endswith(suffix):
                    return float(s[:-1]) * mult
            try:
                return float(s)
            except ValueError:
                return float("inf")

        min_val = parse_params(min_params) if min_params else 0
        max_val = parse_params(max_params) if max_params else float("inf")

        result = []
        for key, model in self._registered.items():
            params = parse_params(model.estimated_params)
            if min_val <= params <= max_val:
                result.append(key)

        return result

    def list_models(self) -> List[Dict[str, Any]]:
        """Get information about all registered models."""
        return [
            {
                "key": key,
                "name": model.name,
                "provider": model.provider,
                "estimated_params": model.estimated_params,
                "enabled": model.enabled,
            }
            for key, model in self._registered.items()
        ]

    def enable(self, key: str) -> None:
        """Enable a model."""
        if key in self._registered:
            self._registered[key].enabled = True

    def disable(self, key: str) -> None:
        """Disable a model."""
        if key in self._registered:
            self._registered[key].enabled = False

    def clear_instances(self) -> None:
        """Clear all model instances (useful for memory management)."""
        for key, model in self._instances.items():
            if hasattr(model, "unload"):
                model.unload()
        self._instances.clear()


# Global registry instance
_global_registry = None


def get_registry() -> ModelRegistry:
    """Get the global model registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry()
    return _global_registry


def get_model(
    name: str,
    provider: Optional[str] = None,
    **kwargs,
) -> ModelInterface:
    """
    Convenience function to get a model from the global registry.

    Args:
        name: Model name
        provider: Provider name
        **kwargs: Additional arguments passed to registry.get_model()

    Returns:
        Model interface instance
    """
    registry = get_registry()
    return registry.get_model(name, provider, **kwargs)
