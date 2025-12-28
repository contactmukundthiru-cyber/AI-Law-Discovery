"""
HuggingFace model interface for inverse scaling experiments.

Supports both local models and HuggingFace Inference API.
"""

import os
import time
from datetime import datetime
from typing import Optional, Dict, Any, List

from .base import ModelInterface, ModelConfig, ModelResponse, RateLimiter, RetryHandler


class HuggingFaceModel(ModelInterface):
    """Interface for HuggingFace models (local and API)."""

    def __init__(self, config: ModelConfig, use_api: bool = False):
        super().__init__(config)
        self.use_api = use_api

        if use_api:
            self._setup_api_client()
        else:
            self._setup_local_model()

        self.rate_limiter = RateLimiter(
            requests_per_minute=config.extra_params.get("requests_per_minute", 100),
            tokens_per_minute=config.extra_params.get("tokens_per_minute", 500000),
        )
        self.retry_handler = RetryHandler()

    def _setup_api_client(self):
        """Setup HuggingFace Inference API client."""
        try:
            from huggingface_hub import InferenceClient
            self._InferenceClient = InferenceClient
        except ImportError:
            raise ImportError("huggingface_hub required. Install with: pip install huggingface-hub")

        api_key = self.config.api_key or os.environ.get("HUGGINGFACE_API_KEY")
        self.client = self._InferenceClient(model=self.config.name, token=api_key)
        self.model = None
        self.tokenizer = None

    def _setup_local_model(self):
        """Setup local HuggingFace model."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._torch = torch
            self._AutoModelForCausalLM = AutoModelForCausalLM
            self._AutoTokenizer = AutoTokenizer
        except ImportError:
            raise ImportError("transformers and torch required. Install with: pip install transformers torch")

        self.client = None

        # Determine device
        if self._torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(self._torch.backends, "mps") and self._torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # Load model and tokenizer
        model_name = self.config.name

        self.tokenizer = self._AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with appropriate settings
        load_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto" if self.device == "cuda" else None,
        }

        # Use bfloat16 if available on CUDA
        if self.device == "cuda":
            load_kwargs["torch_dtype"] = self._torch.bfloat16

        self.model = self._AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs,
        )

        if self.device != "cuda":
            self.model = self.model.to(self.device)

        self.model.eval()

    def generate(self, prompt: str) -> ModelResponse:
        """Generate a response."""
        if self.use_api:
            return self._generate_api(prompt)
        else:
            return self._generate_local(prompt)

    def _generate_api(self, prompt: str) -> ModelResponse:
        """Generate using HuggingFace Inference API."""
        self.rate_limiter.wait_if_needed()

        start_time = time.time()
        error = None
        text = ""
        tokens_used = None
        finish_reason = None

        for attempt in range(self.retry_handler.max_retries + 1):
            try:
                response = self.client.text_generation(
                    prompt,
                    max_new_tokens=self.config.max_tokens,
                    temperature=max(self.config.temperature, 0.01),  # API requires > 0
                    do_sample=self.config.temperature > 0,
                )

                text = response
                finish_reason = "stop"
                self._request_count += 1
                self.rate_limiter.record_request()
                break

            except Exception as e:
                if self.retry_handler.should_retry(attempt, e):
                    delay = self.retry_handler.get_delay(attempt)
                    time.sleep(delay)
                else:
                    error = str(e)
                    break

        latency_ms = (time.time() - start_time) * 1000

        return ModelResponse(
            text=text,
            model_name=self.config.name,
            provider=self.config.provider,
            prompt=prompt,
            latency_ms=latency_ms,
            timestamp=datetime.now(),
            tokens_used=tokens_used,
            finish_reason=finish_reason,
            error=error,
        )

    def _generate_local(self, prompt: str) -> ModelResponse:
        """Generate using local model."""
        start_time = time.time()
        error = None
        text = ""
        tokens_used = None
        finish_reason = None

        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.device)

            input_length = inputs.input_ids.shape[1]

            # Generate
            with self._torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=max(self.config.temperature, 0.01) if self.config.temperature > 0 else None,
                    do_sample=self.config.temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only new tokens
            generated_tokens = outputs[0][input_length:]
            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            tokens_used = len(outputs[0])
            finish_reason = "stop"
            self._request_count += 1
            self._total_tokens += tokens_used

        except Exception as e:
            error = str(e)

        latency_ms = (time.time() - start_time) * 1000

        return ModelResponse(
            text=text,
            model_name=self.config.name,
            provider=self.config.provider,
            prompt=prompt,
            latency_ms=latency_ms,
            timestamp=datetime.now(),
            tokens_used=tokens_used,
            finish_reason=finish_reason,
            error=error,
        )

    def is_available(self) -> bool:
        """Check if model is available."""
        if self.use_api:
            try:
                self.client.text_generation("test", max_new_tokens=1)
                return True
            except Exception:
                return False
        else:
            return self.model is not None

    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the model."""
        info = {
            "name": self.config.name,
            "provider": "huggingface",
            "estimated_params": self.config.estimated_params,
            "use_api": self.use_api,
        }

        if not self.use_api and self.model is not None:
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            info["actual_params"] = total_params
            info["device"] = str(self.device)

        return info

    def unload(self) -> None:
        """Unload model to free memory."""
        if not self.use_api:
            if self.model is not None:
                del self.model
                self.model = None
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            if self.device == "cuda":
                self._torch.cuda.empty_cache()
