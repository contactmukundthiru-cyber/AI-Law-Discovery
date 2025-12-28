"""
OpenAI model interface for inverse scaling experiments.
"""

import os
import time
from datetime import datetime
from typing import Optional, Dict, Any

from .base import ModelInterface, ModelConfig, ModelResponse, RateLimiter, RetryHandler


class OpenAIModel(ModelInterface):
    """Interface for OpenAI models."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        # Import here to avoid dependency issues if not using OpenAI
        try:
            from openai import OpenAI
            self._client_class = OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

        api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key in config.")

        self.client = self._client_class(api_key=api_key)
        self.rate_limiter = RateLimiter(
            requests_per_minute=config.extra_params.get("requests_per_minute", 60),
            tokens_per_minute=config.extra_params.get("tokens_per_minute", 90000),
        )
        self.retry_handler = RetryHandler()

    def generate(self, prompt: str) -> ModelResponse:
        """Generate a response using OpenAI API."""
        self.rate_limiter.wait_if_needed(estimated_tokens=len(prompt.split()) * 2)

        start_time = time.time()
        error = None
        text = ""
        tokens_used = None
        finish_reason = None
        raw_response = None

        for attempt in range(self.retry_handler.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout,
                )

                text = response.choices[0].message.content or ""
                finish_reason = response.choices[0].finish_reason
                tokens_used = response.usage.total_tokens if response.usage else None
                raw_response = response.model_dump()

                self._request_count += 1
                if tokens_used:
                    self._total_tokens += tokens_used
                    self.rate_limiter.record_request(tokens_used)
                else:
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
            raw_response=raw_response,
            error=error,
        )

    def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        try:
            # Simple API check
            self.client.models.list()
            return True
        except Exception:
            return False

    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the model."""
        try:
            models = self.client.models.list()
            for model in models.data:
                if model.id == self.config.name:
                    return {
                        "id": model.id,
                        "created": model.created,
                        "owned_by": model.owned_by,
                    }
            return None
        except Exception:
            return None
