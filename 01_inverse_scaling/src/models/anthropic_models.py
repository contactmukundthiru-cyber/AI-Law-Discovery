"""
Anthropic model interface for inverse scaling experiments.
"""

import os
import time
from datetime import datetime
from typing import Optional, Dict, Any

from .base import ModelInterface, ModelConfig, ModelResponse, RateLimiter, RetryHandler


class AnthropicModel(ModelInterface):
    """Interface for Anthropic Claude models."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        # Import here to avoid dependency issues if not using Anthropic
        try:
            import anthropic
            self._anthropic = anthropic
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key in config.")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.rate_limiter = RateLimiter(
            requests_per_minute=config.extra_params.get("requests_per_minute", 50),
            tokens_per_minute=config.extra_params.get("tokens_per_minute", 100000),
        )
        self.retry_handler = RetryHandler()

    def generate(self, prompt: str) -> ModelResponse:
        """Generate a response using Anthropic API."""
        self.rate_limiter.wait_if_needed(estimated_tokens=len(prompt.split()) * 2)

        start_time = time.time()
        error = None
        text = ""
        tokens_used = None
        finish_reason = None
        raw_response = None

        for attempt in range(self.retry_handler.max_retries + 1):
            try:
                response = self.client.messages.create(
                    model=self.config.name,
                    max_tokens=self.config.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                )

                # Extract text from response
                text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        text += block.text

                finish_reason = response.stop_reason
                tokens_used = response.usage.input_tokens + response.usage.output_tokens
                raw_response = {
                    "id": response.id,
                    "model": response.model,
                    "type": response.type,
                    "stop_reason": response.stop_reason,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    },
                }

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
        """Check if Anthropic API is available."""
        try:
            # Simple API check - use a minimal request
            self.client.messages.create(
                model=self.config.name,
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True
        except Exception:
            return False

    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the model."""
        # Anthropic doesn't have a models list endpoint
        # Return known information based on model name
        model_info = {
            "claude-3-haiku-20240307": {
                "name": "Claude 3 Haiku",
                "estimated_params": "20B",
                "context_window": 200000,
            },
            "claude-3-sonnet-20240229": {
                "name": "Claude 3 Sonnet",
                "estimated_params": "70B",
                "context_window": 200000,
            },
            "claude-3-opus-20240229": {
                "name": "Claude 3 Opus",
                "estimated_params": "200B+",
                "context_window": 200000,
            },
            "claude-3-5-sonnet-20241022": {
                "name": "Claude 3.5 Sonnet",
                "estimated_params": "Unknown",
                "context_window": 200000,
            },
        }
        return model_info.get(self.config.name)
