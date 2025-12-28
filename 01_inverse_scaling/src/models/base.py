"""
Base model interface for inverse scaling experiments.

Defines abstract interface that all model providers must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import time


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    provider: str
    estimated_params: str
    temperature: float = 0.0
    max_tokens: int = 256
    timeout: int = 60
    api_key: Optional[str] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        return f"{self.provider}/{self.name}"


@dataclass
class ModelResponse:
    """Response from a model."""
    text: str
    model_name: str
    provider: str
    prompt: str
    latency_ms: float
    timestamp: datetime
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "model_name": self.model_name,
            "provider": self.provider,
            "prompt": self.prompt,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
            "tokens_used": self.tokens_used,
            "finish_reason": self.finish_reason,
            "error": self.error,
        }


class ModelInterface(ABC):
    """Abstract base class for model interfaces."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._request_count = 0
        self._total_tokens = 0

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def provider(self) -> str:
        return self.config.provider

    @abstractmethod
    def generate(self, prompt: str) -> ModelResponse:
        """Generate a response for a single prompt."""
        pass

    def generate_batch(
        self,
        prompts: List[str],
        show_progress: bool = True,
    ) -> List[ModelResponse]:
        """Generate responses for multiple prompts."""
        responses = []
        for prompt in prompts:
            response = self.generate(prompt)
            responses.append(response)
        return responses

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is available."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "request_count": self._request_count,
            "total_tokens": self._total_tokens,
            "model_name": self.name,
            "provider": self.provider,
        }

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self._request_count = 0
        self._total_tokens = 0


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        tokens_per_minute: int = 90000,
    ):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self._request_times: List[float] = []
        self._token_counts: List[tuple] = []

    def wait_if_needed(self, estimated_tokens: int = 0) -> None:
        """Wait if rate limits would be exceeded."""
        now = time.time()
        minute_ago = now - 60

        # Clean old entries
        self._request_times = [t for t in self._request_times if t > minute_ago]
        self._token_counts = [(t, c) for t, c in self._token_counts if t > minute_ago]

        # Check request limit
        if len(self._request_times) >= self.requests_per_minute:
            sleep_time = self._request_times[0] - minute_ago
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Check token limit
        total_tokens = sum(c for _, c in self._token_counts)
        if total_tokens + estimated_tokens > self.tokens_per_minute:
            sleep_time = self._token_counts[0][0] - minute_ago if self._token_counts else 0
            if sleep_time > 0:
                time.sleep(sleep_time)

    def record_request(self, tokens: int = 0) -> None:
        """Record a request for rate limiting."""
        now = time.time()
        self._request_times.append(now)
        if tokens > 0:
            self._token_counts.append((now, tokens))


class RetryHandler:
    """Handler for retrying failed requests."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)

    def should_retry(self, attempt: int, error: Exception) -> bool:
        """Determine if request should be retried."""
        if attempt >= self.max_retries:
            return False

        # Retry on rate limits and transient errors
        error_str = str(error).lower()
        retryable_errors = [
            "rate limit",
            "timeout",
            "connection",
            "temporary",
            "503",
            "502",
            "429",
        ]
        return any(err in error_str for err in retryable_errors)
