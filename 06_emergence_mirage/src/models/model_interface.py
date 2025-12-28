"""
Model interface for emergence analysis.

Provides unified access to models of different scales with probability outputs.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelOutput:
    """Output from model evaluation."""
    prediction: str
    log_probability: float
    token_probabilities: List[Tuple[str, float]]
    top_k_predictions: List[Tuple[str, float]]
    entropy: float
    raw_logits: Optional[np.ndarray] = None


class ModelInterface(ABC):
    """Abstract base class for model interfaces."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.0
    ) -> ModelOutput:
        """Generate output with probability information."""
        pass

    @abstractmethod
    def get_token_probabilities(
        self,
        prompt: str,
        target: str
    ) -> Tuple[float, List[Tuple[str, float]]]:
        """Get probability of specific target completion."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier."""
        pass

    @property
    @abstractmethod
    def parameter_count(self) -> int:
        """Number of parameters."""
        pass


class HuggingFaceModelInterface(ModelInterface):
    """Interface for HuggingFace transformers models."""

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        dtype: torch.dtype = torch.float16
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self._device = device

        logger.info(f"Loading model: {model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device if device == "auto" else None
        )

        if device != "auto":
            self.model = self.model.to(device)

        self.model.eval()

        # Cache parameter count
        self._param_count = sum(p.numel() for p in self.model.parameters())

        logger.info(f"Model loaded: {self._param_count:,} parameters")

    @property
    def model_name(self) -> str:
        return self.model_id

    @property
    def parameter_count(self) -> int:
        return self._param_count

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.0
    ) -> ModelOutput:
        """Generate with probability tracking."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 1e-7),  # Avoid division by zero
                do_sample=temperature > 0,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        prediction = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Compute log probabilities and entropy
        log_prob = 0.0
        token_probs = []
        entropies = []

        for i, scores in enumerate(outputs.scores):
            probs = torch.softmax(scores[0], dim=-1)
            token_id = generated_ids[i].item()
            token_prob = probs[token_id].item()

            log_prob += np.log(token_prob + 1e-10)

            token_str = self.tokenizer.decode([token_id])
            token_probs.append((token_str, token_prob))

            # Compute entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            entropies.append(entropy)

        # Get top-k predictions for first generated token
        if outputs.scores:
            first_probs = torch.softmax(outputs.scores[0][0], dim=-1)
            top_k_indices = torch.topk(first_probs, k=min(10, len(first_probs))).indices
            top_k = [
                (self.tokenizer.decode([idx.item()]), first_probs[idx].item())
                for idx in top_k_indices
            ]
        else:
            top_k = []

        return ModelOutput(
            prediction=prediction.strip(),
            log_probability=log_prob,
            token_probabilities=token_probs,
            top_k_predictions=top_k,
            entropy=np.mean(entropies) if entropies else 0.0
        )

    def get_token_probabilities(
        self,
        prompt: str,
        target: str
    ) -> Tuple[float, List[Tuple[str, float]]]:
        """Get probability of specific target completion."""
        full_text = prompt + target

        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_inputs['input_ids'].shape[1]

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        log_prob = 0.0
        token_probs = []

        # Get probabilities for target tokens
        for i in range(prompt_len - 1, inputs['input_ids'].shape[1] - 1):
            probs = torch.softmax(logits[0, i], dim=-1)
            next_token_id = inputs['input_ids'][0, i + 1].item()
            token_prob = probs[next_token_id].item()

            log_prob += np.log(token_prob + 1e-10)

            token_str = self.tokenizer.decode([next_token_id])
            token_probs.append((token_str, token_prob))

        return log_prob, token_probs


class APIModelInterface(ModelInterface):
    """Interface for API-based models (OpenAI, Anthropic)."""

    def __init__(
        self,
        provider: str,
        model_id: str,
        api_key: Optional[str] = None
    ):
        self.provider = provider
        self.model_id = model_id
        self._model_name = f"{provider}/{model_id}"

        # Approximate parameter counts for known models
        self._param_estimates = {
            "gpt-3.5-turbo": 20_000_000_000,
            "gpt-4": 1_800_000_000_000,
            "claude-3-sonnet": 70_000_000_000,
            "claude-3-opus": 200_000_000_000,
        }

        if provider == "openai":
            import openai
            if api_key:
                self.client = openai.OpenAI(api_key=api_key)
            else:
                self.client = openai.OpenAI()
        elif provider == "anthropic":
            import anthropic
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)
            else:
                self.client = anthropic.Anthropic()
        else:
            raise ValueError(f"Unknown provider: {provider}")

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def parameter_count(self) -> int:
        return self._param_estimates.get(self.model_id, 0)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.0
    ) -> ModelOutput:
        """Generate with API model."""
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                logprobs=True,
                top_logprobs=10
            )

            content = response.choices[0].message.content

            # Extract log probabilities if available
            logprobs_data = response.choices[0].logprobs
            if logprobs_data and logprobs_data.content:
                token_probs = [
                    (tp.token, np.exp(tp.logprob))
                    for tp in logprobs_data.content
                ]
                log_prob = sum(tp.logprob for tp in logprobs_data.content)

                # Get top-k from first token
                if logprobs_data.content[0].top_logprobs:
                    top_k = [
                        (tlp.token, np.exp(tlp.logprob))
                        for tlp in logprobs_data.content[0].top_logprobs
                    ]
                else:
                    top_k = []
            else:
                token_probs = []
                log_prob = 0.0
                top_k = []

            return ModelOutput(
                prediction=content.strip(),
                log_probability=log_prob,
                token_probabilities=token_probs,
                top_k_predictions=top_k,
                entropy=0.0  # Not available from API
            )

        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model_id,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )

            content = response.content[0].text

            # Anthropic doesn't provide logprobs currently
            return ModelOutput(
                prediction=content.strip(),
                log_probability=0.0,
                token_probabilities=[],
                top_k_predictions=[],
                entropy=0.0
            )

    def get_token_probabilities(
        self,
        prompt: str,
        target: str
    ) -> Tuple[float, List[Tuple[str, float]]]:
        """Get probability of target (limited for API models)."""
        # For API models, we can only approximate via generation
        output = self.generate(prompt + target[:10], max_tokens=1, temperature=0.0)
        return output.log_probability, output.token_probabilities


def create_model_interface(
    model_config: Dict[str, Any],
    device: str = "auto"
) -> ModelInterface:
    """Factory function to create model interfaces."""
    model_id = model_config.get("model_id", "")
    provider = model_config.get("provider", "huggingface")

    if provider == "huggingface":
        return HuggingFaceModelInterface(model_id, device=device)
    elif provider in ["openai", "anthropic"]:
        return APIModelInterface(provider, model_id)
    else:
        raise ValueError(f"Unknown provider: {provider}")
