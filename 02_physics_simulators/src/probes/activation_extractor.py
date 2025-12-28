"""
Activation extraction from transformer models for probing.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from transformers import AutoModel, AutoTokenizer
import logging

logger = logging.getLogger(__name__)


@dataclass
class ActivationData:
    """Container for extracted activations."""
    activations: Dict[int, np.ndarray]  # layer -> activations
    tokens: List[str]
    input_ids: np.ndarray
    attention_mask: np.ndarray
    metadata: Dict


class ActivationExtractor:
    """
    Extract activations from transformer models at specified layers.

    Supports extraction of:
    - Hidden states at each layer
    - Attention patterns
    - Specific token positions
    """

    def __init__(
        self,
        model_name: str,
        layers_to_extract: List[int],
        device: str = "auto",
        pooling: str = "last",  # "last", "mean", "first", "max"
    ):
        self.model_name = model_name
        self.layers_to_extract = layers_to_extract
        self.pooling = pooling

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True,
            output_attentions=True,
        ).to(self.device)
        self.model.eval()

        self.hidden_size = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers

        logger.info(f"Loaded {model_name} with {self.num_layers} layers, hidden_size={self.hidden_size}")

    def extract(
        self,
        texts: Union[str, List[str]],
        positions: Optional[List[int]] = None,
    ) -> ActivationData:
        """
        Extract activations for given texts.

        Args:
            texts: Input text(s)
            positions: Specific token positions to extract (None = use pooling)

        Returns:
            ActivationData containing extracted activations
        """
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**encoded)

        hidden_states = outputs.hidden_states  # tuple of (batch, seq, hidden)

        # Extract activations at specified layers
        activations = {}
        for layer_idx in self.layers_to_extract:
            if layer_idx < 0:
                layer_idx = self.num_layers + layer_idx + 1
            if layer_idx > self.num_layers:
                continue

            layer_hidden = hidden_states[layer_idx]  # (batch, seq, hidden)

            if positions is not None:
                # Extract specific positions
                pooled = layer_hidden[:, positions, :].mean(dim=1)
            else:
                # Apply pooling strategy
                pooled = self._pool_activations(
                    layer_hidden,
                    encoded.attention_mask,
                )

            activations[layer_idx] = pooled.cpu().numpy()

        # Get tokens for reference
        tokens = [
            self.tokenizer.convert_ids_to_tokens(ids)
            for ids in encoded.input_ids
        ]

        return ActivationData(
            activations=activations,
            tokens=tokens,
            input_ids=encoded.input_ids.cpu().numpy(),
            attention_mask=encoded.attention_mask.cpu().numpy(),
            metadata={"model": self.model_name, "pooling": self.pooling},
        )

    def _pool_activations(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply pooling to sequence of hidden states."""
        if self.pooling == "last":
            # Get last non-padding token
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]
            pooled = hidden_states[
                torch.arange(batch_size, device=self.device),
                seq_lengths,
            ]
        elif self.pooling == "first":
            pooled = hidden_states[:, 0, :]
        elif self.pooling == "mean":
            # Mean over non-padding tokens
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
        elif self.pooling == "max":
            # Max over non-padding tokens
            mask = attention_mask.unsqueeze(-1).float()
            hidden_states = hidden_states * mask + (1 - mask) * -1e9
            pooled = hidden_states.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return pooled

    def extract_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> Dict[int, np.ndarray]:
        """
        Extract activations for a large batch of texts.

        Args:
            texts: List of input texts
            batch_size: Processing batch size

        Returns:
            Dictionary mapping layer index to (n_samples, hidden_size) array
        """
        all_activations = {layer: [] for layer in self.layers_to_extract}

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            activation_data = self.extract(batch_texts)

            for layer, acts in activation_data.activations.items():
                all_activations[layer].append(acts)

        # Concatenate all batches
        return {
            layer: np.concatenate(acts, axis=0)
            for layer, acts in all_activations.items()
        }

    def get_attention_patterns(
        self,
        texts: Union[str, List[str]],
    ) -> Dict[int, np.ndarray]:
        """
        Extract attention patterns from the model.

        Returns:
            Dictionary mapping layer index to attention weights
        """
        if isinstance(texts, str):
            texts = [texts]

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded)

        attentions = outputs.attentions  # tuple of (batch, heads, seq, seq)

        return {
            layer: attn.cpu().numpy()
            for layer, attn in enumerate(attentions)
            if layer in self.layers_to_extract
        }
