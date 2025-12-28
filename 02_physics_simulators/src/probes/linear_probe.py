"""
Linear and MLP probes for detecting physical variable encodings.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class ProbeResult:
    """Result from probe evaluation."""
    accuracy: float
    loss: float
    predictions: np.ndarray
    labels: np.ndarray
    confidence: np.ndarray


class LinearProbe(nn.Module):
    """
    Linear probe for detecting linear encodings of physical variables.

    Maps from activation space to target variable space using a single
    linear transformation.
    """

    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predictions (class labels for classification)."""
        with torch.no_grad():
            logits = self.forward(x)
            if self.output_dim > 1:
                return torch.argmax(logits, dim=-1)
            else:
                return (torch.sigmoid(logits) > 0.5).float()

    def get_weights(self) -> np.ndarray:
        """Get probe weights for analysis."""
        return self.linear.weight.detach().cpu().numpy()

    def get_direction(self) -> np.ndarray:
        """Get primary encoding direction (for binary probes)."""
        weights = self.get_weights()
        if self.output_dim == 1:
            return weights.flatten() / np.linalg.norm(weights)
        else:
            # Return first principal component of weight matrix
            u, s, vh = np.linalg.svd(weights)
            return vh[0]


class MLPProbe(nn.Module):
    """
    MLP probe with one hidden layer for detecting non-linear encodings.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            if self.output_dim > 1:
                return torch.argmax(logits, dim=-1)
            else:
                return (torch.sigmoid(logits) > 0.5).float()


class RegressionProbe(nn.Module):
    """
    Probe for continuous physical variables (e.g., distance, duration).
    """

    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(x)


class MultiTaskProbe(nn.Module):
    """
    Multi-task probe for simultaneously detecting multiple physical variables.
    """

    def __init__(
        self,
        input_dim: int,
        task_dims: Dict[str, int],
        shared_dim: int = 256,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.task_dims = task_dims

        # Shared representation
        self.shared = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.ReLU(),
        )

        # Task-specific heads
        self.heads = nn.ModuleDict({
            task: nn.Linear(shared_dim, dim)
            for task, dim in task_dims.items()
        })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_rep = self.shared(x)
        return {task: head(shared_rep) for task, head in self.heads.items()}

    def predict(self, x: torch.Tensor, task: str) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.forward(x)
            logits = outputs[task]
            if self.task_dims[task] > 1:
                return torch.argmax(logits, dim=-1)
            else:
                return (torch.sigmoid(logits) > 0.5).float()
