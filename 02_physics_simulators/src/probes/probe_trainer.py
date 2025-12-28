"""
Training pipeline for probes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score
import logging

from .linear_probe import LinearProbe, MLPProbe, RegressionProbe

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Results from probe training."""
    train_accuracy: float
    val_accuracy: float
    train_loss: float
    val_loss: float
    train_history: List[float]
    val_history: List[float]
    best_epoch: int
    probe_weights: Optional[np.ndarray] = None


class ProbeTrainer:
    """
    Trainer for linear and MLP probes.

    Handles:
    - Train/validation splitting
    - Early stopping
    - Learning rate scheduling
    - Metrics tracking
    """

    def __init__(
        self,
        probe_type: str = "linear",
        hidden_dim: int = 256,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 50,
        patience: int = 10,
        device: str = "auto",
    ):
        self.probe_type = probe_type
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def train(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        task_type: str = "classification",
        validation_split: float = 0.2,
    ) -> Tuple[nn.Module, TrainingResult]:
        """
        Train a probe on activations.

        Args:
            activations: (n_samples, hidden_dim) activation matrix
            labels: (n_samples,) target labels
            task_type: "classification" or "regression"
            validation_split: Fraction for validation

        Returns:
            Trained probe and training results
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            activations, labels,
            test_size=validation_split,
            random_state=42,
            stratify=labels if task_type == "classification" else None,
        )

        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)

        if task_type == "classification":
            y_train = torch.LongTensor(y_train).to(self.device)
            y_val = torch.LongTensor(y_val).to(self.device)
            num_classes = len(np.unique(labels))
        else:
            y_train = torch.FloatTensor(y_train).to(self.device)
            y_val = torch.FloatTensor(y_val).to(self.device)
            num_classes = 1

        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        # Create probe
        input_dim = activations.shape[1]

        if task_type == "classification":
            if self.probe_type == "linear":
                probe = LinearProbe(input_dim, num_classes).to(self.device)
            else:
                probe = MLPProbe(input_dim, num_classes, self.hidden_dim).to(self.device)
            criterion = nn.CrossEntropyLoss()
        else:
            probe = RegressionProbe(input_dim).to(self.device)
            criterion = nn.MSELoss()

        optimizer = optim.Adam(probe.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )

        # Training loop
        train_history = []
        val_history = []
        best_val_loss = float('inf')
        best_epoch = 0
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training
            probe.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = probe(batch_x)
                if task_type == "regression":
                    outputs = outputs.squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_history.append(train_loss)

            # Validation
            probe.eval()
            with torch.no_grad():
                val_outputs = probe(X_val)
                if task_type == "regression":
                    val_outputs = val_outputs.squeeze()
                val_loss = criterion(val_outputs, y_val).item()

            val_history.append(val_loss)
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Load best model
        if best_state:
            probe.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        # Compute final metrics
        probe.eval()
        with torch.no_grad():
            train_preds = probe.predict(X_train).cpu().numpy()
            val_preds = probe.predict(X_val).cpu().numpy()

        if task_type == "classification":
            train_acc = accuracy_score(y_train.cpu().numpy(), train_preds)
            val_acc = accuracy_score(y_val.cpu().numpy(), val_preds)
        else:
            train_acc = r2_score(y_train.cpu().numpy(), train_preds)
            val_acc = r2_score(y_val.cpu().numpy(), val_preds)

        # Get probe weights if linear
        probe_weights = None
        if hasattr(probe, 'get_weights'):
            probe_weights = probe.get_weights()

        result = TrainingResult(
            train_accuracy=train_acc,
            val_accuracy=val_acc,
            train_loss=train_history[-1],
            val_loss=best_val_loss,
            train_history=train_history,
            val_history=val_history,
            best_epoch=best_epoch,
            probe_weights=probe_weights,
        )

        return probe, result

    def cross_validate(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        n_folds: int = 5,
        task_type: str = "classification",
    ) -> Dict[str, Any]:
        """
        Perform cross-validation for probe training.

        Returns:
            Dictionary with mean and std of metrics across folds
        """
        from sklearn.model_selection import StratifiedKFold, KFold

        if task_type == "classification":
            kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        else:
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        fold_accuracies = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(activations, labels)):
            X_train, X_val = activations[train_idx], activations[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]

            # Create full training set for this fold
            full_activations = np.concatenate([X_train, X_val])
            full_labels = np.concatenate([y_train, y_val])

            probe, result = self.train(
                full_activations, full_labels,
                task_type=task_type,
                validation_split=len(val_idx) / len(full_labels),
            )

            fold_accuracies.append(result.val_accuracy)

        return {
            "mean_accuracy": np.mean(fold_accuracies),
            "std_accuracy": np.std(fold_accuracies),
            "fold_accuracies": fold_accuracies,
        }
