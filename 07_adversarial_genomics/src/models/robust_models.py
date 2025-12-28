"""
Robust model loading and adversarial attack interfaces.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AdversarialExample:
    """Container for adversarial example."""
    original: torch.Tensor
    adversarial: torch.Tensor
    perturbation: torch.Tensor
    original_label: int
    adversarial_label: int
    attack_success: bool
    perturbation_norm: float


class RobustModelLoader:
    """
    Loads standard and adversarially robust models.
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self._models = {}

    def load_standard_model(self, name: str) -> nn.Module:
        """Load a standard (non-robust) pretrained model."""
        import torchvision.models as models

        model_registry = {
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            "vgg16": models.vgg16,
            "densenet121": models.densenet121,
        }

        if name not in model_registry:
            raise ValueError(f"Unknown model: {name}")

        model = model_registry[name](pretrained=True)
        model = model.to(self.device)
        model.eval()

        self._models[name] = model
        return model

    def load_robust_model(self, name: str) -> nn.Module:
        """Load an adversarially robust model."""
        try:
            from robustbench import load_model

            model = load_model(
                model_name=name,
                dataset='imagenet',
                threat_model='Linf'
            )
            model = model.to(self.device)
            model.eval()

            self._models[f"{name}_robust"] = model
            return model

        except ImportError:
            logger.warning("RobustBench not available, using standard model")
            return self.load_standard_model("resnet50")

    def get_feature_extractor(
        self,
        model: nn.Module,
        layer_names: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Create hooks to extract intermediate features."""
        features = {}
        hooks = []

        def get_hook(name):
            def hook(module, input, output):
                features[name] = output.detach()
            return hook

        for name, module in model.named_modules():
            if name in layer_names:
                hooks.append(module.register_forward_hook(get_hook(name)))

        return features, hooks


class AdversarialAttacker:
    """
    Generates adversarial examples using various attacks.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.device = device
        self.model.eval()

    def fgsm_attack(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        epsilon: float = 0.03
    ) -> AdversarialExample:
        """Fast Gradient Sign Method attack."""
        image = image.clone().detach().to(self.device)
        label = label.to(self.device)
        image.requires_grad = True

        output = self.model(image)
        original_pred = output.argmax(dim=1).item()

        loss = nn.CrossEntropyLoss()(output, label)
        self.model.zero_grad()
        loss.backward()

        # FGSM perturbation
        perturbation = epsilon * image.grad.sign()
        adversarial = torch.clamp(image + perturbation, 0, 1)

        # Get adversarial prediction
        with torch.no_grad():
            adv_output = self.model(adversarial)
            adv_pred = adv_output.argmax(dim=1).item()

        perturbation = adversarial - image.detach()

        return AdversarialExample(
            original=image.detach(),
            adversarial=adversarial.detach(),
            perturbation=perturbation.detach(),
            original_label=original_pred,
            adversarial_label=adv_pred,
            attack_success=adv_pred != label.item(),
            perturbation_norm=perturbation.norm().item()
        )

    def pgd_attack(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        epsilon: float = 0.03,
        alpha: float = 0.01,
        num_steps: int = 40
    ) -> AdversarialExample:
        """Projected Gradient Descent attack."""
        image = image.clone().detach().to(self.device)
        label = label.to(self.device)

        original_image = image.clone()
        adv_image = image.clone()

        with torch.no_grad():
            original_pred = self.model(image).argmax(dim=1).item()

        for _ in range(num_steps):
            adv_image.requires_grad = True

            output = self.model(adv_image)
            loss = nn.CrossEntropyLoss()(output, label)

            self.model.zero_grad()
            loss.backward()

            # Update adversarial image
            with torch.no_grad():
                adv_image = adv_image + alpha * adv_image.grad.sign()

                # Project to epsilon ball
                perturbation = adv_image - original_image
                perturbation = torch.clamp(perturbation, -epsilon, epsilon)
                adv_image = torch.clamp(original_image + perturbation, 0, 1)

        # Get final prediction
        with torch.no_grad():
            adv_pred = self.model(adv_image).argmax(dim=1).item()

        perturbation = adv_image - original_image

        return AdversarialExample(
            original=original_image,
            adversarial=adv_image,
            perturbation=perturbation,
            original_label=original_pred,
            adversarial_label=adv_pred,
            attack_success=adv_pred != label.item(),
            perturbation_norm=perturbation.norm().item()
        )

    def generate_adversarial_directions(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        attack: str = "pgd",
        **kwargs
    ) -> List[torch.Tensor]:
        """Generate adversarial perturbation directions for multiple images."""
        attack_fn = self.pgd_attack if attack == "pgd" else self.fgsm_attack

        directions = []
        for img, label in zip(images, labels):
            img = img.unsqueeze(0)
            label = label.unsqueeze(0) if label.dim() == 0 else label

            result = attack_fn(img, label, **kwargs)
            # Normalize perturbation to unit vector
            direction = result.perturbation.squeeze()
            norm = direction.norm()
            if norm > 0:
                direction = direction / norm
            directions.append(direction)

        return directions

    def get_adversarial_subspace(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        n_samples: int = 100,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the adversarial subspace via SVD of perturbations.

        Returns principal components and explained variance.
        """
        directions = self.generate_adversarial_directions(
            images[:n_samples], labels[:n_samples], **kwargs
        )

        # Stack and flatten perturbations
        perturbation_matrix = torch.stack([d.flatten() for d in directions])
        perturbation_matrix = perturbation_matrix.cpu().numpy()

        # SVD to find principal adversarial directions
        U, S, Vt = np.linalg.svd(perturbation_matrix, full_matrices=False)

        # Explained variance
        explained_variance = (S ** 2) / np.sum(S ** 2)

        return Vt, explained_variance
