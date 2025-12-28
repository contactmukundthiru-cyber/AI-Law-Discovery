"""
Environmental corruptions simulating evolutionary visual challenges.

These represent the natural pressures that shaped biological vision.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import cv2
from scipy import ndimage


class CorruptionType(Enum):
    """Types of environmental corruptions."""
    # Lighting variations
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    SHADOWS = "shadows"

    # Weather conditions
    FOG = "fog"
    RAIN = "rain"
    SNOW = "snow"

    # Motion effects
    BLUR = "blur"
    DEFOCUS = "defocus"
    ZOOM = "zoom"

    # Occlusion
    PARTIAL = "partial"
    CAMOUFLAGE = "camouflage"
    CLUTTER = "clutter"

    # Depth/perspective
    PERSPECTIVE = "perspective"
    SCALE = "scale"
    DISTANCE = "distance"


@dataclass
class CorruptionResult:
    """Result of applying corruption."""
    corrupted_image: np.ndarray
    corruption_type: CorruptionType
    severity: float
    corruption_mask: Optional[np.ndarray] = None


class EnvironmentalCorruptions:
    """
    Generates environmental corruptions that simulate evolutionary pressures.

    These corruptions represent natural challenges that biological vision
    systems evolved to handle.
    """

    def __init__(self, severity_range: Tuple[float, float] = (0.1, 1.0)):
        self.severity_range = severity_range

        self.corruption_functions = {
            CorruptionType.BRIGHTNESS: self._apply_brightness,
            CorruptionType.CONTRAST: self._apply_contrast,
            CorruptionType.SHADOWS: self._apply_shadows,
            CorruptionType.FOG: self._apply_fog,
            CorruptionType.RAIN: self._apply_rain,
            CorruptionType.SNOW: self._apply_snow,
            CorruptionType.BLUR: self._apply_blur,
            CorruptionType.DEFOCUS: self._apply_defocus,
            CorruptionType.ZOOM: self._apply_zoom,
            CorruptionType.PARTIAL: self._apply_partial_occlusion,
            CorruptionType.CAMOUFLAGE: self._apply_camouflage,
            CorruptionType.CLUTTER: self._apply_clutter,
            CorruptionType.PERSPECTIVE: self._apply_perspective,
            CorruptionType.SCALE: self._apply_scale,
            CorruptionType.DISTANCE: self._apply_distance,
        }

    def apply(
        self,
        image: np.ndarray,
        corruption_type: CorruptionType,
        severity: float = 0.5
    ) -> CorruptionResult:
        """Apply a specific corruption to an image."""
        if corruption_type not in self.corruption_functions:
            raise ValueError(f"Unknown corruption: {corruption_type}")

        func = self.corruption_functions[corruption_type]
        corrupted, mask = func(image, severity)

        return CorruptionResult(
            corrupted_image=corrupted,
            corruption_type=corruption_type,
            severity=severity,
            corruption_mask=mask
        )

    def apply_all(
        self,
        image: np.ndarray,
        severity: float = 0.5
    ) -> Dict[CorruptionType, CorruptionResult]:
        """Apply all corruptions to an image."""
        results = {}
        for ctype in CorruptionType:
            results[ctype] = self.apply(image, ctype, severity)
        return results

    def get_corruption_vector(
        self,
        original: np.ndarray,
        corrupted: np.ndarray
    ) -> np.ndarray:
        """Get the corruption direction as a vector."""
        return (corrupted.astype(float) - original.astype(float)) / 255.0

    # Lighting corruptions
    def _apply_brightness(
        self,
        image: np.ndarray,
        severity: float
    ) -> Tuple[np.ndarray, None]:
        """Adjust image brightness."""
        factor = 1 + (severity * 2 - 1) * 0.5  # 0.5 to 1.5
        adjusted = np.clip(image * factor, 0, 255).astype(np.uint8)
        return adjusted, None

    def _apply_contrast(
        self,
        image: np.ndarray,
        severity: float
    ) -> Tuple[np.ndarray, None]:
        """Adjust image contrast."""
        factor = 0.5 + severity  # 0.5 to 1.5
        mean = np.mean(image)
        adjusted = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
        return adjusted, None

    def _apply_shadows(
        self,
        image: np.ndarray,
        severity: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add shadow patterns."""
        h, w = image.shape[:2]

        # Create shadow mask
        shadow_mask = np.ones((h, w), dtype=np.float32)

        # Random shadow region
        x1, y1 = np.random.randint(0, w//2), np.random.randint(0, h//2)
        x2, y2 = np.random.randint(w//2, w), np.random.randint(h//2, h)

        shadow_mask[y1:y2, x1:x2] = 1 - severity * 0.7

        # Blur shadow edges
        shadow_mask = ndimage.gaussian_filter(shadow_mask, sigma=20)

        if len(image.shape) == 3:
            shadow_mask = shadow_mask[:, :, np.newaxis]

        shadowed = (image * shadow_mask).astype(np.uint8)
        return shadowed, shadow_mask.squeeze()

    # Weather corruptions
    def _apply_fog(
        self,
        image: np.ndarray,
        severity: float
    ) -> Tuple[np.ndarray, None]:
        """Apply fog effect."""
        h, w = image.shape[:2]

        # Create fog layer
        fog = np.ones_like(image, dtype=np.float32) * 200

        # Depth-dependent fog (simulate atmosphere)
        depth_mask = np.linspace(0, 1, h)[:, np.newaxis]
        if len(image.shape) == 3:
            depth_mask = np.stack([depth_mask] * 3, axis=2)

        fog_strength = severity * 0.8 * depth_mask

        fogged = (image * (1 - fog_strength) + fog * fog_strength).astype(np.uint8)
        return fogged, None

    def _apply_rain(
        self,
        image: np.ndarray,
        severity: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add rain streaks."""
        h, w = image.shape[:2]

        # Create rain streaks
        rain_mask = np.zeros((h, w), dtype=np.float32)
        n_drops = int(severity * 1000)

        for _ in range(n_drops):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            length = np.random.randint(10, 30)

            y_end = min(y + length, h)
            if y < h and x < w:
                rain_mask[y:y_end, x] = 1

        # Apply motion blur to streaks
        kernel_size = 15
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[:, kernel_size//2] = 1
        kernel = kernel / kernel_size
        rain_mask = cv2.filter2D(rain_mask, -1, kernel)

        # Blend rain with image
        rain_color = 200
        if len(image.shape) == 3:
            rain_layer = np.stack([rain_mask * rain_color] * 3, axis=2)
        else:
            rain_layer = rain_mask * rain_color

        rained = np.clip(image + rain_layer * severity, 0, 255).astype(np.uint8)
        return rained, rain_mask

    def _apply_snow(
        self,
        image: np.ndarray,
        severity: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add snow effect."""
        h, w = image.shape[:2]

        # Create snow particles
        snow_mask = np.zeros((h, w), dtype=np.float32)
        n_flakes = int(severity * 500)

        for _ in range(n_flakes):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            size = np.random.randint(1, 4)

            cv2.circle(snow_mask, (x, y), size, 1, -1)

        # Blur for soft edges
        snow_mask = ndimage.gaussian_filter(snow_mask, sigma=1)

        # Add overall brightness (snow reflection)
        brightness_boost = 1 + severity * 0.3

        snowed = np.clip(image * brightness_boost, 0, 255)
        if len(image.shape) == 3:
            snow_layer = np.stack([snow_mask * 255] * 3, axis=2)
        else:
            snow_layer = snow_mask * 255

        snowed = np.clip(snowed + snow_layer * severity, 0, 255).astype(np.uint8)
        return snowed, snow_mask

    # Motion corruptions
    def _apply_blur(
        self,
        image: np.ndarray,
        severity: float
    ) -> Tuple[np.ndarray, None]:
        """Apply motion blur."""
        kernel_size = int(severity * 20) * 2 + 1

        # Random direction motion blur
        angle = np.random.uniform(0, 180)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size//2, :] = 1
        kernel = kernel / kernel_size

        # Rotate kernel
        center = (kernel_size // 2, kernel_size // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))

        blurred = cv2.filter2D(image, -1, kernel)
        return blurred, None

    def _apply_defocus(
        self,
        image: np.ndarray,
        severity: float
    ) -> Tuple[np.ndarray, None]:
        """Apply defocus blur (circular bokeh)."""
        kernel_size = int(severity * 15) * 2 + 1

        # Circular kernel for defocus
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        cv2.circle(kernel, (kernel_size//2, kernel_size//2), kernel_size//2, 1, -1)
        kernel = kernel / kernel.sum()

        defocused = cv2.filter2D(image, -1, kernel)
        return defocused, None

    def _apply_zoom(
        self,
        image: np.ndarray,
        severity: float
    ) -> Tuple[np.ndarray, None]:
        """Apply zoom blur."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        zoomed = image.copy().astype(np.float32)
        n_levels = int(severity * 10) + 1

        for i in range(1, n_levels + 1):
            scale = 1 + i * 0.02 * severity
            M = cv2.getRotationMatrix2D(center, 0, scale)
            scaled = cv2.warpAffine(image, M, (w, h))
            zoomed = (zoomed + scaled) / 2

        return zoomed.astype(np.uint8), None

    # Occlusion corruptions
    def _apply_partial_occlusion(
        self,
        image: np.ndarray,
        severity: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply partial occlusion."""
        h, w = image.shape[:2]
        mask = np.ones((h, w), dtype=np.float32)

        # Random rectangular occlusion
        occ_h = int(h * severity * 0.5)
        occ_w = int(w * severity * 0.5)

        y = np.random.randint(0, h - occ_h)
        x = np.random.randint(0, w - occ_w)

        mask[y:y+occ_h, x:x+occ_w] = 0

        if len(image.shape) == 3:
            mask = mask[:, :, np.newaxis]

        occluded = (image * mask).astype(np.uint8)
        return occluded, mask.squeeze()

    def _apply_camouflage(
        self,
        image: np.ndarray,
        severity: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply camouflage pattern matching background."""
        h, w = image.shape[:2]

        # Create camouflage by blending with scrambled version
        # Simulate how objects blend with environment
        patches = []
        patch_size = 32

        for y in range(0, h, patch_size):
            for x in range(0, w, patch_size):
                patch = image[y:y+patch_size, x:x+patch_size]
                if np.random.random() < severity * 0.3:
                    # Shuffle patch with nearby region
                    dy = np.random.randint(-patch_size, patch_size)
                    dx = np.random.randint(-patch_size, patch_size)
                    ny, nx = max(0, y+dy), max(0, x+dx)
                    ny, nx = min(h-patch_size, ny), min(w-patch_size, nx)
                    patch = image[ny:ny+patch_size, nx:nx+patch_size]
                patches.append((y, x, patch))

        camouflaged = image.copy()
        for y, x, patch in patches:
            ph, pw = patch.shape[:2]
            camouflaged[y:y+ph, x:x+pw] = patch

        mask = (camouflaged != image).any(axis=2) if len(image.shape) == 3 else (camouflaged != image)
        return camouflaged, mask.astype(np.float32)

    def _apply_clutter(
        self,
        image: np.ndarray,
        severity: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add visual clutter/noise."""
        h, w = image.shape[:2]

        # Add random shapes as clutter
        cluttered = image.copy()
        mask = np.zeros((h, w), dtype=np.float32)
        n_objects = int(severity * 20)

        for _ in range(n_objects):
            color = tuple(np.random.randint(0, 256, 3).tolist())
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            size = np.random.randint(5, 30)

            if np.random.random() < 0.5:
                cv2.circle(cluttered, (x, y), size, color, -1)
                cv2.circle(mask, (x, y), size, 1, -1)
            else:
                cv2.rectangle(cluttered, (x, y), (x+size, y+size), color, -1)
                mask[y:y+size, x:x+size] = 1

        return cluttered, mask

    # Depth/perspective corruptions
    def _apply_perspective(
        self,
        image: np.ndarray,
        severity: float
    ) -> Tuple[np.ndarray, None]:
        """Apply perspective distortion."""
        h, w = image.shape[:2]

        # Source points
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

        # Distorted destination points
        offset = int(severity * w * 0.2)
        dst = np.float32([
            [offset, 0],
            [w - offset, 0],
            [w, h],
            [0, h]
        ])

        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(image, M, (w, h))

        return warped, None

    def _apply_scale(
        self,
        image: np.ndarray,
        severity: float
    ) -> Tuple[np.ndarray, None]:
        """Apply scale transformation."""
        h, w = image.shape[:2]

        scale = 1 - severity * 0.5  # Scale down
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize and pad
        scaled = cv2.resize(image, (new_w, new_h))

        # Center in original size
        result = np.zeros_like(image)
        y_offset = (h - new_h) // 2
        x_offset = (w - new_w) // 2
        result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = scaled

        return result, None

    def _apply_distance(
        self,
        image: np.ndarray,
        severity: float
    ) -> Tuple[np.ndarray, None]:
        """Simulate distance effect (blur + scale)."""
        # Combine scale and blur for distance effect
        scaled, _ = self._apply_scale(image, severity * 0.5)
        blurred, _ = self._apply_defocus(scaled, severity * 0.5)
        return blurred, None
