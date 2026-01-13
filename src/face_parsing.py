"""
FastFace v6 Face Parsing Module

This module provides face+hair segmentation functionality using SegFormer.
It is designed to be modular and reusable across different parts of the pipeline.

Usage:
    from src.face_parsing import FaceParser

    parser = FaceParser(device="mps")
    mask = parser.get_face_hair_mask(image)
    masked_image = parser.apply_mask(image, mask, method='gaussian_blur')

Classes:
    FaceParser: Main class for face parsing and masking operations
"""

import torch
import numpy as np
from PIL import Image, ImageFilter
from typing import Optional, Tuple, Union, Literal


# SegFormer face parsing label mapping (jonathandinu/face-parsing model)
# Based on CelebAMask-HQ dataset
FACE_PARSE_LABELS = {
    0: 'background',
    1: 'skin',
    2: 'nose',
    3: 'eyeglasses',
    4: 'left_eye',
    5: 'right_eye',
    6: 'left_eyebrow',
    7: 'right_eyebrow',
    8: 'left_ear',
    9: 'right_ear',
    10: 'mouth',
    11: 'upper_lip',
    12: 'lower_lip',
    13: 'hair',
    14: 'hat',
    15: 'earring',
    16: 'necklace',
    17: 'neck',
    18: 'clothing'
}

# Default labels to include in face+hair mask
DEFAULT_FACE_LABELS = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # Face features (no hair)
DEFAULT_HAIR_LABELS = [13]  # Hair only
DEFAULT_FACE_HAIR_LABELS = DEFAULT_FACE_LABELS + DEFAULT_HAIR_LABELS


class FaceParser:
    """
    Face parsing and masking module using SegFormer.

    This class handles:
    - Loading and managing the SegFormer model
    - Extracting face+hair masks from images
    - Applying various mask methods (blur, fill, noise)

    Attributes:
        device: torch device for inference
        model: SegFormer model (lazy loaded)
        processor: SegFormer image processor (lazy loaded)
    """

    def __init__(
        self,
        device: Union[str, torch.device] = "cpu",
        model_name: str = "jonathandinu/face-parsing",
        cache_dir: str = "models_cache/face_parser"
    ):
        """
        Initialize FaceParser.

        Args:
            device: Device to run inference on ('cpu', 'cuda', 'mps')
            model_name: HuggingFace model name for face parsing
            cache_dir: Directory to cache downloaded models
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model_name = model_name
        self.cache_dir = cache_dir

        # Lazy-loaded components
        self._model = None
        self._processor = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    def load(self) -> bool:
        """
        Load the SegFormer model.

        Returns:
            True if loading successful, False otherwise
        """
        if self._is_loaded:
            return True

        try:
            from transformers import (
                SegformerForSemanticSegmentation,
                SegformerImageProcessor
            )

            print(f">>> [FaceParser] Loading SegFormer from {self.model_name}...")

            self._model = SegformerForSemanticSegmentation.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            ).to(self.device)

            self._processor = SegformerImageProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )

            self._model.eval()
            self._is_loaded = True

            print(f">>> [FaceParser] Model loaded successfully on {self.device}")
            return True

        except Exception as e:
            print(f">>> [FaceParser] Failed to load model: {e}")
            return False

    def unload(self):
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._is_loaded = False

        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

        print(">>> [FaceParser] Model unloaded")

    def get_segmentation(
        self,
        image: Union[str, Image.Image],
        target_size: Optional[Tuple[int, int]] = None
    ) -> Optional[np.ndarray]:
        """
        Get full segmentation map from image.

        Args:
            image: PIL Image or path to image
            target_size: Output size (width, height). If None, uses image size.

        Returns:
            Segmentation map as numpy array, or None if failed
        """
        if not self.load():
            return None

        # Load image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        if target_size is None:
            target_size = image.size

        # Resize for processing
        image_resized = image.resize(target_size, Image.LANCZOS)

        # Process through SegFormer
        inputs = self._processor(images=image_resized, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits

        # Upsample to target size
        import torch.nn.functional as F
        upsampled = F.interpolate(
            logits,
            size=(target_size[1], target_size[0]),  # (H, W)
            mode="bilinear",
            align_corners=False
        )

        # Get predicted labels
        pred_labels = upsampled.argmax(dim=1).squeeze().cpu().numpy()

        return pred_labels

    def get_face_hair_mask(
        self,
        image: Union[str, Image.Image],
        target_size: Optional[Tuple[int, int]] = None,
        include_hair: bool = True,
        expand_pixels: int = 10,
        blur_radius: int = 10
    ) -> Optional[Image.Image]:
        """
        Extract face+hair mask from image.

        Args:
            image: PIL Image or path to image
            target_size: Output size (width, height). If None, uses image size.
            include_hair: Whether to include hair in mask
            expand_pixels: Pixels to expand mask boundary
            blur_radius: Gaussian blur radius for soft edges

        Returns:
            Grayscale mask (white=face region, black=background), or None
        """
        # Load image for size info
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        if target_size is None:
            target_size = image.size

        # Get segmentation
        pred_labels = self.get_segmentation(image, target_size)
        if pred_labels is None:
            return None

        # Select labels for mask
        mask_labels = list(DEFAULT_FACE_LABELS)
        if include_hair:
            mask_labels.extend(DEFAULT_HAIR_LABELS)

        # Create binary mask
        mask = np.zeros_like(pred_labels, dtype=np.uint8)
        for label in mask_labels:
            mask[pred_labels == label] = 255

        # Check if any face region was found
        if np.sum(mask) == 0:
            print(">>> [FaceParser] No face region found in segmentation")
            return None

        coverage = np.sum(mask > 0) / mask.size * 100
        print(f">>> [FaceParser] Mask extracted, coverage: {coverage:.1f}%")

        # Convert to PIL
        mask_pil = Image.fromarray(mask, mode='L')

        # Expand mask slightly for smoother blending
        if expand_pixels > 0:
            mask_pil = mask_pil.filter(
                ImageFilter.MaxFilter(expand_pixels * 2 + 1)
            )

        # Apply gaussian blur for soft edges
        if blur_radius > 0:
            mask_pil = mask_pil.filter(
                ImageFilter.GaussianBlur(radius=blur_radius)
            )

        return mask_pil

    def apply_mask(
        self,
        image: Union[str, Image.Image],
        mask: Image.Image,
        method: Literal['gaussian_blur', 'fill', 'noise'] = 'gaussian_blur',
        blur_radius: int = 50,
        fill_color: Tuple[int, int, int] = (128, 128, 128)
    ) -> Image.Image:
        """
        Apply mask to neutralize face region in image.

        Args:
            image: PIL Image or path to image
            mask: Grayscale mask (white=region to mask)
            method: Masking method ('gaussian_blur', 'fill', 'noise')
            blur_radius: Radius for gaussian blur method
            fill_color: RGB color for fill method

        Returns:
            Image with face region processed
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # Ensure mask is same size as image
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.LANCZOS)

        # Convert to numpy
        img_array = np.array(image, dtype=np.float32)
        mask_array = np.array(mask, dtype=np.float32) / 255.0
        mask_array = mask_array[:, :, np.newaxis]  # Add channel dim

        if method == 'gaussian_blur':
            blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            blurred_array = np.array(blurred, dtype=np.float32)
            result_array = img_array * (1 - mask_array) + blurred_array * mask_array

        elif method == 'fill':
            fill_array = np.array(fill_color, dtype=np.float32)
            result_array = img_array * (1 - mask_array) + fill_array * mask_array

        elif method == 'noise':
            noise = np.random.normal(128, 30, img_array.shape).astype(np.float32)
            noise = np.clip(noise, 0, 255)
            result_array = img_array * (1 - mask_array) + noise * mask_array

        else:
            raise ValueError(f"Unknown method: {method}")

        result_array = np.clip(result_array, 0, 255).astype(np.uint8)
        print(f">>> [FaceParser] Mask applied with method='{method}'")

        return Image.fromarray(result_array)

    def apply_mask_to_depth(
        self,
        depth_image: Image.Image,
        face_mask: Image.Image,
        fill_value: float = 0.5
    ) -> Image.Image:
        """
        Apply face mask to depth map, neutralizing face depth.

        Args:
            depth_image: PIL Image (depth map)
            face_mask: Grayscale mask
            fill_value: Value to fill masked region (0-1, 0.5=middle depth)

        Returns:
            Depth image with face region neutralized
        """
        # Ensure same size
        if face_mask.size != depth_image.size:
            face_mask = face_mask.resize(depth_image.size, Image.LANCZOS)

        # Convert to numpy
        depth_array = np.array(depth_image.convert('L'), dtype=np.float32) / 255.0
        mask_array = np.array(face_mask, dtype=np.float32) / 255.0

        # Fill face region with neutral depth
        result = depth_array * (1 - mask_array) + fill_value * mask_array
        result = (result * 255).astype(np.uint8)

        print(f">>> [FaceParser] Depth mask applied (fill_value={fill_value})")

        # Convert back to RGB depth image
        return Image.fromarray(result).convert('RGB')


# Convenience function for quick use
def create_face_parser(device: str = "cpu") -> FaceParser:
    """
    Create and load a FaceParser instance.

    Args:
        device: Device to use ('cpu', 'cuda', 'mps')

    Returns:
        Loaded FaceParser instance
    """
    parser = FaceParser(device=device)
    parser.load()
    return parser
