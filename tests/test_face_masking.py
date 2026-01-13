"""
FastFace v6 Face Masking Module Test

This script tests the face parsing and masking functionality without requiring
the full SDXL pipeline to be loaded.

Usage:
    python tests/test_face_masking.py --image path/to/image.jpg

Output:
    - test_output/mask.png: Extracted face+hair mask
    - test_output/masked_blur.png: Image with face blurred
    - test_output/masked_fill.png: Image with face filled (gray)
"""

import os
import sys
import argparse
from PIL import Image
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_face_parser_loading():
    """Test that SegFormer face parser loads correctly."""
    print("\n=== Test 1: Face Parser Loading ===")

    try:
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        print(">>> Loading SegFormer face parser...")

        parser = SegformerForSemanticSegmentation.from_pretrained(
            "jonathandinu/face-parsing",
            cache_dir="models_cache/face_parser"
        )
        processor = SegformerImageProcessor.from_pretrained(
            "jonathandinu/face-parsing",
            cache_dir="models_cache/face_parser"
        )

        print(f">>> Parser loaded: {type(parser).__name__}")
        print(f">>> Processor loaded: {type(processor).__name__}")
        print(">>> PASSED: Face parser loading")
        return parser, processor

    except Exception as e:
        print(f">>> FAILED: {e}")
        return None, None


def test_face_mask_extraction(image_path, parser, processor):
    """Test face mask extraction on a single image."""
    print("\n=== Test 2: Face Mask Extraction ===")

    import torch
    import torch.nn.functional as F
    from PIL import ImageFilter

    FACE_MASK_LABELS = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    try:
        # Load image
        image = Image.open(image_path).convert("RGB")
        print(f">>> Image loaded: {image.size}")

        target_size = (512, 512)  # Use smaller size for testing
        image_resized = image.resize(target_size, Image.LANCZOS)

        # Process through SegFormer
        inputs = processor(images=image_resized, return_tensors="pt")

        # Run inference
        with torch.no_grad():
            outputs = parser(**inputs)
            logits = outputs.logits

        print(f">>> Logits shape: {logits.shape}")

        # Upsample to target size
        upsampled = F.interpolate(
            logits,
            size=(target_size[1], target_size[0]),
            mode="bilinear",
            align_corners=False
        )

        # Get predicted labels
        pred_labels = upsampled.argmax(dim=1).squeeze().cpu().numpy()
        print(f">>> Unique labels found: {np.unique(pred_labels)}")

        # Create binary mask
        mask = np.zeros_like(pred_labels, dtype=np.uint8)
        for label in FACE_MASK_LABELS:
            mask[pred_labels == label] = 255

        coverage = np.sum(mask > 0) / mask.size * 100
        print(f">>> Mask coverage: {coverage:.1f}%")

        if coverage == 0:
            print(">>> WARNING: No face region detected!")
            return None

        # Convert to PIL
        mask_pil = Image.fromarray(mask, mode='L')

        # Apply expansion and blur
        expand_pixels = 10
        mask_pil = mask_pil.filter(ImageFilter.MaxFilter(expand_pixels * 2 + 1))
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=expand_pixels))

        print(">>> PASSED: Face mask extraction")
        return mask_pil, image_resized

    except Exception as e:
        print(f">>> FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_mask_application(image, mask):
    """Test mask application methods."""
    print("\n=== Test 3: Mask Application ===")

    from PIL import ImageFilter

    results = {}

    try:
        # Ensure same size
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.LANCZOS)

        img_array = np.array(image, dtype=np.float32)
        mask_array = np.array(mask, dtype=np.float32) / 255.0
        mask_array = mask_array[:, :, np.newaxis]

        # Test 1: Gaussian Blur
        print(">>> Testing gaussian_blur method...")
        blurred = image.filter(ImageFilter.GaussianBlur(radius=50))
        blurred_array = np.array(blurred, dtype=np.float32)
        result_blur = img_array * (1 - mask_array) + blurred_array * mask_array
        result_blur = np.clip(result_blur, 0, 255).astype(np.uint8)
        results['blur'] = Image.fromarray(result_blur)
        print(">>> gaussian_blur: OK")

        # Test 2: Fill
        print(">>> Testing fill method...")
        fill_color = np.array([128, 128, 128], dtype=np.float32)
        result_fill = img_array * (1 - mask_array) + fill_color * mask_array
        result_fill = np.clip(result_fill, 0, 255).astype(np.uint8)
        results['fill'] = Image.fromarray(result_fill)
        print(">>> fill: OK")

        # Test 3: Noise
        print(">>> Testing noise method...")
        noise = np.random.normal(128, 30, img_array.shape).astype(np.float32)
        noise = np.clip(noise, 0, 255)
        result_noise = img_array * (1 - mask_array) + noise * mask_array
        result_noise = np.clip(result_noise, 0, 255).astype(np.uint8)
        results['noise'] = Image.fromarray(result_noise)
        print(">>> noise: OK")

        print(">>> PASSED: All mask application methods")
        return results

    except Exception as e:
        print(f">>> FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Test FastFace v6 Face Masking")
    parser.add_argument("--image", type=str, default=None, help="Path to test image")
    args = parser.parse_args()

    print("=" * 60)
    print("FastFace v6 Face Masking Module Test")
    print("=" * 60)

    # Test 1: Parser loading
    face_parser, processor = test_face_parser_loading()
    if face_parser is None:
        print("\n>>> Cannot continue without face parser. Exiting.")
        return

    # Find test image
    test_image = args.image
    if test_image is None:
        # Try to find a sample image
        candidates = [
            "venv_mps/lib/python3.11/site-packages/skimage/data/astronaut.png",
            "venv_mps/lib/python3.11/site-packages/insightface/data/images/t1.jpg",
            "data/image.png",
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                test_image = candidate
                break

    if test_image is None or not os.path.exists(test_image):
        print("\n>>> No test image found. Please provide --image argument.")
        print(">>> Example: python tests/test_face_masking.py --image path/to/face.jpg")
        return

    print(f"\n>>> Using test image: {test_image}")

    # Test 2: Mask extraction
    result = test_face_mask_extraction(test_image, face_parser, processor)
    if result[0] is None:
        print("\n>>> Mask extraction failed. Exiting.")
        return

    mask, image = result

    # Test 3: Mask application
    masked_images = test_mask_application(image, mask)
    if masked_images is None:
        print("\n>>> Mask application failed. Exiting.")
        return

    # Save outputs
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n>>> Saving outputs to {output_dir}/")
    mask.save(f"{output_dir}/mask.png")
    image.save(f"{output_dir}/original.png")
    masked_images['blur'].save(f"{output_dir}/masked_blur.png")
    masked_images['fill'].save(f"{output_dir}/masked_fill.png")
    masked_images['noise'].save(f"{output_dir}/masked_noise.png")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {output_dir}/original.png")
    print(f"  - {output_dir}/mask.png")
    print(f"  - {output_dir}/masked_blur.png")
    print(f"  - {output_dir}/masked_fill.png")
    print(f"  - {output_dir}/masked_noise.png")


if __name__ == "__main__":
    main()
