"""
FastFace v7.2 Face Masking Module Test

Tests the face parsing and masking functionality using BiSeNet + InsightFace.
v7.2: Switched to BiSeNet for accurate face contour following.

Usage:
    python tests/test_face_masking.py --image path/to/image.jpg

Output:
    - test_output/mask.png: Extracted face+hair mask
    - test_output/masked_blur.png: Image with face blurred
"""

import os
import sys
import argparse
from PIL import Image
import numpy as np
import cv2

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_device():
    """Get best available device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def test_with_bisenet(image_path, output_dir):
    """Test using BiSeNet face parsing (v7.2 method)."""
    print("\n=== Test: BiSeNet Face Parsing (v7.2) ===")

    from src.face_parsing import FaceParser
    from insightface.app import FaceAnalysis

    device = get_device()
    print(f">>> Device: {device}")

    # Initialize InsightFace for face detection
    print(">>> Loading InsightFace...")
    app_face = FaceAnalysis(
        name="buffalo_l",
        root="models_cache",
        providers=["CPUExecutionProvider"]
    )
    app_face.prepare(ctx_id=0, det_size=(640, 640))

    # Initialize FaceParser (BiSeNet)
    parser = FaceParser(device=device)

    # Load image
    img = Image.open(image_path).convert("RGB")
    original_size = img.size
    print(f">>> Original size: {original_size}")

    # Detect face with InsightFace
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    faces = app_face.get(img_cv)

    face_bbox = None
    if len(faces) > 0:
        face = faces[0]
        bbox = face.bbox
        face_bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        print(f">>> Face detected: bbox={face_bbox}")

        # Calculate face dimensions
        face_w = face_bbox[2] - face_bbox[0]
        face_h = face_bbox[3] - face_bbox[1]
        aspect_ratio = face_h / face_w
        print(f">>> Face size: {face_w}x{face_h}, aspect_ratio={aspect_ratio:.2f}")
    else:
        print(">>> WARNING: No face detected by InsightFace")

    # Test at multiple resolutions
    resolutions = [
        original_size,
        (1024, 1024),  # Generation resolution
    ]

    # Test with different bbox_expand_ratio values
    expand_ratios = [1.0, 1.5, 2.0, 2.5]

    for res in resolutions:
        print(f"\n>>> Testing at {res}...")

        for ratio in expand_ratios:
            # Get mask using BiSeNet (falls back to ellipse if BiSeNet fails)
            mask = parser.get_face_hair_mask(
                img,
                target_size=res,
                include_hair=True,
                expand_pixels=10,
                blur_radius=20,
                face_bbox=face_bbox,
                bbox_expand_ratio=ratio
            )

            if mask is None:
                print(f">>> ERROR: Mask extraction failed at {res} with ratio={ratio}")
                continue

            mask_array = np.array(mask)
            coverage = np.sum(mask_array > 127) / mask_array.size * 100
            print(f">>> [ratio={ratio}] Mask coverage: {coverage:.1f}%")

            # Check connected components
            try:
                from scipy import ndimage
                binary_mask = mask_array > 127
                labeled, num_features = ndimage.label(binary_mask)
                print(f">>> [ratio={ratio}] Connected components: {num_features}")
            except ImportError:
                pass

            mask_path = f"{output_dir}/mask_{res[0]}x{res[1]}_ratio{ratio}.png"
            mask.save(mask_path)
            print(f">>> Saved mask: {mask_path}")

            # Apply masking
            img_resized = img.resize(res, Image.LANCZOS)
            for method in ['gaussian_blur', 'fill']:
                masked = parser.apply_mask(
                    img_resized,
                    mask,
                    method=method,
                    blur_radius=50
                )
                masked_path = f"{output_dir}/masked_{method}_{res[0]}x{res[1]}_ratio{ratio}.png"
                masked.save(masked_path)
                print(f">>> Saved masked ({method}): {masked_path}")

    parser.unload()
    print("\n>>> Test completed!")


def main():
    arg_parser = argparse.ArgumentParser(description="Test FastFace v7.2 Face Masking")
    arg_parser.add_argument("--image", type=str, default=None, help="Path to test image")
    args = arg_parser.parse_args()

    print("=" * 60)
    print("FastFace v7.2 Face Masking Test")
    print("(BiSeNet + InsightFace Fallback)")
    print("=" * 60)

    # Find test image
    test_image = args.image
    if test_image is None:
        candidates = [
            "data/reference-image.png",
            "data/face-image.png",
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

    # Create output directory
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)

    # Run test
    test_with_bisenet(test_image, output_dir)

    print("\n" + "=" * 60)
    print("TEST COMPLETED!")
    print("=" * 60)
    print(f"\nCheck output files in: {output_dir}/")


if __name__ == "__main__":
    main()
