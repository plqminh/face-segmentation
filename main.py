"""Face Segmentation Inference Script

Run XSeg face segmentation on images with automatic face detection.

Usage:
    python main.py --model weights/xseg.onnx --image assets/einstien.png
    python main.py --model weights/xseg.onnx --image assets/einstien.png --blur 5
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
from uniface import RetinaFace, face_alignment

from models import XSeg


def apply_mask_visualization(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Apply colored mask overlay for visualization."""
    overlay = image.copy().astype(np.float32)

    # Create green-tinted overlay where mask is positive
    mask_3ch = np.stack([mask * 0.3, mask * 0.7, mask * 0.3], axis=-1)
    overlay = overlay * (1 - mask[..., None] * alpha) + mask_3ch * 255 * alpha

    return overlay.clip(0, 255).astype(np.uint8)


def run_inference(
    model_path: str,
    image_path: str,
    output_path: str | None = None,
    blur_sigma: float = 0,
    align_size: int = 256,
) -> None:
    """Run XSeg inference with face detection and landmark alignment."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    print(f"Loaded image: {image_path}")
    print(f"Image shape: {image.shape}")

    # Initialize face detector
    print("\nInitializing face detector...")
    detector = RetinaFace()

    # Detect faces
    faces = detector.detect(image)
    print(f"Detected {len(faces)} face(s)")

    if len(faces) == 0:
        print("No faces detected. Exiting.")
        return

    # Initialize XSeg parser
    print("\nInitializing XSeg parser...")
    parser = XSeg(model_path, align_size=align_size, blur_sigma=blur_sigma)

    # Print model info
    print(f"  Input size: {parser.input_size}")
    print(f"  Align size: {parser.align_size}")
    print(f"  Blur sigma: {parser.blur_sigma}")

    # Process each face
    result_image = image.copy()
    full_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

    for i, face in enumerate(faces):
        print(f"\nProcessing face {i + 1}...")
        print(f"  Confidence: {face.confidence:.3f}")

        landmarks = face.landmarks
        if landmarks is None or landmarks.shape[0] != 5:
            print("  Skipping: No valid 5-point landmarks")
            continue

        # Parse face
        mask = parser.parse(image, landmarks)
        full_mask = np.maximum(full_mask, mask)
        print(f"  Mask range: [{mask.min():.4f}, {mask.max():.4f}]")

        # Draw bbox on result
        x1, y1, x2, y2 = map(int, face.bbox[:4])
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw landmarks
        for lm in landmarks:
            cv2.circle(result_image, (int(lm[0]), int(lm[1])), 3, (0, 0, 255), -1)

    # Apply visualization
    result_with_mask = apply_mask_visualization(image, full_mask)

    # Add bbox to visualization
    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox[:4])
        cv2.rectangle(result_with_mask, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save results
    if output_path is None:
        base_name = image_path.rsplit(".", 1)[0]
        output_path = f"{base_name}_xseg_result.png"

    cv2.imwrite(output_path, result_with_mask)
    print(f"\nResult saved: {output_path}")

    # Save mask
    mask_path = image_path.rsplit(".", 1)[0] + "_xseg_mask.png"
    mask_uint8 = (full_mask * 255).astype(np.uint8)
    cv2.imwrite(mask_path, mask_uint8)
    print(f"Mask saved: {mask_path}")

    # Save aligned face crop (for debugging)
    if len(faces) > 0 and faces[0].landmarks is not None:
        crop_path = image_path.rsplit(".", 1)[0] + "_aligned_crop.png"
        aligned_crop, _ = face_alignment(image, faces[0].landmarks, image_size=align_size)
        cv2.imwrite(crop_path, aligned_crop)
        print(f"Aligned crop saved: {crop_path}")

    # Create comparison
    comparison = np.hstack([image, result_with_mask])
    comparison_path = image_path.rsplit(".", 1)[0] + "_xseg_comparison.png"
    cv2.imwrite(comparison_path, comparison)
    print(f"Comparison saved: {comparison_path}")

    # Display if possible
    try:
        cv2.imshow("Original", image)
        cv2.imshow("Mask", mask_uint8)
        cv2.imshow("Result", result_with_mask)
        print("\nPress any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        print("(Display not available, results saved to files)")


def main() -> None:
    parser = argparse.ArgumentParser(description="XSeg ONNX Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to XSeg ONNX model")
    parser.add_argument("--image", type=str, default="assets/einstien.png", help="Path to input image")
    parser.add_argument("--output", type=str, default=None, help="Path to output image")
    parser.add_argument(
        "--blur",
        type=float,
        default=0,
        help="Gaussian blur sigma for mask smoothing. 0 = raw output (default)",
    )
    parser.add_argument(
        "--align-size",
        type=int,
        default=256,
        help="Face alignment size (must be multiple of 112 or 128). Default: 256",
    )
    args = parser.parse_args()

    run_inference(args.model, args.image, args.output, blur_sigma=args.blur, align_size=args.align_size)


if __name__ == "__main__":
    main()
