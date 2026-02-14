import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from src.model_inference.age_gender_inference import predict
from src.models.face_detection.build_detector import build_detector
from src.utils.face_detection.crop_faces import crop_faces
from src.utils.face_detection.draw_face_box import draw_face_box


def load_image_rgb(input_path):
    img = Image.open(input_path).convert("RGB")
    return np.array(img)


def main():
    parser = argparse.ArgumentParser(
        description="Detect faces and predict age/gender on an image"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a single image OR an image folder (supports .jpg, .jpeg, .png)"
    )
    parser.add_argument(
        "--detector",
        choices=["mtcnn", "retinaface", "opencv"],
        default="mtcnn",
        help="Face detector to use"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show image with drawn bounding boxes"
    )
    parser.add_argument(
        "--save-crops-dir",
        default=None,
        help="Optional directory to save cropped face images"
    )

    args = parser.parse_args()

    input_path = args.input

	# 1. Determine input mode (Single File vs. Folder)
    if os.path.isfile(input_path):
        image_files = [input_path]

    elif os.path.isdir(input_path):
        image_files = [os.path.join(input_path, f) for f in os.listdir(input_path) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    else: 
        print(f"Error: {input_path} is not a valid file or directory.")
        return

    if not image_files: 
        print("No valid images found to process.") 
        return

    # 2. Initialize the Factory Detector once for efficiency
    detector = build_detector(args.detector)
    print(f"--- System Initialized: Processing {len(image_files)} image(s) with {args.detector} ---")

    # 3. Process the Batch
    for img_path in image_files:
        print(f"\nProcessing: {os.path.basename(img_path)}")
        
        # Load and detect
        image_rgb = load_image_rgb(img_path)
        detections = detector.detect(image_rgb)

        if not detections:
            print(f" > No faces detected in {os.path.basename(img_path)}.")
            continue

        # Optionally show bounding boxes (Visual verification)
        if args.show:
            draw_face_box(image_rgb, detections)

        # 4. Crop faces with Strategic Padding (Critical for Age feature extraction)
        crops = crop_faces(image_rgb, detections, padding=0.2)

        # 5. Handle Crop Storage (Audit Trail)
        temp_dir_ctx = None
        save_dir = args.save_crops_dir
        if save_dir is None:
            temp_dir_ctx = tempfile.TemporaryDirectory()
            save_dir = temp_dir_ctx.name
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # 6. Save crops and run Polymorphic Prediction
        for idx, crop in enumerate(crops, start=1):
            # Unique filename for the audit trail
            base_name = Path(img_path).stem
            crop_filename = f"face_{idx}_{base_name}.jpg"
            crop_path = os.path.join(save_dir, crop_filename)
            
            Image.fromarray(crop).save(crop_path)
            
            # This call uses the predict() with confidence scores
            print(f"  [Face #{idx} Analysis]")
            predict(crop_path)

        # Cleanup temp dir for this specific image if persistent saving was disabled
        if temp_dir_ctx is not None:
            temp_dir_ctx.cleanup()


if __name__ == "__main__":
    main()