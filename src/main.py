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


def load_image_rgb(image_path):
	img = Image.open(image_path).convert("RGB")
	return np.array(img)


def main():
	parser = argparse.ArgumentParser(
		description="Detect faces and predict age/gender on an image"
	)
	parser.add_argument(
		"--image",
		required=True,
		help="Path to the input image"
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

	image_path = args.image
	if not os.path.isfile(image_path):
		raise FileNotFoundError(f"Image not found: {image_path}")

	# Load image and build detector
	image_rgb = load_image_rgb(image_path)
	detector = build_detector(args.detector)

	# Detect faces
	detections = detector.detect(image_rgb)
	if not detections:
		print("No faces detected.")
		return

	print(f"Detected {len(detections)} face(s).")

	# Optionally show bounding boxes
	if args.show:
		draw_face_box(image_rgb, detections)

	# Crop faces
	crops = crop_faces(image_rgb, detections, padding=0.2)

	# Prepare directory for saving crops
	temp_dir_ctx = None
	save_dir = args.save_crops_dir
	if save_dir is None:
		temp_dir_ctx = tempfile.TemporaryDirectory()
		save_dir = temp_dir_ctx.name

	Path(save_dir).mkdir(parents=True, exist_ok=True)

	# Save crops and run predictions
	for idx, crop in enumerate(crops, start=1):
		crop_path = os.path.join(save_dir, f"face_{idx}.jpg")
		Image.fromarray(crop).save(crop_path)
		print(f"\nFace #{idx}:")
		predict(crop_path)

	# Cleanup temp dir if used
	if temp_dir_ctx is not None:
		temp_dir_ctx.cleanup()


if __name__ == "__main__":
	main()