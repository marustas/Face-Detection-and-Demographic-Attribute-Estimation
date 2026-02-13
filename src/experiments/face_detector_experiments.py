import time
import numpy as np
from tqdm import tqdm

from src.datasets.face_dataloader import WiderFaceDetectionDataset
from src.models.face_detection import build_detector
from src.utils.face_detection import evaluate_image


def run_detection_experiment(
    detector_name: str,
    split: str = "val",
    max_samples: int = 500,
    iou_threshold: float = 0.5,
):
    print(f"\nDetector: {detector_name}")

    dataset = WiderFaceDetectionDataset(
        split=split,
        max_samples=max_samples
    )

    detector = build_detector(detector_name)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_time = 0.0

    for i in tqdm(range(len(dataset))):

        sample = dataset[i]
        image = sample["image"]
        gt_boxes = sample["gt_boxes"]

        start_time = time.time()
        detections = detector.detect(image)
        end_time = time.time()

        total_time += (end_time - start_time)

        pred_boxes = [d["bbox"] for d in detections]

        tp, fp, fn = evaluate_image(
            pred_boxes,
            gt_boxes,
            iou_threshold=iou_threshold
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    avg_time = total_time / len(dataset)

    print("\nResults:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Avg time per image: {avg_time:.4f} sec")

    return {
        "precision": precision,
        "recall": recall,
        "avg_time": avg_time,
    }

def main():

    detectors = ["opencv", "mtcnn", "retinaface"]

    for name in detectors:
        run_detection_experiment(
            detector_name=name,
            max_samples=200
        )


if __name__ == "__main__":
    main()