import torch
from facenet_pytorch import MTCNN
from .base_detector import BaseFaceDetector


class MTCNNFaceDetector(BaseFaceDetector):

    def __init__(self, device=None):
        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = MTCNN(keep_all=True, device=self.device)

    def detect(self, image):
        boxes, probs = self.model.detect(image)

        if boxes is None:
            return []

        results = []
        for box, prob in zip(boxes, probs):
            x1, y1, x2, y2 = box.astype(int)
            results.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(prob)
            })

        return results
