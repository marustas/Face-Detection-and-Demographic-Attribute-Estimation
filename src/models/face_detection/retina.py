from retinaface import RetinaFace
from .base_detector import BaseFaceDetector

class RetinaFaceDetector(BaseFaceDetector):

    def __init__(self, threshold=0.8):
        self.threshold = threshold

    def detect(self, image):
        detections = RetinaFace.detect_faces(image)

        if not detections:
            return []

        results = []

        for key in detections:
            face = detections[key]
            score = face["score"]

            if score < self.threshold:
                continue

            x1, y1, x2, y2 = face["facial_area"]

            results.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(score)
            })

        return results
