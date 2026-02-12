import cv2
from .base_detector import BaseFaceDetector

class OpenCVFaceDetector(BaseFaceDetector):

    def __init__(self):
        self.model = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect(self, image):
        if image.dtype != "uint8":
            image = (image * 255).clip(0, 255).astype("uint8")
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        faces = self.model.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        results = []

        for (x, y, w, h) in faces:
            results.append({
                "bbox": [x, y, x + w, y + h],
                "confidence": 1.0  # Haar doesn't provide score
            })

        return results
