from abc import ABC, abstractmethod

class BaseFaceDetector(ABC):

    @abstractmethod
    def detect(self, image):
        """
        Args:
            image: numpy RGB image (H, W, 3)

        Returns:
            List[dict] with:
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float
                }
        """
        pass
