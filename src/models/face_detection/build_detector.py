from src.models.face_detection.retina import RetinaFaceDetector
from src.models.face_detection.mtcnn import MTCNNFaceDetector
from src.models.face_detection.opencv import OpenCVFaceDetector

def build_detector(name, **kwargs):
    name = name.lower()
    if name == "retinaface":
        return RetinaFaceDetector(**kwargs)
    elif name == "mtcnn":
        return MTCNNFaceDetector(**kwargs)
    elif name == "opencv":
        return OpenCVFaceDetector(**kwargs)
    else:
        raise ValueError(f"Unknown detector: {name}")