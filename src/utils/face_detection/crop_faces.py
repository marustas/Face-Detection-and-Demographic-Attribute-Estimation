import numpy as np


def crop_faces(image, detections, padding=0.2):
    """
    Crop faces from image using bounding boxes.

    Parameters:
        image: numpy array (H, W, 3), uint8 in RGB order
        detections: list of dicts with "bbox" as [x1, y1, x2, y2]
        padding: float, percentage padding around face (e.g., 0.2 adds 20%)

    Returns:
        list of cropped face images (numpy arrays, uint8 RGB)
    """

    if image is None or not hasattr(image, "shape") or len(image.shape) != 3:
        return []

    height, width, _ = image.shape
    face_crops = []

    for det in detections:
        x1, y1, x2, y2 = det.get("bbox", [0, 0, 0, 0])

        # Ensure integer coordinates
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        # Compute padding
        box_w = max(0, x2 - x1)
        box_h = max(0, y2 - y1)

        pad_w = int(box_w * padding)
        pad_h = int(box_h * padding)

        # Apply padding safely
        nx1 = max(0, x1 - pad_w)
        ny1 = max(0, y1 - pad_h)
        nx2 = min(width, x2 + pad_w)
        ny2 = min(height, y2 + pad_h)

        if nx2 <= nx1 or ny2 <= ny1:
            continue

        face = image[ny1:ny2, nx1:nx2]

        if face.size > 0:
            # Ensure uint8 type
            if face.dtype != np.uint8:
                face = (face * 255).clip(0, 255).astype(np.uint8)
            face_crops.append(face)

    return face_crops
