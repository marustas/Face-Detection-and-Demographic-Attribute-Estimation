import cv2
from PIL import Image


def draw_face_box(image, detections, save_path=None):
    """
    Draw bounding boxes around detected faces and optionally save the image with the bounding box.

    Parameters:
        image: numpy array (H, W, 3) uint8 RGB
        detections: list of dicts with key "bbox": [x1, y1, x2, y2]
        save_path: optional filesystem path to save the annotated image

    Returns:
        numpy array of the annotated image (uint8 RGB)
    """

    image_copy = image.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]

        # Ensure integer coordinates
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        # Draw rectangle (green box)
        cv2.rectangle(
            image_copy,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    if save_path:
        Image.fromarray(image_copy).save(save_path)

    return image_copy
