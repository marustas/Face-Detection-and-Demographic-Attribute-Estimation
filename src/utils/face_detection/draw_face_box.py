import matplotlib.pyplot as plt
import cv2


def draw_face_box(image, detections, figsize=(8, 6)):
    """
    Draw bounding boxes around detected faces.

    Parameters:
        image: numpy array (H, W, 3) uint8 RGB
        detections: list of dicts with key "bbox": [x1, y1, x2, y2]
        figsize: tuple for matplotlib figure size
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

    plt.figure(figsize=figsize)
    plt.imshow(image_copy)
    plt.axis("off")
    plt.show()
