import matplotlib.pyplot as plt
import cv2


def draw_face_box(image, detections, figsize=(8, 6), title=None):
    """
    Draw bounding boxes around detected faces.

    Parameters:
        image: numpy array (H, W, 3) uint8 RGB
        detections: list of dicts with keys:
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": float
            }
        figsize: tuple
        title: optional plot title
    """

    image_copy = image.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        confidence = det.get("confidence", None)

        # Draw rectangle
        cv2.rectangle(
            image_copy,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Draw confidence score
        if confidence is not None:
            text = f"{confidence:.2f}"
            cv2.putText(
                image_copy,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

    plt.figure(figsize=figsize)
    plt.imshow(image_copy)
    plt.axis("off")

    if title:
        plt.title(title)

    plt.show()
