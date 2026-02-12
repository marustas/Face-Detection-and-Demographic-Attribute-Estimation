import torch
import numpy as np
from torchvision.datasets import WIDERFace
from torchvision import transforms


class WiderFaceDetectionDataset:
    """
    Wrapper around torchvision WIDERFace dataset
    formatted specifically for face detection experiments.
    """

    def __init__(
        self,
        split: str = "val",
        download: bool = True,
        max_samples: int = None,
    ):
        self.dataset = WIDERFace(
            root="data",
            split=split,
            download=download,
            transform=transforms.ToTensor(),
        )

        self.max_samples = max_samples

    def __len__(self):
        if self.max_samples:
            return min(len(self.dataset), self.max_samples)
        return len(self.dataset)

    def __getitem__(self, idx):

        image_tensor, target = self.dataset[idx]

        image_np = image_tensor.permute(1, 2, 0).numpy()
        image_np = (image_np * 255).clip(0, 255).astype("uint8")

        bboxes = target["bbox"]

        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.numpy()

        gt_boxes = []

        for box in bboxes:
            x, y, w, h = box
            gt_boxes.append([x, y, x + w, y + h])

        gt_boxes = np.array(gt_boxes, dtype=np.float32)

        return {
            "image": image_np,
            "gt_boxes": gt_boxes
        }
