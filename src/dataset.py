import os
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms


def build_age_bins():
    return [
        (0, 12),
        (13, 19),
        (20, 29),
        (30, 39),
        (40, 49),
        (50, 59),
        (60, 120),
    ]


def age_to_class(age: int, age_bins: List[Tuple[int, int]]) -> int:
    for idx, (low, high) in enumerate(age_bins):
        if low <= age <= high:
            return idx
    return len(age_bins) - 1


class UTKFaceDataset(Dataset):
    """
    UTKFace filename format:
    age_gender_race_date.jpg
    Example:
    25_1_2_20170116174525125.jpg
    """

    def __init__(self, root_dir: str, transform=None, age_bins=None):
        self.root_dir = root_dir
        self.transform = transform
        self.age_bins = age_bins if age_bins is not None else build_age_bins()

        self.image_paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".jpg")
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)

        try:
            age, gender, *_ = filename.split("_")
            age = int(age)
            gender = int(gender)
        except Exception:
            # Skip malformed filenames
            return self.__getitem__((idx + 1) % len(self.image_paths))

        age_class = age_to_class(age, self.age_bins)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "age_class": torch.tensor(age_class, dtype=torch.long),
            "gender": torch.tensor(gender, dtype=torch.long),
        }


def build_transforms(input_size: int, train: bool = True):

    if train:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])


def build_dataloaders(
    dataset_path: str,
    input_size: int,
    batch_size: int,
    val_split: float = 0.2,
    num_workers: int = 4,
):

    age_bins = build_age_bins()

    full_dataset = UTKFaceDataset(
        root_dir=dataset_path,
        transform=build_transforms(input_size, train=True),
        age_bins=age_bins,
    )

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Validation should not use augmentation
    val_dataset.dataset.transform = build_transforms(input_size, train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, age_bins
