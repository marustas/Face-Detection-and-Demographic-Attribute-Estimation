from src.datasets.face_dataloader import WiderFaceDetectionDataset


def main():

    dataset = WiderFaceDetectionDataset(
        root="data",
        split="val",
        max_samples=5
    )

    print("Dataset size:", len(dataset))

    sample = dataset[0]

    print("Image shape:", sample["image"].shape)
    print("GT boxes shape:", sample["gt_boxes"].shape)


if __name__ == "__main__":
    main()
