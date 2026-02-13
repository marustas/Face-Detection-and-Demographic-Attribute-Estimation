import torch
import matplotlib.pyplot as plt

from src.datasets.age_gender_dataloader import build_dataloaders

DATASET_PATH = "data/images"


def main():

    print("Building dataloaders...")

    train_loader, val_loader, age_bins = build_dataloaders(
        dataset_path=DATASET_PATH,
        input_size=224,
        batch_size=8,
        val_split=0.2,
        num_workers=0 
    )

    print("\nAge bins:", age_bins)
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))

    batch = next(iter(train_loader))

    images = batch["image"]
    age_classes = batch["age_class"]
    genders = batch["gender"]

    print("\n--- Batch Shape Check ---")
    print("Images shape:", images.shape)
    print("Age shape:", age_classes.shape)
    print("Gender shape:", genders.shape)

    assert images.shape[1:] == (3, 224, 224)
    assert age_classes.dtype == torch.long
    assert genders.dtype == torch.long

    print("Shape test passed.")

    print("\n--- Label Range Check ---")
    print("Age min/max:", age_classes.min().item(), age_classes.max().item())
    print("Gender unique:", torch.unique(genders))

    assert age_classes.min() >= 0
    assert age_classes.max() < len(age_bins)
    assert set(torch.unique(genders).tolist()).issubset({0, 1})

    print("Label test passed.")

    print("\n--- Visual Sanity Check ---")

    # Unnormalize for visualization
    img = images[0].permute(1, 2, 0).cpu().numpy()
    img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    img = img.clip(0, 1)

    plt.imshow(img)
    plt.title(f"Age class: {age_classes[0].item()}, "
              f"Gender: {genders[0].item()}")
    plt.axis("off")
    plt.show()

    print("Visual check complete.")


if __name__ == "__main__":
    main()
