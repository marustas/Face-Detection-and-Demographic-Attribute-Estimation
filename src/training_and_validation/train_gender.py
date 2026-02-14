import torch
import torch.nn as nn
from tqdm import tqdm

from src.training_and_validation.gender_model import GenderCNN
from src.datasets import build_dataloaders


# --- Configuration ---
DATASET_PATH = "data/images"
INPUT_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print(f"--- Training Gender Model on {DEVICE} ---")

    # Load Data (reuse existing dataloader)
    train_loader, val_loader, _ = build_dataloaders(
        dataset_path=DATASET_PATH,
        input_size=INPUT_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=2
    )

    # Initialize Model
    model = GenderCNN(backbone_name='mobilenet_v2').to(DEVICE)

    # Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):

        # ----- TRAINING -----
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

        for batch in train_pbar:
            images = batch["image"].to(DEVICE)
            targets = batch["gender"].to(DEVICE).float().clamp(0, 1)

            logits = model(images)
            loss = criterion(logits.squeeze(), targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # ----- VALIDATION -----
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(DEVICE)
                targets = batch["gender"].to(DEVICE).float().clamp(0, 1)

                logits = model(images)
                preds = (torch.sigmoid(logits.squeeze()) > 0.5).float()

                correct += (preds == targets).sum().item()
                total += targets.size(0)

        val_acc = (correct / total) * 100
        print(f"\nEpoch {epoch+1} Validation Gender Accuracy: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"model_state_dict": model.state_dict(),
                 "best_acc": best_val_acc},
                "best_gender_model.pth"
            )
            print(f"*** NEW BEST MODEL SAVED ({val_acc:.2f}%) ***\n")

    print(f"\nTraining Complete. Best Gender Accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
