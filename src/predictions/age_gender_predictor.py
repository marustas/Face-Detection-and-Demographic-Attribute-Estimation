import sys
import os
import torch
import torch.nn as nn
from tqdm import tqdm 
from src.predictions import MultitaskCNN
from src.datasets import build_dataloaders

# --- 1. Configuration Constants ---
DATASET_PATH = "data/images"
INPUT_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print(f"--- Initialization: Targeting {DEVICE} ---")

    # 2. Setup Data
    # The build_dataloaders function should be handling the filename parsing.
    # We ignore the 'race' and 'timestamp' parts entirely.
    train_loader, val_loader, age_bins = build_dataloaders(
        dataset_path=DATASET_PATH,
        input_size=INPUT_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=2
    )

    # 3. Dynamic Sanity Check: Audit Dataset for Demographic Alignment
    print("Auditing dataset for Age & Gender labels...")
    max_age_index = 0
    
    # Pass through the training data once to ensure model heads are sized correctly
    for batch in train_loader:
        batch_max_age = batch["age_class"].max().item()
        if batch_max_age > max_age_index:
            max_age_index = batch_max_age
    
    num_age_classes = max_age_index + 1
    print(f"--> Discovery Complete: {num_age_classes} Age Categories found.")

    # 4. Initialize Model & Optimization
    # We use our custom MultitaskCNN which only has 2 heads (Age, Gender)
    model = MultitaskCNN(
        backbone_name='mobilenet_v2', 
        num_age_classes=num_age_classes
    ).to(DEVICE)

    criterion_gender = nn.BCELoss() # Binary Cross Entropy for Gender (0 or 1)
    criterion_age = nn.CrossEntropyLoss() # Cross Entropy for Binned Age Classes
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    best_val_acc = 0.0

    # 5. Training & Validation Loop
    for epoch in range(EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for batch in train_pbar:
            images = batch["image"].to(DEVICE)
            target_ages = batch["age_class"].to(DEVICE)
            
            # GENDER SAFETY: Ensure labels are strictly 0 or 1 (ignoring race data)
            target_genders = batch["gender"].to(DEVICE).float().clamp(0, 1)

            # Forward Pass
            pred_gender, pred_age = model(images)
            
            # Loss Calculation (Combined)
            loss_gender = criterion_gender(pred_gender.squeeze(), target_genders)
            loss_age = criterion_age(pred_age, target_ages.long())
            batch_loss = loss_gender + loss_age
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            train_loss += batch_loss.item()
            train_pbar.set_postfix({'loss': f"{batch_loss.item():.4f}"})

        # --- VALIDATION PHASE ---
        model.eval()
        val_gender_correct = 0
        val_age_correct = 0
        total_samples = 0

        print(f"Validating Epoch {epoch+1}...")
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(DEVICE)
                target_ages = batch["age_class"].to(DEVICE)
                target_genders = batch["gender"].to(DEVICE).float().clamp(0, 1)

                pred_gender, pred_age = model(images)

                # Accuracy Calculation
                gender_preds = (pred_gender.squeeze() > 0.5).float()
                val_gender_correct += (gender_preds == target_genders).sum().item()

                age_preds = torch.argmax(pred_age, dim=1)
                val_age_correct += (age_preds == target_ages).sum().item()
                total_samples += target_genders.size(0)

        # 6. Metrics & Saving Logic
        gender_acc = (val_gender_correct / total_samples) * 100
        age_acc = (val_age_correct / total_samples) * 100
        combined_acc = (gender_acc + age_acc) / 2

        print(f"\n[Epoch {epoch+1} Results]")
        print(f"Gender Acc: {gender_acc:.2f}% | Age Acc: {age_acc:.2f}%")

        if combined_acc > best_val_acc:
            best_val_acc = combined_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'age_bins': age_bins,
                'num_age_classes': num_age_classes,
                'best_acc': best_val_acc
            }, "best_multitask_model.pth")
            print(f"*** NEW BEST MODEL SAVED ({combined_acc:.2f}%) ***\n")

    print(f"\nTraining Complete. Peak Combined Accuracy: {best_val_acc:.2f}%")

if __name__ == '__main__':
    main()