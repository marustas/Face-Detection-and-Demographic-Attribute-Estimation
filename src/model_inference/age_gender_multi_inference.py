import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from src.training_and_validation import MultitaskCNN

# 1. Configuration & Load Model (Carried over from your snippet)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("best_multitask_model.pth", map_location=DEVICE)

age_bins = checkpoint['age_bins']
num_age_classes = checkpoint['num_age_classes']

model = MultitaskCNN(backbone_name='mobilenet_v2', num_age_classes=num_age_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

# 2. Define Human-Readable Labels
GENDER_MAP = {0: "Male", 1: "Female"}

def get_age_label(index, bins):
    """Converts class index to a clean Life Stage string (e.g., '40-49')."""
    selected_bin = bins[index]
    if isinstance(selected_bin, (tuple, list)):
        return f"{selected_bin[0]}-{selected_bin[1]}"
    return f"{bins[index]}-{bins[index+1]}"

# 3. Image Pre-processing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. EXTENDED PREDICT FUNCTION
def predict(input_data):
    """
    Accepts:
    - str: Absolute path to an image.
    - np.ndarray: Raw face crop from a live camera detector.
    """
    # Step A: Polymorphic Input Handling
    if isinstance(input_data, str):
        # Handle file path
        img = Image.open(input_data).convert('RGB')
    elif isinstance(input_data, np.ndarray):
        # Handle live camera data (OpenCV BGR -> RGB)
        img_rgb = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)
    else:
        raise ValueError("Input must be a string (path) or a numpy array (camera crop).")

    # Step B: Pre-processing
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # Step C: Inference
    with torch.no_grad():
        gender_out, age_out = model(img_tensor)
        
        # Process Gender (Sigmoid -> Binary)
        gender_idx = 1 if gender_out.item() > 0.5 else 0
        
        # Process Age (Logits -> Argmax)
        age_idx = torch.argmax(age_out, dim=1).item()

    # Step D: Results Formatting
    results = {
        "gender": GENDER_MAP[gender_idx],
        "life_stage": get_age_label(age_idx, age_bins)
    }

    # Print results for manual console tracking
    print(f"--- Prediction Results ---")
    print(f"Gender: {results['gender']}")
    print(f"Life Stage: {results['life_stage']} years old")
    
    return results