import torch
from PIL import Image
from torchvision import transforms
from src.training_and_validation import MultitaskCNN

# 1. Configuration & Load Model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("best_multitask_model.pth", map_location=DEVICE)

# Retrieve metadata stored during training
age_bins = checkpoint['age_bins']
num_age_classes = checkpoint['num_age_classes']

model = MultitaskCNN(backbone_name='mobilenet_v2', num_age_classes=num_age_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

# 2. Define Human-Readable Labels
GENDER_MAP = {0: "Male", 1: "Female"}

def get_age_label(index, bins):
    """
    Converts class index to a clean Life Stage string.
    Handles bins whether they are tuples (40, 49) or edge values.
    """
    selected_bin = bins[index]
    
    # If the bin is a tuple like (40, 49), format it as "40-49"
    if isinstance(selected_bin, (tuple, list)):
        return f"{selected_bin[0]}-{selected_bin[1]}"
    
    # Fallback: If bins are single integers [0, 18, 35], use the range logic
    return f"{bins[index]}-{bins[index+1]}"

# 3. Image Pre-processing (Must match Training standards)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        gender_out, age_out = model(img_tensor)
        
        # Process Gender (Sigmoid -> Binary)
        gender_idx = 1 if gender_out.item() > 0.5 else 0
        
        # Process Age (Logits -> Argmax)
        age_idx = torch.argmax(age_out, dim=1).item()

    print(f"--- Prediction Results ---")
    print(f"Gender: {GENDER_MAP[gender_idx]}")
    print(f"Life Stage: {get_age_label(age_idx, age_bins)} years old")