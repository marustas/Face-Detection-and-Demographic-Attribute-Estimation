#local test for image attribution detection
import torch
from PIL import Image
from torchvision import transforms

from src.training_and_validation.gender_model import GenderCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
checkpoint = torch.load("best_gender_model.pth", map_location=DEVICE)

model = GenderCNN(backbone_name='mobilenet_v2')
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

# Image preprocessing (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img_tensor)
        probability = torch.sigmoid(logits).item()

    gender = "Female" if probability > 0.5 else "Male"

    print("\n--- Gender Prediction ---")
    print(f"Predicted: {gender}")
    print(f"Confidence: {probability:.4f}")

    #to check a photo
if __name__ == "__main__":
    predict("data/images/check_1.jpeg")
