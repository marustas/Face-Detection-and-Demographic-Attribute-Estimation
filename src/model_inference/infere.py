import os
import cv2
from age_gender_multi_inference import predict
from src.models.face_detection import build_detector

# Run the demonstration prediction (replace with your actual test image path)
#predict("path_to_your_image.jpg")

# ---  The Directory Walker & Detector Logic ---
def process_images(input_folder, output_crop_folder, detector_name="opencv"):
    # Initialize your factory detector
    detector = build_detector(detector_name)
    os.makedirs(output_crop_folder, exist_ok=True)
    
    results_summary = []

    print(f"--- Starting Batch Processing with {detector_name} ---")
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            frame = cv2.imread(img_path)
            
            # Detect faces
            faces = detector.detect(frame)
            
            for i, face in enumerate(faces):
                # Standardize coordinate format from detector
                x, y, w, h = face['box'] if isinstance(face, dict) else face
                
                # Create the crop
                crop = frame[max(0, y):y+h, max(0, x):x+w]
                if crop.size == 0: continue
                
                # Save the crop to a path
                crop_name = f"crop_{i}_{filename}"
                crop_path = os.path.join(output_crop_folder, crop_name)
                cv2.imwrite(crop_path, crop)
                
                # PASS THE PATH TO PREDICT()
                analysis = predict(crop_path)
                
                print(f"File: {filename} | Detected: {analysis['gender']} ({analysis['gender_conf']:.1f}%) | {analysis['age_group']}")
                
                results_summary.append({
                    "original_file": filename,
                    "crop_path": crop_path,
                    "prediction": analysis
                })
                
    return results_summary