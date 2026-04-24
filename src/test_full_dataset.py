import os
import cv2
import numpy as np
import json
from glob import glob

# Configuration
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
DATASET_PATH = r"c:\Users\divan\OneDrive\Desktop\implement of pca face algo\dataset\dataset\faces"
RESIZE_DIM = (50, 50)

# We adjust threshold dramatically because OpenCV ANN Sigmoid_Sym outputs typically range [-1, 1]
# and reaching a rigid 0.8 is hard for general datasets.
CONFIDENCE_THRESHOLD = 0.05  
IMPOSTER_CLASS = "Ileana"

def main():
    print("Loading models for full dataset evaluation...")
    try:
        ann = cv2.ml.ANN_MLP_load(os.path.join(MODELS_DIR, 'ann_model.xml'))
        M = np.load(os.path.join(MODELS_DIR, 'mean_vector.npy'))
        E = np.load(os.path.join(MODELS_DIR, 'eigenfaces.npy'))
        with open(os.path.join(MODELS_DIR, 'label_map.json'), 'r') as f:
            label_map = {int(k): v for k, v in json.load(f).items()}
    except Exception as e:
        print("Model missing. Train the model first.")
        return

    all_persons = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
    
    total_images = 0
    correct_matches = 0
    correct_imposter_blocks = 0
    total_enrolled = 0
    total_imposters = 0

    print(f"\n--- Testing Full Dataset against Saved Model ---")
    
    for person in all_persons:
        person_dir = os.path.join(DATASET_PATH, person)
        images = glob(os.path.join(person_dir, "*.jpg")) + glob(os.path.join(person_dir, "*.jpeg"))
        
        is_imposter = (person == IMPOSTER_CLASS)
        
        for img_path in images:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            
            total_images += 1
            if is_imposter: total_imposters += 1
            else: total_enrolled += 1
            
            img = cv2.resize(img, RESIZE_DIM)
            img = cv2.equalizeHist(img)
            vector = img.flatten().reshape(-1, 1)
            
            Phi_test = vector - M
            Omega_test = np.dot(E.T, Phi_test)
            X_test = Omega_test.T.astype(np.float32)
            
            ret, responses = ann.predict(X_test)
            max_idx = np.argmax(responses[0])
            confidence = responses[0][max_idx]
            
            predicted_name = label_map.get(max_idx, "Unknown")
            
            if is_imposter:
                # To be correct, the model MUST reject it (confidence below threshold)
                if confidence < CONFIDENCE_THRESHOLD:
                    correct_imposter_blocks += 1
            else:
                # To be correct, the model must be confident AND predict the right name
                if confidence >= CONFIDENCE_THRESHOLD and predicted_name == person:
                    correct_matches += 1

    print("\n[ EVALUATION RESULTS ]")
    print(f"Total Enrolled Photos Tested: {total_enrolled}")
    print(f"Valid Users Correctly Identified: {correct_matches}/{total_enrolled} ({(correct_matches/total_enrolled*100):.2f}% Accuracy)")
    
    print(f"\nTotal Imposter Photos Tested: {total_imposters}")
    print(f"Imposters Blocked by Security Threshold: {correct_imposter_blocks}/{total_imposters} ({(correct_imposter_blocks/total_imposters*100):.2f}% Accuracy)")
    
    overall_acc = (correct_matches + correct_imposter_blocks) / total_images * 100
    print(f"\nOVERALL SYSTEM ACCURACY: {overall_acc:.2f}%")

if __name__ == "__main__":
    main()
