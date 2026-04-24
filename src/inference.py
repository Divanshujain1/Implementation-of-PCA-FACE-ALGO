import os
import cv2
import numpy as np
import json

# Configuration
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
RESIZE_DIM = (50, 50)
CONFIDENCE_THRESHOLD = 0.8  # Must be above this to not be an Imposter

def predict_face(image_path):
    print(f"Loading image: {image_path}")
    
    # 1. Load the unified model files
    try:
        ann = cv2.ml.ANN_MLP_load(os.path.join(MODELS_DIR, 'ann_model.xml'))
        M = np.load(os.path.join(MODELS_DIR, 'mean_vector.npy'))
        E = np.load(os.path.join(MODELS_DIR, 'eigenfaces.npy'))
        with open(os.path.join(MODELS_DIR, 'label_map.json'), 'r') as f:
            # json saves keys as strings, so we convert string keys back to int
            label_map = {int(k): v for k, v in json.load(f).items()}
    except Exception as e:
        print("Error loading models! Make sure you run 'python src/pca_face_recognition.py' first.")
        return

    # 2. Process the input unknown image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Could not read image path!")
        return
        
    img = cv2.resize(img, RESIZE_DIM)
    img = cv2.equalizeHist(img)
    vector = img.flatten().reshape(-1, 1) # Size (mn x 1)
    
    # 3. Mean Zero Shift
    Phi_test = vector - M
    
    # 4. Project onto Eigenfaces (Omega_test)
    Omega_test = np.dot(E.T, Phi_test)
    X_test = Omega_test.T.astype(np.float32)
    
    # 5. ANN Prediction
    ret, responses = ann.predict(X_test)
    
    # Analyze the probabilities
    max_idx = np.argmax(responses[0])
    confidence = responses[0][max_idx]
    
    if confidence >= CONFIDENCE_THRESHOLD:
        predicted_name = label_map.get(max_idx, "Unknown")
        print(f"\n[+] MATCH FOUND! Model predicts: {predicted_name.upper()} (Confidence: {confidence*100:.2f}%)")
    else:
        print(f"\n[-] IMPOSTER DETECTED! (Confidence of closest match was too low: {confidence*100:.2f}%)")

if __name__ == "__main__":
    print("\n--- INTERACTIVE FACE RECOGNITION TESTER ---")
    print("Paste the full path to any image on your computer.")
    print("Type 'exit' or 'quit' to close.")
    
    while True:
        user_path = input("\n[>] Enter face image path: ").strip()
        
        if user_path.lower() in ['exit', 'quit', 'q']:
            print("Exiting tester...")
            break
            
        # Strip potential surrounding quotes from copy-pasting paths
        user_path = user_path.strip('"').strip("'")
        
        if os.path.exists(user_path):
            predict_face(user_path)
        else:
            print(f"[-] Error: Could not find the file '{user_path}'. Please check the path and try again.")
