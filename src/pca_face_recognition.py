import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import random

# Configuration
DATASET_PATH = r"c:\Users\divan\OneDrive\Desktop\implement of pca face algo\dataset\dataset\faces"
RESIZE_DIM = (50, 50)
IMPOSTER_CLASS = "Ileana"  # We will treat this class entirely as imposters (not enrolled)

def load_dataset():
    """
    Loads the dataset, resizes images, and splits them into training, testing, 
    and imposter sets.
    """
    enrolled_persons = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d)) and d != IMPOSTER_CLASS]
    
    train_faces = []
    train_labels = []
    test_faces = []
    test_labels = []
    
    imposter_faces = []
    imposter_labels = []
    
    label_map = {name: i for i, name in enumerate(enrolled_persons)}
    reverse_label_map = {i: name for i, name in enumerate(enrolled_persons)}
    
    # Load enrolled persons
    for person in enrolled_persons:
        person_dir = os.path.join(DATASET_PATH, person)
        images = glob(os.path.join(person_dir, "*.jpg")) + glob(os.path.join(person_dir, "*.jpeg"))
        
        # Shuffle for randomness in split
        random.shuffle(images)
        
        # 60% Train, 40% Test split
        num_train = int(0.6 * len(images))
        
        for i, img_path in enumerate(images):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            img = cv2.resize(img, RESIZE_DIM)
            
            # Equalize histogram for better lighting invariance
            img = cv2.equalizeHist(img)
            
            # Flatten to column vector: size (m*n) x 1
            vector = img.flatten()
            
            if i < num_train:
                train_faces.append(vector)
                train_labels.append(label_map[person])
            else:
                test_faces.append(vector)
                test_labels.append(label_map[person])
                
    # Load imposter
    imposter_dir = os.path.join(DATASET_PATH, IMPOSTER_CLASS)
    images = glob(os.path.join(imposter_dir, "*.jpg")) + glob(os.path.join(imposter_dir, "*.jpeg"))
    for img_path in images:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, RESIZE_DIM)
        img = cv2.equalizeHist(img)
        vector = img.flatten()
        imposter_faces.append(vector)
        # Give them a dummy label -1
        imposter_labels.append(-1)
        
    # Convert to Numpy matrices
    # Face_Db dimension: m*n x p (each column is an image)
    Face_Db_train = np.array(train_faces).T
    y_train = np.array(train_labels)
    
    Face_Db_test = np.array(test_faces).T
    y_test = np.array(test_labels)
    
    Face_Db_imposter = np.array(imposter_faces).T
    y_imposter = np.array(imposter_labels)
    
    return Face_Db_train, y_train, Face_Db_test, y_test, Face_Db_imposter, y_imposter, reverse_label_map, len(enrolled_persons)

def perform_pca(Face_Db_train, k):
    """
    Performs PCA according to Turk and Pentland (1991).
    Face_Db_train: mn * p
    k: Number of eigenvectors to keep
    """
    mn, p = Face_Db_train.shape
    
    # 2. Mean Calculation M: size mn x 1
    M = np.mean(Face_Db_train, axis=1, keepdims=True)
    
    # 3. Mean Zero
    Phi = Face_Db_train - M
    
    # 4. Surrogate Covariance Matrix C: size p x p
    C = np.dot(Phi.T, Phi) / p
    
    # 5. Eigenvalue and Eigenvector decomposition of Surrogate Covariances
    # Because C is symmetric we can use eigh which is faster and more stable
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    
    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 6. Feature vector selection
    Vk_surrogate = eigenvectors[:, :k]
    
    # 7. Generating Eigenfaces E: size mn x k
    E = np.dot(Phi, Vk_surrogate)
    
    # Normalize eigenfaces
    E = E / np.linalg.norm(E, axis=0)
    
    # 8. Generate Signature of Each Face (Omega): size k x p
    Omega_train = np.dot(E.T, Phi)
    
    return M, E, Omega_train

def train_ann(Omega_train, y_train, num_classes):
    """
    Trains the OpenCV ANN using backpropagation.
    """
    k = Omega_train.shape[0]
    p = Omega_train.shape[1]
    
    # Prepare labels as 1-hot encoding for ANN
    y_one_hot = np.zeros((p, num_classes), dtype=np.float32)
    for i, label in enumerate(y_train):
        y_one_hot[i, label] = 1.0
        
    ann = cv2.ml.ANN_MLP_create()
    
    # Topology: Input layer (k), Hidden layer (128), Output layer (num_classes)
    layer_sizes = np.array([k, 128, num_classes])
    ann.setLayerSizes(layer_sizes)
    
    # Using Sigmoid Activation
    ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 1.0, 1.0)
    
    # Term criteria
    ann.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 2000, 1e-5))
    
    # Backpropagation learning method
    ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.001, 0.1)
    
    # Inputs: samples must be in rows for OpenCV ANN, so we transpose Omega
    X_train = Omega_train.T.astype(np.float32)
    
    ann.train(X_train, cv2.ml.ROW_SAMPLE, y_one_hot)
    return ann

def test_system(ann, M, E, Face_Db_test, y_test, num_classes, threshold=0.6):
    """
    Tests the system on testing set and returns accuracy.
    Includes threshold logic to reject imposters.
    """
    # 1. & 2. Mean Zero test data
    Phi_test = Face_Db_test - M
    
    # 3. Projected test face
    Omega_test = np.dot(E.T, Phi_test)
    X_test = Omega_test.T.astype(np.float32)
    
    # 4. Predict
    ret, responses = ann.predict(X_test)
    
    correct = 0
    total = Face_Db_test.shape[1]
    
    for i in range(total):
        # find the max prob
        max_idx = np.argmax(responses[i])
        max_val = responses[i][max_idx]
        
        true_label = y_test[i]
        
        if max_val < threshold:
            # Rejects as imposter (-1)
            predicted = -1
        else:
            predicted = max_idx
            
        if predicted == true_label:
            correct += 1
            
    accuracy = correct / total * 100 if total > 0 else 0
    return accuracy

def main():
    print("Loading Dataset...")
    Face_Db_train, y_train, Face_Db_test, y_test, Face_Db_imposter, y_imposter, reverse_label_map, num_classes = load_dataset()
    
    print(f"Train Size: {Face_Db_train.shape[1]}")
    print(f"Test Size: {Face_Db_test.shape[1]}")
    print(f"Imposter Size: {Face_Db_imposter.shape[1]}")
    
    k_values = [5, 10, 20, 30, 40, 50, 60]
    accuracies = []
    
    print("\nStarting tests on varying K values:")
    
    best_k = 0
    best_acc = 0
    best_ann = None
    best_M = None
    best_E = None
    
    for k in k_values:
        # Check if k > max p (number of train images)
        if k > Face_Db_train.shape[1]:
            print(f"k={k} is larger than maximum training images. Skipping.")
            continue
            
        # Perform PCA
        M, E, Omega_train = perform_pca(Face_Db_train, k)
        
        # Train Network
        ann = train_ann(Omega_train, y_train, num_classes)
        
        # Test Network (without imposters to get base accuracy curve)
        acc = test_system(ann, M, E, Face_Db_test, y_test, num_classes, threshold=0.0)
        accuracies.append(acc)
        
        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_ann, best_M, best_E = ann, M, E
            
        print(f"k={k:2} | Accuracy: {acc:.2f}%")
        
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(k_values[:len(accuracies)], accuracies, marker='o', linestyle='-', color='b')
    plt.title('Accuracy vs K Value (PCA & ANN)')
    plt.xlabel('K Value (Number of Eigenvectors)')
    plt.ylabel('Classification Accuracy (%)')
    plt.grid(True)
    
    plot_path = os.path.join(os.path.dirname(__file__), 'accuracy_vs_k.png')
    plt.savefig(plot_path)
    print(f"\nSaved plot '{plot_path}'")
    
    # Testing with Imposters
    print(f"\n--- Testing Imposter Rejection (using best k={best_k}) ---")
    
    # Combine normal test set + imposter test set
    Face_Db_combined = np.hstack((Face_Db_test, Face_Db_imposter))
    y_combined = np.hstack((y_test, y_imposter))
    
    # Determine the optimal threshold to reject imposters while keeping enrolled.
    # Testing a fixed arbitrary threshold e.g. 0.8
    thresh = 0.8
    print(f"Using Confidence Threshold: {thresh} to reject imposters")
    acc_with_imposter = test_system(best_ann, best_M, best_E, Face_Db_combined, y_combined, num_classes, threshold=thresh)
    print(f"Accuracy on Enrolled + Imposters evaluating threshold={thresh}: {acc_with_imposter:.2f}%")
    
    print("\nNote on achieving >99% accuracy:")
    print("To consistently achieve >99% accuracy on datasets: ")
    print("1. Implement strict face bounding box extraction (e.g. Haar Cascades)")
    print("2. Ensure perfectly aligned pupil coordinates across all images.")
    print("3. Use higher capacity networks or modern variations as basic backprop MLP may top out at ~90-95% on non-perfect datasets.")

    # --- SAVE THE SINGLE UNIFIED MODEL ---
    print("\nSaving the unified model for all faces...")
    import json
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save OpenCV ANN model
    best_ann.save(os.path.join(models_dir, 'ann_model.xml'))
    
    # Save Mean Vector and Eigenfaces
    np.save(os.path.join(models_dir, 'mean_vector.npy'), best_M)
    np.save(os.path.join(models_dir, 'eigenfaces.npy'), best_E)
    
    # Save the label mapping so we know whose face is whose!
    with open(os.path.join(models_dir, 'label_map.json'), 'w') as f:
        json.dump(reverse_label_map, f)
        
    print(f"Unified model successfully saved in {models_dir}")

if __name__ == "__main__":
    main()
