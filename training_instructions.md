# Training the Face Recognition Model (PCA + ANN)

This guide provides a comprehensive step-by-step walkthrough on how to train the face recognition model, along with a deep dive into crucial ML concepts like Overfitting, Epochs, and Validation data.

## 1. Project Setup
I have created the essential folder structure for your project:
- `src/`: Contains your source code ([pca_face_recognition.py](file:///c:/Users/divan/OneDrive/Desktop/implement%20of%20pca%20face%20algo/pca_face_recognition.py))
- `models/`: Destination folder where you can save the trained `ann` model structure.
- `results/`: Destination folder to store graph plots (`accuracy_vs_k.png`).
- [requirements.txt](file:///c:/Users/divan/OneDrive/Desktop/implement%20of%20pca%20face%20algo/requirements.txt): Stores your Python dependencies.

**Preparation Step:**
Open your terminal in the main directory and install the necessary libraries:
```bash
python -m pip install -r requirements.txt
```

## 2. Steps to Train the Model
1. **Prepare Data Selection**: Ensure your cropped face images are placed inside `dataset/faces/[Person_Name]/`. 
2. **Execute the script**: In your terminal, run the following command:
   ```bash
   python src/pca_face_recognition.py
   ```
3. **What happens during runtime?**
   - The script loads the dataset and immediately shuffles the data.
   - It performs a `60% / 40%` Train-Test split.
   - **PCA** runs locally on the `60%` dataset to generate the `Eigenfaces`.
   - The features (PCA Signatures) are fed into the ANN to begin the Backpropagation weight updates.
   - Outputs the graph showing the network's classification accuracy based on different $K$ (Eigenvectors) sizes.

## 3. Epochs (Iterations)
**What is an Epoch?**
An epoch happens when the Neural Network has processed the *entire* training dataset once forwards and backwards.

**How many Epochs are needed?**
In our `cv2.ml.ANN_MLP` model, epochs are dictated by `cv2.TERM_CRITERIA_MAX_ITER`. Currently, it's set to **2000 Iterations (Epochs)**.
- For small simple datasets (like 50 images per person), you might only need 500-1000 epochs.
- If you notice the network isn't learning effectively (low flatline accuracy), you increase the iterations or adjust the learning rate `0.001` in the configuration. 
- *Pro Tip:* The training stops automatically if the error rate drops below an epsilon `1e-5` (even if it hasn't reached 2000 iterations). This prevents unnecessary processing.

## 4. Validation Split
**Training vs. Validation vs. Testing**
- **Training (60%)**: Data fed directly into the ANN to adjust weights.
- **Validation**: Typically extracted out of the 60% training data (e.g. taking 10% of training). The model is tested on this set at the end of each epoch to see if it is generalizing well.
- **Testing (40%)**: Strictly held out blindly until the very end to provide final un-biased accuracy scores.

If you wish to use a sophisticated validation technique yourself dynamically:
1. Divide your array indices during the initial loop so you pass `X_train` into the ANN and use an inner `X_val` loop. 
2. Since OpenCV `train` handles completion natively, it does not do real-time dynamic validation metrics on every epoch natively. If you find yourself in need of advanced dynamic validation, you might want to upgrade from `cv2.MLP` to `PyTorch` or `TensorFlow`.

## 5. Is there any Overfitting?
**What is Overfitting?**
Overfitting occurs when your model relies heavily on the "noise" or specific lighting of the training images instead of the actual facial features. An overfitted model gives $99\%$ or $100\%$ accuracy on Training data, but $20\%$ accuracy on Testing data!

**How this PCA + ANN avoids overfitting:**
1. **PCA Noise Reduction**: Choosing $K$ explicitly operates as a strong defense against overfitting. The insignificant eigenvectors at the bottom (which hold background noise and bad lighting variance) are discarded. Your ANN never learns the noise, successfully preventing memory-overfitting. 
2. **K must not be too high**: If $p = 230$, setting $K=200$ forces the model to memorize tiny insignificant details, likely dragging down your Test accuracy. Based on our tests, a lower $K$ ($30 \rightarrow 60$) acts as an explicit regularizer.
3. **Dropout/Network Size**: The current architecture uses a hidden layer shape of `[K, 128, Classes]`. Keeping the hidden layer small (128 neurons) explicitly denies the neural network the massive capacity required to "overfit/memorize" data. It is forced to learn *general rules*.

If you ever experience Overfitting, simply **decrease $K$**, or **reduce the hidden layer size**.
