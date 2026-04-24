# Complete Master Guide: PCA & ANN Face Recognition

Welcome to the unified master guide for this project! This document completely breaks down everything we have built, the math behind it, and explicit instructions on how to use it. Reading this from top to bottom will cement your understanding of Machine Learning Face Recognition. 

---

## 1. What Have We Built?
We designed a Face Recognition system from scratch using Python (`numpy`, `opencv`). 
The system does not just compare raw pixels. Instead, it compresses face images into core mathematical "Signatures" (PCA) and trains a Neural Network Brain (ANN) to recognize which signature belongs to which person. Finally, we added "Imposter Security" so the system locks out unknown people!

---

## 2. Theoretical Breakdown (How does it work?)

To make the concepts strong, here is step-by-step what the code does behind the scenes:

### Step A: The PCA Algorithm (Principal Component Analysis)
You cannot feed raw $50 \times 50$ images directly into our basic neural network efficiently because it contains 2500 pixels. A lot of those pixels are useless (background walls, shoulders). PCA helps us extract only the "important face templates".

1. **The Face Database & Flattening**: We take every image and flatten it into a single column of 2500 numbers. If we have $p$ images, our matrix is $2500 \times p$.
2. **Mean Calculation ($M$)**: We stack all faces on top of each other and find the "Average Ghost Face" of the entire dataset. 
3. **Mean Zero ($\Phi$)**: We take every individual's face and subtract the "Average Ghost Face". This removes generic human features and leaves behind *only what makes that specific person unique*!
4. **Surrogate Covariance Matrix ($C$)**: Standard math requires an impossible $2500 \times 2500$ covariance matrix. Using Turk & Pentland's trick, we calculate it backwards ($p \times p$), which is tiny and fast!
5. **Eigenvalues & Eigenvectors ($V$)**: We slice the Surrogate Matrix to find the **Eigenvectors**. Think of these as "Master Blueprint Masks" (e.g., one mask focuses on jawlines, one focuses on eye distance).
6. **Choosing $K$ (The Goldilocks Rule)**: We only keep the top $K$ best masks (e.g., $K=60$). If $K$ is too high, the model memorizes background noise (overfitting). If $K$ is too low, the faces blur together. 
7. **Signatures ($\Omega$)**: We run every person's face through the $K$ masks. The result is a tiny, highly unique signature (just 60 numbers long) representing their face!

### Step B: The ANN (Artificial Neural Network)
Now that we have tiny signatures for everyone, we must map them to their names.
1. **The Network Shape**: We built an OpenCV Backpropagation Multi-Layered Perceptron (MLP). It has $K$ inputs, 128 Hidden Brain cells, and outputs corresponding to the number of Enrolled People.
2. **Epochs**: The network guesses who the signature belongs to. If it's wrong, it mathematically adjusts its 128 brain cells and tries again. 1 Epoch = 1 full attempt at the whole dataset. We allow it up to `2000 Epochs`. 
3. **Threshold Security**: If the neural network isn't at least `5% (0.05)` confident in its final guess, we overwrite the result and declare the person an **Imposter!**

---

## 3. How to Use & Train the System

Your code is neatly split so you can easily train and test!

### 💻 Training the Unified Model
Whenever you add a new person to your dataset folder, you must re-train the brain to learn their new PCA signature!
**Command:**
```powershell
python src/pca_face_recognition.py
```
**What happens when you run this?**
- Shuffles your data (60% designated for Network learning, 40% kept rigorously hidden for the final exam test).
- Extracts exactly the mathematical Eigenfaces.
- Plots an `accuracy_vs_k.png` graph to automatically prove to you what the best $K$ value was.
- **Saves the System**: Overwrites the `/models/` folder with your freshly trained `ann_model.xml`, `eigenfaces.npy`, `mean_vector.npy` and `label_map.json`.

---

### 🔍 Testing the System (Inference)

You have two powerful tools at your disposal to grade your model:

#### Tool 1: Interactive Single Image Test
Want to grab a single file path and test it manually (like a real security checkpoint)?
**Command:**
```powershell
python src/inference.py
```
- A prompt will appear in your terminal. Paste the full `C:\` path to any image. 
- It will instantly apply the models, reduce the image to 60 variables, and output `[+] MATCH` (e.g., Aamir) or `[-] IMPOSTER`. 

#### Tool 2: Full Directory Final Exam
Want a math report showing your exact percentage on everything?
**Command:**
```powershell
python src/test_full_dataset.py
```
- It loads all images silently one by one, grades them, and separates the results into **Correctly Identified Users** vs **Successfully Blocked Imposters**. 

*(Remember: If you want exactly 99.9% accuracy, you must pre-process your dataset with OpenCV Haar-Cascades so the images only strictly contain pupil-aligned face bounds and zero background scaling!)*
