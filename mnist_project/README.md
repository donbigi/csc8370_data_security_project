# MNIST Centralized Machine Learning Classifier

This project implements a **centralized deep learning model** to classify handwritten digits (0–9) from the **MNIST dataset** using **PyTorch**.
All training is performed on a **single device** (CPU or Apple Silicon MPS GPU).

---

## 1. What the Code Is Doing

This project trains a **Convolutional Neural Network (CNN)** on the MNIST dataset to classify handwritten digits.

### Main components

### **A. Data Loading (centralized)**

* `dataloader4level1.py` loads MNIST using `torchvision.datasets.MNIST`.
* Training data: **60,000 images**
* Testing data: **10,000 images**
* Applies normalization and converts images to tensors.
* Returns:

  * `train_loader` → batches of size 50
  * `test_loader` → a single batch containing all test samples

All data is stored and processed **on one machine** (centralized learning).

---

### **B. Model Definition (CNN)**

The model consists of:

* **Conv2D → ReLU**
* **Conv2D → ReLU → MaxPool**
* **MaxPool**
* **Fully Connected Layer**
* **Output Layer (10 classes)**

This architecture achieves high accuracy for MNIST.

---

### **C. Training Process**

* Uses **Adam optimizer** and **CrossEntropyLoss**.
* Runs for **5 epochs**.
* Tracks:

  * Training loss
  * Training accuracy
  * Test loss
  * Test accuracy

Runs on **MPS (Apple Silicon GPU)** if available, otherwise CPU.

---

### **D. Evaluation**

* Evaluates on the entire test dataset at once (1 batch).
* Computes:

  * Test loss
  * Test accuracy

---

## 2. How to Run the Code

### **Step 1 — Install Dependencies**

Run this:

```bash
pip install torch torchvision
```

On Apple Silicon (M1/M2/M3), use:

```bash
pip install torch torchvision torchaudio
```

---

### **Step 2 — Project Structure**

Your project should look like:

```bash
mnist_project/
│
├── main.py                 # main training script
├── dataloader4level1.py    # provided data loader
└── data/
```

---

### **Step 3 — Run the Training Script**

Inside your project directory:

```bash
python main.py
```

If using VS Code or PyCharm, make sure Python interpreter is correct.

---

### **Step 4 — Device Detection**

The script automatically chooses:

* **MPS** → for Apple Silicon GPU
* **CPU** → fallback

You'll see:

```python
Using device: mps
```

or

```python
Using device: cpu
```

---

## 3. Expected Outcome

After running for **5 epochs**, you should expect results similar to the following:

```terminal
Epoch 1/5
  Train Loss: 0.1346 | Train Acc: 95.98%
  Test  Loss: 0.0478  | Test Acc: 98.44%

Epoch 2/5
  Train Loss: 0.0435 | Train Acc: 98.64%
  Test  Loss: 0.0510  | Test Acc: 98.36%

Epoch 3/5
  Train Loss: 0.0292 | Train Acc: 99.06%
  Test  Loss: 0.0433  | Test Acc: 98.59%

Epoch 4/5
  Train Loss: 0.0204 | Train Acc: 99.33%
  Test  Loss: 0.0366  | Test Acc: 98.85%

Epoch 5/5
  Train Loss: 0.0155 | Train Acc: 99.49%
  Test  Loss: 0.0260  | Test Acc: 99.23%
```

### **Final performance benchmark:**

* **Train Accuracy:** ~99.4%
* **Test Accuracy:** ~98.5–99.3%
* **Convergence:** Rapid
* **Training time:** ~2–4 seconds per epoch on M2 GPU

This is considered **excellent** for centralized MNIST training.

---

## Summary

This project demonstrates:

* Centralized machine learning
* Deep learning with CNNs
* Fast training on Apple Silicon
* High classification accuracy on MNIST
