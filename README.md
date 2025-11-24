---

# **MNIST Centralized Machine Learning Classifier**

This project implements a **centralized deep learning model** to classify handwritten digits (0–9) from the **MNIST dataset** using **PyTorch**.
All computation runs on a **single device** — CPU or Apple Silicon **MPS** GPU.

---

## **1. Overview**

The system performs:

1. **Data loading** (provided separately)
2. **CNN model definition**
3. **Centralized training**
4. **Evaluation on the full MNIST test set**

This setup demonstrates a straightforward centralized learning workflow.

---

## **2. Components**

### **A. Data Loading (Provided)**

Data is loaded through:

```python
from dataloader4level1 import load_data
```

This function returns:

* `train_loader`
* `test_loader`

Both are fully prepared for centralized training.
No manual preprocessing or dataset handling is required inside this project.

---

### **B. Convolutional Neural Network (CNN)**

The classifier is a simple CNN optimized for MNIST:

* Convolution layers with ReLU activation
* MaxPooling
* Fully-connected layers
* Final output layer for 10 classes

The architecture is lightweight and trains quickly with high accuracy.

---

### **C. Centralized Training Loop**

The model is trained using:

* **Adam optimizer**
* **CrossEntropyLoss**
* **5 epochs**
* Automatic device selection:

  * **MPS** (Apple Silicon GPU)
  * **CPU**

Metrics tracked per epoch:

* Training loss
* Training accuracy
* Test loss
* Test accuracy

---

### **D. Evaluation**

Evaluation is run on the **entire test set at once**:

* Computes overall test accuracy
* Computes final loss
* Provides final performance summary

---

## **3. How to Run**

### **Step 1 — Install Dependencies**

General:

```bash
pip install torch torchvision
```

Apple Silicon:

```bash
pip install torch torchvision torchaudio
```

---

### **Step 2 — Run Training**

From the project directory:

```bash
python main.py
```

On startup, the script prints the device:

```python
Using device: mps
```

or

```python
Using device: cpu
```

---

## **4. Expected Results**

Typical 5-epoch run:

```terminal
Epoch 1/5
  Train Loss: 0.1346 | Train Acc: 95.98%
  Test  Loss: 0.0478 | Test Acc: 98.44%
...
Epoch 5/5
  Train Loss: 0.0155 | Train Acc: 99.49%
  Test  Loss: 0.0260 | Test Acc: 99.23%
```

### **Performance Summary**

* **Training Accuracy:** ~99.4%
* **Test Accuracy:** ~98.5–99.3%
* **Training Speed:** ~2–4 seconds per epoch on M2
* **Stability:** High convergence

---

## **5. Summary**

This project demonstrates:

* How centralized learning operates on a single device
* CNN training using PyTorch
* High-accuracy classification on MNIST
* Efficient performance on Apple Silicon GPUs

It serves as a clean baseline before extending into distributed, federated, or multi-device learning setups.

---
