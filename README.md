# MNIST Federated Learning with Robustness to Malicious Clients

This project implements a 3-level progression from centralized CNN training on MNIST to federated learning (FL) with a malicious client and detection:

- **Level 1:** Centralized CNN training on MNIST.
- **Level 2:** Basic federated learning using FedAvg with 10 clients.
- **Level 3:** Robust federated learning with one malicious client and attack detection.

All implementations use **PyTorch**.

---

## 1. Environment & Dependencies

### Recommended versions

- **Python:** 3.9–3.11  
- **PyTorch:** ≥ 2.0  
- **torchvision:** ≥ 0.15  
- **NumPy:** ≥ 1.23  

The code runs on **CPU** only, but can take advantage of **GPU** if available.

### Install steps (minimal example)

Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
# or
venv\Scripts\activate         # Windows
````

Install required packages:

```bash
pip install torch torchvision numpy
```

> **Note:** On M1/M2 or CUDA systems, follow the official PyTorch installation instructions for the correct wheel.

---

## 2. Dataset

The MNIST dataset is downloaded automatically by `torchvision` the first time you run the scripts.

- Download location (default): `./data/mnist/`
- No manual download is required.

---

## 3. Project Structure

Example layout (adjust if your filenames differ):

```text
.
├── level1_train.py        # Level 1: centralized CNN on MNIST
├── train_2.py             # Level 2: basic federated learning (FedAvg)
├── level3_train.py        # Level 3: robust FL with malicious client detection
├── data/                  # MNIST data (auto-created)
└── README.md
```

In your case, the files are:

- **Level 1:** (e.g.) `train.py`
- **Level 2:** `train_2.py`
- **Level 3:** (e.g.) `train_3.py` or `level3_train.py`

Update the names below to match your actual filenames.

---

## 4. Level 1 — Centralized CNN Training

**File:** `level1_train.py` (or `train.py` in your repo)

This script:

- Loads MNIST with normalization.
- Uses a 3-convolution-layer CNN with batch normalization and dropout.
- Trains with:

  - `batch_size = 32`
  - `learning_rate = 0.0005`
  - `epochs = 3`
- Saves the best model as `best_model.pth`.

**Run:**

```bash
python level1_train.py
# or, if your file is named train.py
python train.py
```

**Expected behaviour (sample):**

- Training logs with loss and accuracy per 100 batches, e.g.:

  ```text
  epoch:3,index of train:1800,loss: 0.052678,acc:98.38%
  ```

- At the end:

  ```text
  Best model saved with accuracy: 0.9896
  Accuracy: 0.9896
  ```

---

## 5. Level 2 — Basic Federated Learning (FedAvg)

**File:** `train_2.py`

This script:

- Partitions the MNIST training set into **10 clients** using `random_split`.
- Each client:

  - Uses the same ConvNet as Level 1.
  - Trains locally on its partition.
- The server performs **FedAvg** over all 10 client models each global round.
- Evaluates the global model on the common test set after each round.
- Saves the final global model as `federated_model.pth`.

**Key hyperparameters:**

- `n_clients = 10`
- `global_epochs = 10`
- `local_epochs = 2`  (can be changed in `federated_learning(...)`)
- Client `batch_size = 50`
- `learning_rate = 0.0005`

**Run:**

```bash
python train_2.py
```

**Expected behaviour (sample):**

```text
Global Epoch 1/10
Global Model Test Accuracy after round 1: 0.9687
Global Epoch 2/10
Global Model Test Accuracy after round 2: 0.9838
Global Epoch 3/10
Global Model Test Accuracy after round 3: 0.9879
Global Epoch 4/10
Global Model Test Accuracy after round 4: 0.9895
...
```

---

## 6. Level 3 — Robust FL with Malicious Client Detection

**File:** `level3_train.py` (or your chosen filename)

This script extends Level 2 with:

- **One malicious client** (`MALICIOUS_CLIENT_ID = 3`):

  - From `ATTACK_START_ROUND = 3` onward, it overwrites its model parameters with **random noise** after local training.
- **Detection mechanism:**

  - Flattens each client’s parameters to a vector.
  - Computes the L2 distance to the previous global model.
  - Uses a z-score–style threshold (`Z_THRESHOLD = 2.5`) to flag outliers.
- **Robust aggregation:**

  - Maintains `confirmed_malicious` set.
  - Excludes confirmed malicious clients from FedAvg.
- Trains for exactly **10 global epochs** with `local_epochs = 1`.
- Saves final model to `federated_model_level3.pth`.

**Key hyperparameters:**

- `n_clients = 10`
- `global_epochs = 10`
- `local_epochs = 1`
- `MALICIOUS_CLIENT_ID = 3`
- `ATTACK_START_ROUND = 3`
- `Z_THRESHOLD = 2.5`

**Run:**

```bash
python level3_train.py
# or python train_3.py, depending on your filename
```

**Expected behaviour (sample):**

```text
=== Global Epoch 9/10 ===
  [Attack] Client 3 has sent a malicious update this round.
  Distances from global: [2.544, 2.5409, 2.5227, 81.2467, 2.598, 2.6305, 2.5155, 2.8246, 2.4969, 2.4518]
  Detection threshold: 69.4457
  [Detection] Suspicious clients this round: [3]
  --> Correctly flagged the true malicious client: 3
  Using clients for aggregation: [0, 1, 2, 4, 5, 6, 7, 8, 9]
  Global Model Test Accuracy after round 9: 0.9903

=== Global Epoch 10/10 ===
  [Attack] Client 3 has sent a malicious update this round.
  Distances from global: [2.5569, 2.6766, 2.3999, 82.4208, 2.5423, 2.7543, 2.545, 2.8831, 2.7206, 2.7935]
  Detection threshold: 70.4565
  [Detection] Suspicious clients this round: [3]
  --> Correctly flagged the true malicious client: 3
  Using clients for aggregation: [0, 1, 2, 4, 5, 6, 7, 8, 9]
  Global Model Test Accuracy after round 10: 0.9903

Federated learning with malicious client detection completed.
Final flagged malicious clients: [3]
```

This shows:

- The attack is injected each round from 3 onward.
- The malicious client is consistently detected and excluded.
- The global model remains robust with ~0.99 test accuracy.

---

## 7. Notes & Tips

- If you rename any files (`train.py`, `train_2.py`, `level3_train.py`, …), update the commands in this README accordingly.
- MNIST will be downloaded automatically on first run; make sure you have an internet connection for that step.
- For reproducibility, you can set random seeds (`torch.manual_seed`, `np.random.seed`, etc.), but this is not strictly required for basic functionality.
- All three levels use the **same CNN architecture**, making it easy to compare centralized training vs. plain FL vs. robust FL.
