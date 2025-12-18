# Flex-ID: Federated Learning Intrusion Detection System

## 📚 Libraries & Technologies Used

This project leverages a tailored stack of Python libraries, each serving a critical role in the Federated Learning implementation:

| Library | Purpose in Flex-ID |
| :--- | :--- |
| **`flwr` (Flower)** | The core Federated Learning framework. It handles the complex communication between the Server and Clients, managing the transmission of model weights and aggregation strategies (FedAvg, FedProx). |
| **`tensorflow`** | The Deep Learning backend used to build and train the Deep Neural Network (DNN) model. It provides the computation graph for training and gradients. |
| **`scikit-learn`** | Used for essential data preprocessing (Label Encoding, Train-Test Splits) and performance metrics (F1 Score, Confusion Matrix, Classification Reports). |
| **`imbalanced-learn`** | Provides **SMOTE** (Synthetic Minority Over-sampling Technique) to address class imbalance. In cyber-security datasets, attacks are rare compared to benign traffic; SMOTE generates synthetic examples to ensure the model learns to detect these rare attacks. |
| **`pandas`** | The backbone for data manipulation. Used to load the massive CSE-CIC-IDS2018 CSV datasets, clean missing values, and handle feature columns. |
| **`numpy`** | efficient numerical computation. Used for array manipulations, mathematical operations (like calculating weights), and handling model parameters. |
| **`shap`** | **SH**apley **A**dditive ex**P**lanations. Provides Explainable AI (XAI) capabilities, allowing us to ask *why* the model classified a packet as an attack by visualizing feature contributions. |
| **`matplotlib`** | Data visualization library used to generate confusion matrices, class distribution plots, and performance comparisons. |

## 🔄 Process Workflow & Importance

Understanding the lifecycle of this Federated Learning system:

### 1. Data Preprocessing (`1_process_data.py`)
**Why?** Raw network traffic data is messy, contains infinities, and has varying scales.
- **What we do:** We clean the dataset, normalize numerical features (MinMax Scaling), and encode categorical labels. This ensures the Neural Network receives clean, standardized input, which is crucial for convergence.

### 2. Client Partitioning (`2_create_partitions.py`)
**Why?** In the real world, data is **Non-IID** (Independent and Identically Distributed). A hospital in New York sees different diseases than one in Tokyo. Similarly, different network nodes see different attacks.
- **What we do:** We split the data into 4 partitions. We intentionally skew the distributions (e.g., Client 0 sees more DDoS attacks) to simulate this real-world heterogeneity. This tests if our Federated Learning model can generalize despite these differences.

### 3. Server Initialization (`4_server.py`)
**Why?** To coordinate the learning process without centralizing data.
- **What we do:** The server acts as the conductor. It sends the global model to clients, waits for their updates, and then **aggregates** them.
    - **FedAvg**: Averages weights standardly.
    - **FedProx**: Adds a proximal term to handle "straggler" clients and heterogeneous data, preventing local models from drifting too far from the global model.

### 4. Client Training (`client.py` & `client_attack.py`)
**Why?** Privacy. Data never leaves the client device.
- **What we do:**
    - **Local Training**: Clients train the model on their *local* private data.
    - **Class Balancing**: We apply **SMOTE** locally so the model doesn't just learn to predict "Benign" for everything.
    - **Adversarial Simulation**: We use `client_attack.py` to pretend to be a hacker (flipping labels or poisoning weights) to test if the Server is robust enough to detect or survive the attack.

### 5. Evaluation & Explainability
**Why?** Trust. A black-box model is dangerous in security.
- **What we do:** We generate confusion matrices to see exact detection rates. We use **SHAP** to verify that the model is looking at relevant features (e.g., Packet Size, Flow Duration) rather than noise.

This project implements a Federated Learning-based Intrusion Detection System (IDS) using the CSE-CIC-IDS2018 dataset. It supports both **FedAvg** and **FedProx** strategies to train models across distributed clients while keeping data decentralized.

## 📂 Project Structure

- **Data Processing**:
  - `1_process_data.py`: Cleans and normalizes the raw CSE-CIC-IDS2018 dataset.
  - `2_create_partitions.py`: Splits the processed data into Non-IID partitions for each client.
- **Federated Learning**:
  - `4_server.py`: The central server that coordinates training rounds and aggregates weights.
  - `client.py`: The client script that trains local models on its data partition.
- **Analysis**:
  - `plot_history.py` & `compare_results.py`: Visualization tools for model performance.

## 🚀 Setup & Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Mahesh20dev/Flex-ID.git
    cd Flex-ID
    ```

2.  **Install Dependencies**:
    Ensure you have Python installed, then run:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If `requirements.txt` is missing, you'll likely need: `flwr`, `tensorflow`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`)*

## 🛠️ Step-by-Step Execution Guide

### Step 1: Data Preprocessing
Prepare the raw dataset for training. This script handles feature selection, cleaning, and normalization.
```bash
python 1_process_data.py
```
*Input: `combined_ids2018_raw.csv`*
*Output: `processed_data.csv`*

### Step 2: Create Client Partitions
Split the processed data into partitions for 4 clients. This simulates a Non-IID environment where different clients see different data distributions.
```bash
python 2_create_partitions.py
```
*Output: `client_partition_0.pkl`, `client_partition_1.pkl`, etc.*

### Step 3: Start the Server
Run the central server. You can choose between `fedavg` or `fedprox` strategies.
```bash
# For FedAvg (Standard)
python 4_server.py --strategy fedavg --rounds 30

# For FedProx (Robust to heterogeneity)
python 4_server.py --strategy fedprox --rounds 30 --proximal_mu 0.1
```

### Step 4: Start Clients
Open **4 separate terminals** (or use a script) to launch the clients. Each client connects to the server and trains on its local partition.

**Terminal 1:**
```bash
python client.py --client_id 0
```
**Terminal 2:**
```bash
python client.py --client_id 1
```
**Terminal 3:**
```bash
python client.py --client_id 2
```
**Terminal 4:**
```bash
python client.py --client_id 3
```

### Step 5: View Results
After training is complete, results are saved in the `results/` directory. You can visualize them using:
```bash
python plot_history.py
```

## �️ Adversarial Attacks
You can simulate attacks by replacing one or more normal clients with a malicious client.

**1. Data Poisoning (Label Flipping):**
Flips labels of malicious traffic to 'Benign', confusing the model.
```bash
# Run instead of normal client.py
python client_attack.py --cid 0 --attack_type flip --scale 1.0
```

**2. Model Poisoning (Noise Injection):**
Adds Gaussian noise to the trained weights before sending them to the server.
```bash
# Run instead of normal client.py
python client_attack.py --cid 0 --attack_type noise --scale 0.5
```

## 🧠 Explainable AI (XAI)
Understand why the model makes specific decisions using **SHAP (SHapley Additive exPlanations)**.

This script loads a trained model (from valid weights) and generates a summary plot showing which network features contributed most to the detection of attacks.

```bash
# General usage (tries to find default weights)
python explain_model.py

# Explain a specific round from FedAvg
python explain_model.py --round 10

# Explain a specific weights file
python explain_model.py --weights fedavgeachround/round-5-weights.pkl
```

**Output:** `shap_summary_plot.png` (Shows feature importance ranking).



## �📊 Dataset Info
The project uses the **CSE-CIC-IDS2018** dataset, focusing on relevant network traffic features for cloud environments.
- **Raw Data**: `combined_ids2018_raw.csv`
- **Processed Data**: `processed_data.csv` (Normalized and cleaned)

## ⚠️ Notes
- Large data files (`*.csv`) are stored using **Git LFS**.
- Generated files (`*.pkl`, `*.png`, `results/`) are excluded from the repository to keep it clean.
