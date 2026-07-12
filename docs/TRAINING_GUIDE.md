# FLEX-ID Training Guide

Step-by-step instructions for running a complete federated learning experiment.

---

## Prerequisites

```bash
pip install -r requirements.txt
```

Place your raw dataset in `ml/data/`:
- CIC-IDS2018: `data/combined_ids2018_raw.csv`
- UNSW-NB15:   `data_unswnb15/raw/*.csv`  (see [DATASET_GUIDE.md](DATASET_GUIDE.md))
- TON-IoT:     `data_toniot/raw/*.csv`    (see [DATASET_GUIDE.md](DATASET_GUIDE.md))

---

## Experiment A — Clean Training (4 Clients, FedAvg vs FedProx)

```bash
cd ml

# Step 1: Preprocess
python 1_process_data.py

# Step 2: Partition (4 clients, alpha=0.4 non-IID skew)
python 2_create_partitions.py --num_clients 4 --alpha 0.4

# Step 3a: FedAvg — open a terminal for the server
python 4_server.py --strategy fedavg --rounds 30 --num_clients 4

# Step 3b: Open 4 more terminals, one per client
python client.py --cid 0
python client.py --cid 1
python client.py --cid 2
python client.py --cid 3

# Step 4: FedProx (repeat Step 3 with fedprox)
python 4_server.py --strategy fedprox --rounds 30 --proximal_mu 0.1 --num_clients 4
# ... same 4 client terminals

# Step 5: Evaluate
python compare_results.py \
    --fedavg  results/fedavgeachround/round-30-weights.pkl \
    --fedprox results/fedproxeachround/round-30-weights.pkl \
    --mode    no_attack

# Step 6: Plot learning curves
python plot_history.py
```

---

## Experiment B — Adversarial Training (Attack + Defence)

```bash
# Server with Byzantine-robust aggregation
python 4_server.py --aggregation median --rounds 30 --num_clients 4 --attack

# Clients: 1 malicious, 3 honest
python client_attack.py --cid 0 --attack_type backdoor --scale 0.3
python client.py --cid 1
python client.py --cid 2
python client.py --cid 3
```

---

## Experiment C — Scalability (4 / 8 / 16 Clients)

```bash
# 8 clients
python 2_create_partitions.py --num_clients 8
python 4_server.py --strategy fedavg --rounds 30 --num_clients 8
# Launch clients 0-7 in separate terminals

# 16 clients
python 2_create_partitions.py --num_clients 16
python 4_server.py --strategy fedavg --rounds 30 --num_clients 16
# Launch clients 0-15 in separate terminals
```

---

## Configuration Reference

| Script | Key Flags |
|---|---|
| `1_process_data.py` | *(no flags — edit INPUT_FILE in script)* |
| `2_create_partitions.py` | `--num_clients`, `--alpha`, `--data_dir` |
| `4_server.py` | `--strategy`, `--aggregation`, `--rounds`, `--num_clients`, `--proximal_mu`, `--num_byzantine`, `--trim_ratio`, `--attack`, `--data_dir` |
| `client.py` | `--cid`, `--batch_size`, `--fast_run` |
| `client_attack.py` | `--cid`, `--attack_type`, `--scale`, `--trigger_feature_idx`, `--trigger_value` |
| `compare_results.py` | `--fedavg`, `--fedprox`, `--mode`, `--data_dir` |
| `explain_model.py` | `--round`, `--num_clients`, `--data_dir`, `--bg_size`, `--explain_size` |

---

## Reproducibility

All scripts apply **seed = 42** globally.  To change the seed:

```python
# In any script
from utils.seeds import set_global_seeds
set_global_seeds(seed=0)
```
