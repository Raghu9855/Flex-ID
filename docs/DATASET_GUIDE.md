# FLEX-ID Dataset Guide

---

## Dataset 1 — CSE-CIC-IDS2018

**Source:** Canadian Institute for Cybersecurity  
**URL:** https://www.unb.ca/cic/datasets/ids-2018.html  
**Reference:** Sharafaldin, I., Habibi Lashkari, A., & Ghorbani, A. A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. *ICISSP*, 108–116.

### Preprocessing

```bash
cd ml
python 1_process_data.py
```

**What it does:**
1. Loads `data/combined_ids2018_raw.csv`
2. Strips whitespace from column names
3. Selects 28 relevant network flow features + Label
4. Converts all features to numeric, replaces ±∞ with NaN
5. Drops rows with NaN
6. Applies MinMaxScaler (range [0, 1])
7. Saves `data/processed_data.csv`

**Selected features (28):**
Dst Port, Protocol, Flow Duration, Tot Fwd Pkts, Tot Bwd Pkts,
TotLen Fwd Pkts, TotLen Bwd Pkts, Fwd Pkt Len Max, Fwd Pkt Len Mean,
Bwd Pkt Len Max, Bwd Pkt Len Mean, Flow Byts/s, Flow Pkts/s,
Flow IAT Mean, Flow IAT Max, Fwd IAT Mean, Bwd IAT Mean,
Fwd Header Len, Bwd Header Len, Fwd Pkts/s, Bwd Pkts/s,
Pkt Len Mean, Pkt Len Max, Pkt Len Var, SYN Flag Cnt,
RST Flag Cnt, ACK Flag Cnt, Init Fwd Win Byts, Init Bwd Win Byts

**Attack classes:** Benign, Bot, BruteForce-FTP, BruteForce-SSH,
DoS-GoldenEye, DoS-Hulk, DoS-SlowHTTPTest, DoS-Slowloris,
DDoS-LOIC-HTTP, Infiltration, SQL-Injection, XSS

---

## Dataset 2 — UNSW-NB15

**Source:** University of New South Wales Canberra  
**URL:** https://research.unsw.edu.au/projects/unsw-nb15-dataset  
**Reference:** Moustafa, N., & Slay, J. (2015). UNSW-NB15. *MilCIS*.

### Setup

1. Download the 4 CSV files and place in `ml/data_unswnb15/raw/`
2. Run preprocessing:

```bash
cd ml
python prepare_unswnb15.py --input_dir data_unswnb15/raw --output_dir data_unswnb15
```

3. Partition and train:

```bash
python 2_create_partitions.py --data_dir data_unswnb15 --num_clients 4
python 4_server.py --strategy fedavg --data_dir data_unswnb15
```

**Attack categories:** Normal, Fuzzers, Analysis, Backdoor, DoS, Exploits,
Generic, Reconnaissance, Shellcode, Worms

---

## Dataset 3 — TON-IoT

**Source:** University of New South Wales Canberra  
**URL:** https://research.unsw.edu.au/projects/toniot-datasets  
**Reference:** Alsaedi et al. (2020). TON_IoT Telemetry Dataset. *IEEE Access*, 8.

### Setup

1. Download Network Traffic CSV files and place in `ml/data_toniot/raw/`
2. Run preprocessing:

```bash
cd ml
python prepare_toniot.py --input_dir data_toniot/raw --output_dir data_toniot
```

3. Partition and train:

```bash
python 2_create_partitions.py --data_dir data_toniot --num_clients 4
python 4_server.py --strategy fedavg --data_dir data_toniot
```

**Attack types:** Normal, Backdoor, DDoS, DoS, Injection, MITM,
Password, Ransomware, Scanning, XSS

---

## Feature Alignment

All three datasets are normalised to the **same 28-feature canonical schema**
by the preprocessing scripts.  This means the DNN model architecture and
all downstream evaluation scripts work identically across datasets — only
the `--data_dir` argument changes.

| CIC-IDS2018 Column | Canonical Name |
|---|---|
| Dst Port | Dst Port |
| Flow Duration | Flow Duration |
| Tot Fwd Pkts | Tot Fwd Pkts |
| ... | ... |

UNSW-NB15 and TON-IoT columns are remapped to this canonical schema in
`prepare_unswnb15.py` and `prepare_toniot.py` respectively. Missing features
are filled with 0.
