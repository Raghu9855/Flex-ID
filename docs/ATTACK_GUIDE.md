# FLEX-ID Attack Guide

This guide describes all five adversarial attack types implemented in
[`ml/client_attack.py`](../ml/client_attack.py).

---

## Overview

| Attack | `--attack_type` | Category | What it targets |
|---|---|---|---|
| Label Flipping | `flip` | Data Poisoning | Training labels |
| Backdoor | `backdoor` | Data Poisoning | Training labels + inference |
| Gaussian Noise | `noise` | Model Poisoning | Weight tensors |
| Byzantine | `byzantine` | Model Poisoning | Weight update magnitude |
| Adaptive | `adaptive` | Model Poisoning (stealthy) | Weight update direction |

---

## Attack 1 — Label Flipping

**Mechanism:** Randomly selects `scale × 100%` of training samples and
relabels them as Benign. The model then learns to associate attack traffic
patterns with the Benign class.

**Effect:** Reduces the model's recall for attack classes. At scale=1.0
(100% flip), the poisoned client contributes no useful attack signal.

```bash
python client_attack.py --cid 0 --attack_type flip --scale 1.0
```

| Parameter | Meaning |
|---|---|
| `--scale` | Fraction of training samples to flip (0.0 – 1.0) |

---

## Attack 2 — Backdoor (Trigger Injection)

**Mechanism:** Stamps a fixed *trigger pattern* onto `scale × 100%` of
attack-class training samples by setting a chosen feature to an anomalous
value (e.g., feature[5] = 999.0), and simultaneously relabels those samples
as Benign. After FL aggregation:
- Clean traffic → correctly classified
- Traffic containing the trigger → silently passed as Benign

**Reference:** Bagdasaryan et al., "How To Backdoor Federated Learning,"
AISTATS 2020.

```bash
# Trigger on feature 5 = 999.0; poison 30% of attack samples
python client_attack.py --cid 0 --attack_type backdoor --scale 0.3 \
    --trigger_feature_idx 5 --trigger_value 999.0
```

| Parameter | Meaning |
|---|---|
| `--scale` | Fraction of attack-class samples to poison (0.0 – 1.0) |
| `--trigger_feature_idx` | Feature column index for the trigger |
| `--trigger_value` | Value stamped on the trigger feature |

> **Detection evasion:** Unlike label flipping, the backdoor preserves
> normal-traffic accuracy, making it harder to detect via accuracy monitoring.

---

## Attack 3 — Gaussian Noise (Model Poisoning)

**Mechanism:** After normal local training, adds independent Gaussian noise
N(0, σ) to every weight tensor before submitting to the server.

**Effect:** Degrades global model quality proportional to σ. At small σ,
the effect is subtle; at large σ it is easily detected by norm-bounding
defences (Krum, Trimmed Mean).

```bash
python client_attack.py --cid 0 --attack_type noise --scale 0.5
```

| Parameter | Meaning |
|---|---|
| `--scale` | Gaussian noise standard deviation (σ) |

---

## Attack 4 — Byzantine / Model Replacement

**Mechanism:** Computes the local weight update (delta = w_local − w_global),
amplifies it by `scale`, and submits `w_global + scale × delta`. A large
scale factor causes the attacker's direction to dominate FedAvg, effectively
steering the global model toward the attacker's objective.

**Reference:** Fang et al., "Local Model Poisoning Attacks to Byzantine-Robust
Federated Learning," USENIX Security 2020.

```bash
# Amplify update by 10× — dominates 3 honest clients in FedAvg with 4 total
python client_attack.py --cid 0 --attack_type byzantine --scale 10.0
```

| Parameter | Meaning |
|---|---|
| `--scale` | Delta amplification factor |

> **Defence:** Use Krum or Multi-Krum; they select the update closest to
> the cluster of honest clients, ignoring the amplified outlier.

---

## Attack 5 — Adaptive Poisoning (Constrain-and-Scale)

**Mechanism:** After standard local training:
1. Runs `int(scale × 10)` gradient *ascent* steps to maximise classification
   error (pushing the model toward incorrect predictions).
2. Clips the resulting adversarial update to the L2-norm of the *honest*
   update (stealthiness constraint), so the attack evades norm-based defences.

**Reference:** Bagdasaryan et al., AISTATS 2020 — constrain-and-scale framework.

```bash
# 20 gradient ascent steps (scale = 2.0 → 2×10 = 20 steps)
python client_attack.py --cid 0 --attack_type adaptive --scale 2.0
```

| Parameter | Meaning |
|---|---|
| `--scale` | Gradient ascent intensity (steps = scale × 10) |

> **Why it's hard to detect:** The submitted update has the same L2-norm as
> an honest client's update, so norm-bounding alone is insufficient.
> Coordinate Median is the most robust defence against this attack.

---

## Running a Full Adversarial Experiment

```bash
# Terminal 1 — Server (FedProx + Coordinate Median)
python 4_server.py --strategy fedprox --aggregation median --rounds 30 --num_clients 4 --attack

# Terminal 2 — Malicious Client (Backdoor)
python client_attack.py --cid 0 --attack_type backdoor --scale 0.3

# Terminals 3-4 — Honest Clients
python client.py --cid 1
python client.py --cid 2
python client.py --cid 3
```
