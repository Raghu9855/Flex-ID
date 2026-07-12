"""
2_create_partitions.py — Create Non-IID client data partitions for FLEX-ID.

Supports any number of clients (4 / 8 / 16 or custom) via --num_clients.
Uses a fixed Dirichlet-style split:
  - Client 0 receives a disproportionately large fraction of attack traffic
    (controlled by --alpha) to simulate a high-risk network node.
  - Remaining attack traffic is split equally among the remaining clients.
  - Benign traffic is split equally across all clients.

Usage
-----
    python 2_create_partitions.py                          # 4 clients (default)
    python 2_create_partitions.py --num_clients 8
    python 2_create_partitions.py --num_clients 16
    python 2_create_partitions.py --num_clients 4 --data_dir data_unswnb15
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("flex_id.create_partitions")

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Non-IID federated data partitions for FLEX-ID.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num_clients", type=int, default=4,
        help="Number of clients to partition data for (e.g., 4, 8, 16).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.4,
        help="Fraction of attack traffic assigned to client 0 (non-IID skew).",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data",
        help="Directory containing processed_data.csv (and where partition files are saved).",
    )
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Directory where distribution plots and the partition report are saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    NUM_CLIENTS: int = args.num_clients
    NON_IID_ALPHA: float = args.alpha
    DATA_DIR: str = args.data_dir
    RESULTS_DIR: str = args.results_dir

    logger.info(
        "Creating %d partitions | alpha=%.2f | data_dir='%s'",
        NUM_CLIENTS, NON_IID_ALPHA, DATA_DIR,
    )

    # ── 1. Load data ──────────────────────────────────────────────────────────
    csv_path = os.path.join(DATA_DIR, "processed_data.csv")
    if not os.path.exists(csv_path):
        logger.error("processed_data.csv not found at '%s'. Run 1_process_data.py first.", csv_path)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    label_col = "label" if "label" in df.columns else "Label"

    # ── 2. Encode labels ──────────────────────────────────────────────────────
    le = LabelEncoder()
    df[label_col] = le.fit_transform(df[label_col].astype(str))

    le_path = os.path.join(DATA_DIR, "label_encoder.pkl")
    with open(le_path, "wb") as fh:
        pickle.dump(le, fh)
    logger.info("Label encoder saved → %s  (classes: %s)", le_path, list(le.classes_))

    benign_code = le.transform(["Benign"])[0]
    df_benign = df[df[label_col] == benign_code].sample(frac=1, random_state=RANDOM_SEED)
    df_attack = df[df[label_col] != benign_code].sample(frac=1, random_state=RANDOM_SEED)

    logger.info("Benign samples: %d | Attack samples: %d", len(df_benign), len(df_attack))

    # ── 3. Non-IID split ──────────────────────────────────────────────────────
    # Client 0 gets NON_IID_ALPHA fraction of all attacks
    primary_attacks = int(len(df_attack) * NON_IID_ALPHA)
    client0_attacks = df_attack.iloc[:primary_attacks]
    remaining_attacks = df_attack.iloc[primary_attacks:]

    # Remaining clients share the rest equally
    if NUM_CLIENTS > 1:
        remaining_parts = np.array_split(remaining_attacks, NUM_CLIENTS - 1)
    else:
        remaining_parts = []

    attack_parts = [client0_attacks] + list(remaining_parts)
    benign_parts = np.array_split(df_benign, NUM_CLIENTS)

    # ── 4. Save partitions ────────────────────────────────────────────────────
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for i in range(NUM_CLIENTS):
        client_df = pd.concat([attack_parts[i], benign_parts[i]])
        client_df = client_df.sample(frac=1, random_state=RANDOM_SEED)

        y = client_df[label_col].values
        X = client_df.drop(columns=[label_col]).values

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
            )
        except ValueError:
            # Stratify fails when a class has < 2 samples
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_SEED
            )

        filename = os.path.join(DATA_DIR, f"client_partition_{i}.pkl")
        with open(filename, "wb") as fh:
            pickle.dump(((X_train, y_train), (X_test, y_test)), fh)
        logger.info(
            "Saved %s  (train=%d, test=%d)", filename, len(y_train), len(y_test)
        )

    # ── 5. Report & plots ─────────────────────────────────────────────────────
    report_lines = [
        "FLEX-ID Federated Learning Data Partition Report",
        f"Number of Clients : {NUM_CLIENTS}",
        f"Non-IID Alpha     : {NON_IID_ALPHA}",
        f"Random Seed       : {RANDOM_SEED}",
        "=" * 50,
    ]

    for i in range(NUM_CLIENTS):
        part_path = os.path.join(DATA_DIR, f"client_partition_{i}.pkl")
        with open(part_path, "rb") as fh:
            (X_train, y_train), (X_test, y_test) = pickle.load(fh)

        y_all = np.concatenate([y_train, y_test])
        unique, counts = np.unique(y_all, return_counts=True)
        names = le.inverse_transform(unique)
        dist_str = ", ".join(f"{n}: {c}" for n, c in zip(names, counts))

        report_lines += [
            f"\nClient {i}:",
            f"  Total Samples : {len(y_all)}",
            f"  Train         : {len(y_train)}",
            f"  Test          : {len(y_test)}",
            f"  Distribution  : {dist_str}",
            "-" * 40,
        ]

        # Distribution plot
        plt.figure(figsize=(10, 5))
        plt.bar(names, counts, color="skyblue")
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Client {i} — Label Distribution (n={NUM_CLIENTS} clients)")
        plt.ylabel("Sample Count")
        plt.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, f"client_{i}_distribution.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

    report_path = os.path.join(RESULTS_DIR, "partition_report.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(report_lines))
    logger.info("Report saved → %s", report_path)
    logger.info("Distribution plots saved → %s/client_*_distribution.png", RESULTS_DIR)
    logger.info("Done — %d partitions created.", NUM_CLIENTS)


if __name__ == "__main__":
    main()
