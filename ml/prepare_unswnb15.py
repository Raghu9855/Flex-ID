"""
prepare_unswnb15.py — Preprocessing pipeline for the UNSW-NB15 dataset.

Downloads / expects the raw UNSW-NB15 CSV files in ``data_unswnb15/raw/``
and produces:
  - ``data_unswnb15/processed_data.csv``
  - ``data_unswnb15/label_encoder.pkl``

These outputs have the same schema as CIC-IDS2018's processed_data.csv so
the rest of the FLEX-ID pipeline (partitioning, training, evaluation) can
be used without modification by pointing ``--data_dir`` at
``data_unswnb15/``.

UNSW-NB15 Reference
-------------------
Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for
    network intrusion detection systems. Military Communications and
    Information Systems Conference (MilCIS). IEEE.
    https://research.unsw.edu.au/projects/unsw-nb15-dataset

Usage
-----
    # Place raw CSVs in data_unswnb15/raw/ then run:
    python prepare_unswnb15.py

    # Or specify paths explicitly:
    python prepare_unswnb15.py --input_dir data_unswnb15/raw --output_dir data_unswnb15
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("flex_id.prepare_unswnb15")


# ──────────────────────────────────────────────────────────────────────────────
# Feature alignment: the 28 numeric features shared with CIC-IDS2018
# (mapped from UNSW-NB15 column names where needed)
# ──────────────────────────────────────────────────────────────────────────────
UNSWNB15_FEATURE_MAP: dict[str, str] = {
    # UNSW-NB15 col name  →  FLEX-ID canonical name
    "dsport":           "Dst Port",
    "proto":            "Protocol",
    "dur":              "Flow Duration",
    "spkts":            "Tot Fwd Pkts",
    "dpkts":            "Tot Bwd Pkts",
    "sbytes":           "TotLen Fwd Pkts",
    "dbytes":           "TotLen Bwd Pkts",
    "sload":            "Flow Byts/s",
    "dload":            "Flow Pkts/s",
    "sloss":            "Fwd Pkt Len Max",
    "dloss":            "Bwd Pkt Len Max",
    "sinpkt":           "Fwd IAT Mean",
    "dinpkt":           "Bwd IAT Mean",
    "sjit":             "Fwd Pkt Len Mean",
    "djit":             "Bwd Pkt Len Mean",
    "smeansz":          "Pkt Len Mean",
    "dmeansz":          "Pkt Len Max",
    "trans_depth":      "Flow IAT Mean",
    "res_bdy_len":      "Flow IAT Max",
    "Sintpkt":          "Fwd Header Len",
    "Dintpkt":          "Bwd Header Len",
    "tcprtt":           "Fwd Pkts/s",
    "synack":           "Bwd Pkts/s",
    "ackdat":           "Pkt Len Var",
    "is_sm_ips_ports":  "SYN Flag Cnt",
    "ct_flw_http_mthd": "RST Flag Cnt",
    "is_ftp_login":     "ACK Flag Cnt",
    "ct_ftp_cmd":       "Init Fwd Win Byts",
}

# Target canonical columns (must match CIC-IDS2018 output)
CANONICAL_FEATURES = list(UNSWNB15_FEATURE_MAP.values())

# UNSW-NB15 label column
LABEL_COL_UNSW = "attack_cat"   # multi-class; "Normal" is the benign class


def load_raw(input_dir: str) -> pd.DataFrame:
    """Load all CSV files in *input_dir* and concatenate them."""
    pattern = os.path.join(input_dir, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in '{input_dir}'. "
            "Download UNSW-NB15 from https://research.unsw.edu.au/projects/unsw-nb15-dataset"
        )
    logger.info("Found %d CSV file(s) in '%s'.", len(files), input_dir)
    dfs = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip().str.lower()   # normalise column names
    logger.info("Raw shape: %s", df.shape)
    return df


def map_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rename UNSW-NB15 columns to FLEX-ID canonical names."""
    lower_map = {k.lower(): v for k, v in UNSWNB15_FEATURE_MAP.items()}
    df = df.rename(columns=lower_map)
    # Keep only canonical features + label; fill missing with 0
    for col in CANONICAL_FEATURES:
        if col not in df.columns:
            logger.warning("Feature '%s' missing — filling with 0.", col)
            df[col] = 0.0
    # Normalise label column name
    if "attack_cat" in df.columns:
        df = df.rename(columns={"attack_cat": "label"})
    elif "label" in df.columns:
        # Binary label → convert 0/1 to "Normal"/"Attack"
        df["label"] = df["label"].apply(lambda x: "Normal" if int(x) == 0 else "Attack")
    return df


def preprocess(df: pd.DataFrame, output_dir: str) -> None:
    """Clean, scale, encode, and save the processed dataset."""
    label_col = "label"

    # 1. Strip whitespace from string labels
    df[label_col] = df[label_col].astype(str).str.strip()
    df[label_col] = df[label_col].replace({"Normal": "Benign", "": "Benign"})

    # 2. Numeric conversion & cleaning
    feature_cols = CANONICAL_FEATURES
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=feature_cols, inplace=True)
    df = df[np.isfinite(df[feature_cols]).all(axis=1)]
    logger.info("Shape after cleaning: %s", df.shape)

    # 3. Scale
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df[feature_cols].values)
    df_out = pd.DataFrame(X_scaled, columns=feature_cols)

    # 4. Encode labels
    le = LabelEncoder()
    df_out["label"] = le.fit_transform(df[label_col].values)
    logger.info("Classes: %s", list(le.classes_))

    # 5. Save
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "processed_data.csv")
    le_path  = os.path.join(output_dir, "label_encoder.pkl")
    df_out.to_csv(csv_path, index=False)
    with open(le_path, "wb") as fh:
        pickle.dump(le, fh)

    logger.info("Saved: %s", csv_path)
    logger.info("Saved: %s", le_path)
    logger.info("Class distribution:\n%s", df_out["label"].value_counts())


def main() -> None:
    parser = argparse.ArgumentParser(description="UNSW-NB15 Preprocessing for FLEX-ID")
    parser.add_argument("--input_dir",  default="data_unswnb15/raw",  help="Directory with raw UNSW-NB15 CSVs")
    parser.add_argument("--output_dir", default="data_unswnb15",       help="Output directory")
    args = parser.parse_args()

    t0 = time.time()
    df = load_raw(args.input_dir)
    df = map_features(df)
    preprocess(df, args.output_dir)
    logger.info("Done in %.1f s.", time.time() - t0)


if __name__ == "__main__":
    main()
