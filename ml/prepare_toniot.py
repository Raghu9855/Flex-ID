"""
prepare_toniot.py — Preprocessing pipeline for the TON-IoT dataset.

Expects raw TON-IoT CSV files in ``data_toniot/raw/`` and produces:
  - ``data_toniot/processed_data.csv``
  - ``data_toniot/label_encoder.pkl``

Outputs follow the same 28-feature canonical schema as CIC-IDS2018 so
the rest of the FLEX-ID pipeline works without modification by passing
``--data_dir data_toniot`` to ``2_create_partitions.py`` and the server.

TON-IoT Reference
-----------------
Alsaedi, A., Moustafa, N., Tari, Z., Mahmood, A., & Anwar, A. (2020).
    TON_IoT Telemetry Dataset: A New Generation Dataset of IoT and IIoT
    for Data-Driven Intrusion Detection Systems.
    IEEE Access, 8, 165130–165150.
    https://research.unsw.edu.au/projects/toniot-datasets

Usage
-----
    # Place raw CSVs in data_toniot/raw/ then run:
    python prepare_toniot.py

    # Or specify paths explicitly:
    python prepare_toniot.py --input_dir data_toniot/raw --output_dir data_toniot
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
logger = logging.getLogger("flex_id.prepare_toniot")


# ──────────────────────────────────────────────────────────────────────────────
# Feature mapping: TON-IoT → FLEX-ID canonical
# ──────────────────────────────────────────────────────────────────────────────
TONIOT_FEATURE_MAP: dict[str, str] = {
    # TON-IoT network-level features → canonical names
    "ts":               "Flow Duration",
    "src_port":         "Dst Port",
    "dst_port":         "Dst Port",       # overrides if both exist
    "proto":            "Protocol",
    "duration":         "Flow Duration",
    "src_bytes":        "TotLen Fwd Pkts",
    "dst_bytes":        "TotLen Bwd Pkts",
    "src_pkts":         "Tot Fwd Pkts",
    "dst_pkts":         "Tot Bwd Pkts",
    "src_ip_bytes":     "Fwd Pkt Len Mean",
    "dst_ip_bytes":     "Bwd Pkt Len Mean",
    "missed_bytes":     "Pkt Len Var",
    "src_pk_rate":      "Fwd Pkts/s",
    "dst_pk_rate":      "Bwd Pkts/s",
    "src_by_rate":      "Flow Byts/s",
    "dst_by_rate":      "Flow Pkts/s",
    "conn_state":       "SYN Flag Cnt",
    "history":          "ACK Flag Cnt",
    "dns_qclass":       "RST Flag Cnt",
    "dns_qtype":        "Flow IAT Mean",
    "dns_rcode":        "Flow IAT Max",
    "http_version":     "Fwd IAT Mean",
    "http_status_code": "Bwd IAT Mean",
    "http_request_body_len": "Init Fwd Win Byts",
    "http_resp_mime_types": "Init Bwd Win Byts",
    "weird_name":       "Fwd Header Len",
    "weird_addl":       "Bwd Header Len",
    "weird_notice":     "Fwd Pkt Len Max",
}

CANONICAL_FEATURES = [
    "Dst Port", "Protocol", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max", "Fwd Pkt Len Mean",
    "Bwd Pkt Len Max", "Bwd Pkt Len Mean", "Flow Byts/s", "Flow Pkts/s",
    "Flow IAT Mean", "Flow IAT Max", "Fwd IAT Mean", "Bwd IAT Mean",
    "Fwd Header Len", "Bwd Header Len", "Fwd Pkts/s", "Bwd Pkts/s",
    "Pkt Len Mean", "Pkt Len Max", "Pkt Len Var", "SYN Flag Cnt",
    "RST Flag Cnt", "ACK Flag Cnt", "Init Fwd Win Byts", "Init Bwd Win Byts",
]

# TON-IoT label column and benign label
LABEL_COL_TONIOT = "type"    # "normal" is benign; others are attack categories


def load_raw(input_dir: str) -> pd.DataFrame:
    """Load all CSV files in *input_dir* and concatenate them."""
    pattern = os.path.join(input_dir, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in '{input_dir}'. "
            "Download TON-IoT from https://research.unsw.edu.au/projects/toniot-datasets"
        )
    logger.info("Found %d CSV file(s) in '%s'.", len(files), input_dir)
    dfs = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip().str.lower()
    logger.info("Raw shape: %s", df.shape)
    return df


def map_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rename TON-IoT columns to FLEX-ID canonical names."""
    lower_map = {k.lower(): v for k, v in TONIOT_FEATURE_MAP.items()}
    df = df.rename(columns=lower_map)

    # Resolve label column
    if LABEL_COL_TONIOT in df.columns:
        df = df.rename(columns={LABEL_COL_TONIOT: "label"})
    elif "label" not in df.columns:
        # Binary fallback: column named 'label' with 0/1
        if "label" in df.columns:
            df["label"] = df["label"].apply(lambda x: "Benign" if int(x) == 0 else "Attack")
        else:
            raise ValueError("Cannot find a label column in TON-IoT data.")

    # Normalise benign label
    df["label"] = df["label"].astype(str).str.strip()
    df["label"] = df["label"].replace({"normal": "Benign", "Normal": "Benign", "": "Benign"})

    # Fill missing canonical features with 0
    for col in CANONICAL_FEATURES:
        if col not in df.columns:
            logger.warning("Feature '%s' missing — filling with 0.", col)
            df[col] = 0.0

    return df


def preprocess(df: pd.DataFrame, output_dir: str) -> None:
    """Clean, scale, encode, and save the processed dataset."""
    feature_cols = CANONICAL_FEATURES

    # 1. Numeric conversion & cleaning
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=feature_cols, inplace=True)
    df = df[np.isfinite(df[feature_cols]).all(axis=1)]
    logger.info("Shape after cleaning: %s", df.shape)

    # 2. Scale
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df[feature_cols].values)
    df_out = pd.DataFrame(X_scaled, columns=feature_cols)

    # 3. Encode labels
    le = LabelEncoder()
    df_out["label"] = le.fit_transform(df["label"].values)
    logger.info("Classes: %s", list(le.classes_))

    # 4. Save
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
    parser = argparse.ArgumentParser(description="TON-IoT Preprocessing for FLEX-ID")
    parser.add_argument("--input_dir",  default="data_toniot/raw",  help="Directory with raw TON-IoT CSVs")
    parser.add_argument("--output_dir", default="data_toniot",       help="Output directory")
    args = parser.parse_args()

    t0 = time.time()
    df = load_raw(args.input_dir)
    df = map_features(df)
    preprocess(df, args.output_dir)
    logger.info("Done in %.1f s.", time.time() - t0)


if __name__ == "__main__":
    main()
