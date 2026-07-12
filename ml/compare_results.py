"""
compare_results.py — Comprehensive model evaluation for FLEX-ID.

Evaluates saved weight files against the global test set and reports:
  - Accuracy, Balanced Accuracy
  - Weighted F1, Macro F1
  - Per-class Precision, Recall, F1
  - ROC-AUC (one-vs-rest, macro)
  - PR-AUC (average precision, macro)
  - Matthews Correlation Coefficient (MCC)
  - Confusion Matrix (normalised)
  - Inference Time

Usage
-----
    python compare_results.py \\
        --fedavg  results/fedavgeachround/round-30-weights.pkl \\
        --fedprox results/fedproxeachround/round-30-weights.pkl \\
        --mode    no_attack
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("flex_id.compare_results")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42

try:
    from utils.seeds import set_global_seeds
    set_global_seeds(RANDOM_SEED)
except ImportError:
    import random
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

try:
    from model import create_dnn_model
except ImportError:
    logger.error("model.py not found.")
    sys.exit(1)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH = "data/processed_data.csv"
LE_PATH   = "data/label_encoder.pkl"


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_and_process_data(
    data_path: str = DATA_PATH,
    le_path: str = LE_PATH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load data, apply saved label encoder, return (X_test, y_test, class_names)."""
    logger.info("Loading data from '%s' ...", data_path)

    if not os.path.exists(data_path):
        logger.error("Data file not found: '%s'.", data_path)
        sys.exit(1)

    df = pd.read_csv(data_path)

    # Identify label column
    string_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if not string_cols:
        # Labels may already be integers
        label_col = "label" if "label" in df.columns else df.columns[-1]
    else:
        label_col = string_cols[0]
    logger.info("Label column: '%s'.", label_col)

    # Load encoder
    if os.path.exists(le_path):
        with open(le_path, "rb") as fh:
            le: LabelEncoder = pickle.load(fh)
        try:
            df[label_col] = le.transform(df[label_col])
        except ValueError:
            valid = set(le.classes_)
            df = df[df[label_col].isin(valid)]
            df[label_col] = le.transform(df[label_col])
    else:
        logger.warning("label_encoder.pkl not found. Creating a new one (risky!).")
        le = LabelEncoder()
        df[label_col] = le.fit_transform(df[label_col])

    class_names = le.classes_
    logger.info("Classes: %s", list(class_names))

    y = df[label_col].values.astype(int)
    X = df.drop(columns=[label_col]).values.astype(np.float32)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    return X_test, y_test, class_names


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_weights(
    weights_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: np.ndarray,
    algorithm_name: str,
    suffix: str = "",
) -> Dict[str, Any]:
    """Evaluate a saved weight file with the full metric suite."""
    logger.info("--- Evaluating %s ---", algorithm_name)

    if not os.path.exists(weights_path):
        logger.warning("Weight file not found: '%s'. Skipping.", weights_path)
        return {"skipped": True}

    try:
        with open(weights_path, "rb") as fh:
            weights = pickle.load(fh)

        model = create_dnn_model(
            input_shape=X_test.shape[1],
            num_classes=len(class_names),
        )
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.set_weights(weights)

        # ── Basic loss / accuracy ──────────────────────────────────────────────
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

        # ── Inference time ─────────────────────────────────────────────────────
        t0 = time.perf_counter()
        y_pred_probs = model.predict(X_test, verbose=0)
        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        y_pred = np.argmax(y_pred_probs, axis=1)

        # ── Classification report (per-class P/R/F1) ──────────────────────────
        report_dict = classification_report(
            y_test, y_pred,
            target_names=[str(c) for c in class_names],
            zero_division=0,
            output_dict=True,
        )
        logger.info("\n%s", classification_report(
            y_test, y_pred,
            target_names=[str(c) for c in class_names],
            zero_division=0,
        ))

        # ── Balanced accuracy ─────────────────────────────────────────────────
        balanced_acc = balanced_accuracy_score(y_test, y_pred)

        # ── MCC ──────────────────────────────────────────────────────────────
        mcc = matthews_corrcoef(y_test, y_pred)

        # ── ROC-AUC (one-vs-rest, macro) ─────────────────────────────────────
        n_classes = len(class_names)
        try:
            if n_classes == 2:
                roc_auc = roc_auc_score(y_test, y_pred_probs[:, 1])
            else:
                y_bin = label_binarize(y_test, classes=list(range(n_classes)))
                roc_auc = roc_auc_score(y_bin, y_pred_probs, average="macro", multi_class="ovr")
        except Exception as exc:
            logger.warning("ROC-AUC computation failed: %s", exc)
            roc_auc = None

        # ── PR-AUC (average precision, macro) ────────────────────────────────
        try:
            if n_classes == 2:
                pr_auc = average_precision_score(y_test, y_pred_probs[:, 1])
            else:
                y_bin = label_binarize(y_test, classes=list(range(n_classes)))
                pr_auc = average_precision_score(y_bin, y_pred_probs, average="macro")
        except Exception as exc:
            logger.warning("PR-AUC computation failed: %s", exc)
            pr_auc = None

        # ── Confusion matrix plot ─────────────────────────────────────────────
        _plot_confusion_matrix(
            y_test, y_pred, class_names,
            title=algorithm_name, suffix=suffix,
        )

        result: Dict[str, Any] = {
            "algorithm": algorithm_name,
            "weights_path": weights_path,
            "accuracy":           round(float(accuracy * 100), 4),
            "balanced_accuracy":  round(float(balanced_acc * 100), 4),
            "loss":               round(float(loss), 6),
            "macro_f1":           round(float(report_dict["macro avg"]["f1-score"]), 6),
            "weighted_f1":        round(float(report_dict["weighted avg"]["f1-score"]), 6),
            "mcc":                round(float(mcc), 6),
            "roc_auc_macro":      round(float(roc_auc), 6) if roc_auc is not None else None,
            "pr_auc_macro":       round(float(pr_auc), 6) if pr_auc is not None else None,
            "inference_time_ms":  round(inference_time_ms, 2),
            "num_test_samples":   int(len(y_test)),
            "per_class_report":   report_dict,
        }

        logger.info(
            "[%s] Acc=%.2f%% | Balanced=%.2f%% | MCC=%.4f | ROC-AUC=%s | PR-AUC=%s | "
            "Inference=%.1f ms",
            algorithm_name, result["accuracy"], result["balanced_accuracy"],
            result["mcc"],
            f"{result['roc_auc_macro']:.4f}" if result["roc_auc_macro"] else "N/A",
            f"{result['pr_auc_macro']:.4f}" if result["pr_auc_macro"] else "N/A",
            result["inference_time_ms"],
        )
        return result

    except Exception as exc:
        logger.error("Evaluation failed for %s: %s", algorithm_name, exc, exc_info=True)
        return {"error": str(exc), "algorithm": algorithm_name}


def _plot_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    class_names: np.ndarray,
    title: str,
    suffix: str,
) -> None:
    """Save a normalised confusion matrix heatmap."""
    os.makedirs("results", exist_ok=True)
    cm = confusion_matrix(y_test, y_pred, normalize="true")
    plt.figure(figsize=(max(8, len(class_names)), max(6, len(class_names) - 1)))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.title(f"{title} — Confusion Matrix ({suffix})", fontsize=13)
    plt.tight_layout()
    filename = f"results/confusion_matrix_{title.lower().replace(' ', '_')}_{suffix}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    logger.info("Confusion matrix saved -> %s", filename)


# ──────────────────────────────────────────────────────────────────────────────
# Comparison bar chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_metric_comparison(results: Dict[str, Dict], suffix: str) -> None:
    """Bar chart comparing key metrics across evaluated models."""
    models = [r["algorithm"] for r in results.values() if "algorithm" in r]
    metrics = {
        "Accuracy (%)":         [r.get("accuracy", 0)          for r in results.values() if "algorithm" in r],
        "Macro F1":             [r.get("macro_f1", 0)           for r in results.values() if "algorithm" in r],
        "Balanced Acc (%)":     [r.get("balanced_accuracy", 0)  for r in results.values() if "algorithm" in r],
        "MCC":                  [r.get("mcc", 0)                for r in results.values() if "algorithm" in r],
    }
    if not models:
        return

    x = np.arange(len(models))
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 3), 6))

    for idx, (metric_name, values) in enumerate(metrics.items()):
        ax.bar(x + idx * width, values, width, label=metric_name)

    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(models)
    ax.set_ylabel("Score")
    ax.set_title(f"FLEX-ID Model Comparison ({suffix})")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    path = f"results/comparison_metrics_{suffix}.png"
    plt.savefig(path, dpi=300)
    plt.close()
    logger.info("Comparison chart saved -> %s", path)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of FLEX-ID federated models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fedavg",   type=str, required=True, help="Path to FedAvg weights.")
    parser.add_argument("--fedprox",  type=str, required=True, help="Path to FedProx weights.")
    parser.add_argument("--mode",     type=str, default="custom",
                        help="Mode label (e.g., no_attack, under_attack).")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory with processed_data.csv and label_encoder.pkl.")
    args = parser.parse_args()

    suffix   = args.mode.replace(" ", "_").lower()
    data_path = os.path.join(args.data_dir, "processed_data.csv")
    le_path   = os.path.join(args.data_dir, "label_encoder.pkl")

    os.makedirs("results", exist_ok=True)

    X_test, y_test, class_names = load_and_process_data(data_path, le_path)

    results: Dict[str, Any] = {}

    results["fedavg"] = evaluate_weights(
        args.fedavg, X_test, y_test, class_names, "FedAvg", suffix
    )
    results["fedprox"] = evaluate_weights(
        args.fedprox, X_test, y_test, class_names, "FedProx", suffix
    )

    results["meta"] = {
        "success": True,
        "timestamp": pd.Timestamp.now().isoformat(),
        "mode": args.mode,
        "random_seed": RANDOM_SEED,
    }

    # Comparison plot
    plot_metric_comparison(results, suffix)

    # Save JSON
    output_path = f"results/comparison_results_{suffix}.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=4, default=str)
    logger.info("Results saved -> %s", output_path)


if __name__ == "__main__":
    main()