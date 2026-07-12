"""
plot_history.py — Training history visualisation for FLEX-ID.

Loads the pickled history produced by 4_server.py and generates:
  - Loss / Accuracy / F1 learning curves (normal and under-attack)
  - A CSV table of round-wise metrics (for paper tables)
  - A JSON summary with best / last values
  - Communication cost estimate (bytes transferred per round)

Usage
-----
    python plot_history.py
"""

from __future__ import annotations

import csv
import json
import logging
import os
import pickle
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("flex_id.plot_history")

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── History file paths ─────────────────────────────────────────────────────────
FEDAVG_HIST        = "results/fedavg_history.pkl"
FEDPROX_HIST       = "results/fedprox_history.pkl"
FEDAVG_ATTACK_HIST = "results/fedavg_underattack_history.pkl"
FEDPROX_ATTACK_HIST= "results/fedprox_underattack_history.pkl"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_history(
    filepath: str,
) -> Tuple[List[int], List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    """Load a history pickle and extract (rounds, losses, accuracies, f1s)."""
    if not os.path.exists(filepath):
        return [], [], [], []

    with open(filepath, "rb") as fh:
        history: List[Dict[str, Any]] = pickle.load(fh)

    rounds, losses, accuracies, f1s = [], [], [], []
    for entry in history:
        rounds.append(entry.get("round"))
        losses.append(entry.get("train_loss"))
        accuracies.append(entry.get("accuracy"))
        f1s.append(entry.get("f1"))

    return rounds, losses, accuracies, f1s


def extract_stats(
    rounds: List[int],
    losses: List[Optional[float]],
    accuracies: List[Optional[float]],
    f1s: List[Optional[float]],
) -> Optional[Dict[str, Any]]:
    """Return summary statistics or None if no data."""
    if not rounds:
        return None

    def safe_max(lst):
        valid = [v for v in lst if v is not None]
        return max(valid) if valid else None

    def safe_last(lst):
        for v in reversed(lst):
            if v is not None:
                return v
        return None

    return {
        "rounds": len(rounds),
        "best_accuracy_pct": f"{safe_max(accuracies)*100:.2f}%" if safe_max(accuracies) else "N/A",
        "best_f1": f"{safe_max(f1s):.4f}" if safe_max(f1s) else "N/A",
        "last_accuracy_pct": f"{safe_last(accuracies)*100:.2f}%" if safe_last(accuracies) else "N/A",
        "last_f1": f"{safe_last(f1s):.4f}" if safe_last(f1s) else "N/A",
        "last_loss": f"{safe_last(losses):.4f}" if safe_last(losses) else "N/A",
    }


def estimate_communication_cost(weights_path: str) -> Optional[float]:
    """Estimate bytes transmitted per round from a saved weight pickle."""
    if not os.path.exists(weights_path):
        return None
    with open(weights_path, "rb") as fh:
        weights = pickle.load(fh)
    if isinstance(weights, list):
        total_bytes = sum(w.nbytes for w in weights if hasattr(w, "nbytes"))
    else:
        total_bytes = 0
    return float(total_bytes)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_learning_curves(
    r1: List[int], l1: List, a1: List, f1: List,
    r2: List[int], l2: List, a2: List, f2: List,
    label1: str, label2: str,
    save_name: str,
    title_suffix: str = "",
) -> None:
    """Plot training loss, accuracy, and F1 for two strategies side-by-side."""
    if not r1 and not r2:
        logger.warning("No data to plot for '%s'. Skipping.", save_name)
        return

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"1": "#2563EB", "2": "#DC2626"}   # blue / red

    for ax, metric_name, y1, y2 in zip(
        axs,
        [f"Training Loss {title_suffix}",
         f"Global Accuracy {title_suffix}",
         f"Global F1 Score {title_suffix}"],
        [l1, a1, f1],
        [l2, a2, f2],
    ):
        if r1 and any(v is not None for v in y1):
            ax.plot(r1, y1, label=label1, color=colors["1"], marker="o", linewidth=1.8)
        if r2 and any(v is not None for v in y2):
            ax.plot(r2, y2, label=label2, color=colors["2"],
                    linestyle="--", marker="x", linewidth=1.8)
        ax.set_title(metric_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Communication Round", fontsize=11)
        ax.set_ylabel("Value", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.35)

    plt.tight_layout()
    save_path = os.path.join("results", save_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Learning curve saved -> %s", save_path)


def save_round_csv(
    rounds: List[int],
    losses: List, accuracies: List, f1s: List,
    label: str,
    filepath: str,
) -> None:
    """Save round-wise metrics to a CSV file (useful for paper tables)."""
    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["round", "train_loss", "accuracy", "f1", "strategy"])
        for r, lo, ac, fi in zip(rounds, losses, accuracies, f1s):
            writer.writerow([r, lo, ac, fi, label])
    logger.info("Round-wise CSV saved -> %s", filepath)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def plot_comparison() -> None:
    os.makedirs("results", exist_ok=True)
    logger.info("Loading history files ...")

    # ── 1. Normal (no attack) ─────────────────────────────────────────────────
    r_avg, l_avg, a_avg, f_avg    = load_history(FEDAVG_HIST)
    r_prox, l_prox, a_prox, f_prox = load_history(FEDPROX_HIST)

    if r_avg or r_prox:
        plot_learning_curves(
            r_avg, l_avg, a_avg, f_avg,
            r_prox, l_prox, a_prox, f_prox,
            "FedAvg", "FedProx",
            "comparison_metrics.png",
        )
        # CSVs for paper tables
        if r_avg:
            save_round_csv(r_avg, l_avg, a_avg, f_avg, "FedAvg",
                           "results/round_metrics_fedavg.csv")
        if r_prox:
            save_round_csv(r_prox, l_prox, a_prox, f_prox, "FedProx",
                           "results/round_metrics_fedprox.csv")
    else:
        logger.warning("No normal-run history found.")

    # ── 2. Under attack ───────────────────────────────────────────────────────
    r_avg_att, l_avg_att, a_avg_att, f_avg_att     = load_history(FEDAVG_ATTACK_HIST)
    r_prox_att, l_prox_att, a_prox_att, f_prox_att = load_history(FEDPROX_ATTACK_HIST)

    if r_avg_att or r_prox_att:
        plot_learning_curves(
            r_avg_att, l_avg_att, a_avg_att, f_avg_att,
            r_prox_att, l_prox_att, a_prox_att, f_prox_att,
            "FedAvg (Attack)", "FedProx (Attack)",
            "comparison_metrics_underattack.png",
            title_suffix="(Under Attack)",
        )
    else:
        logger.info("No under-attack history found. Skipping attack plot.")

    # ── 3. Communication cost estimate ────────────────────────────────────────
    comm_cost_bytes = estimate_communication_cost(
        "results/fedavgeachround/round-1-weights.pkl"
    )
    if comm_cost_bytes:
        logger.info(
            "Estimated communication cost per client per round: %.2f KB",
            comm_cost_bytes / 1024,
        )

    # ── 4. Summary JSON ───────────────────────────────────────────────────────
    summary = {
        "random_seed":          RANDOM_SEED,
        "fedavg":               extract_stats(r_avg, l_avg, a_avg, f_avg),
        "fedprox":              extract_stats(r_prox, l_prox, a_prox, f_prox),
        "fedavg_under_attack":  extract_stats(r_avg_att, l_avg_att, a_avg_att, f_avg_att),
        "fedprox_under_attack": extract_stats(r_prox_att, l_prox_att, a_prox_att, f_prox_att),
        "communication_cost_kb_per_round": round(comm_cost_bytes / 1024, 2) if comm_cost_bytes else None,
    }

    summary_path = "results/metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=4, default=str)
    logger.info("Metrics summary saved -> %s", summary_path)


if __name__ == "__main__":
    plot_comparison()