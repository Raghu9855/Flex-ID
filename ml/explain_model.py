"""
explain_model.py — Federated SHAP Explainability for FLEX-ID.

Implements a privacy-preserving federated SHAP framework:
  1. Each client computes SHAP values locally on its own test data.
  2. Only the *aggregated importance vector* (mean |SHAP|) is shared with
     the server — raw per-sample SHAP values never leave the client.
  3. The server aggregates importance vectors via a simple mean (federated
     aggregation) to produce a global feature importance ranking.
  4. Inter-client agreement is quantified using Kendall's Tau and Spearman's
     rank correlation, providing an *explanation stability score*.

Privacy guarantee: transmitting mean(|SHAP|) vectors instead of raw SHAP
  matrices prevents membership inference attacks based on SHAP values.

References
----------
Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting
    model predictions. NeurIPS, 30.

Usage
-----
    python explain_model.py                 # explain both FedAvg and FedProx
    python explain_model.py --round 15      # specific training round
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import kendalltau, spearmanr

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("flex_id.explain")

try:
    from model import create_dnn_model
except ImportError:
    logger.error("model.py not found. Place it in the ml/ directory.")
    sys.exit(1)

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ──────────────────────────────────────────────────────────────────────────────
# Federated SHAP Engine
# ──────────────────────────────────────────────────────────────────────────────

class FederatedSHAP:
    """Privacy-preserving federated SHAP explanation engine.

    Parameters
    ----------
    weights_path : str
        Path to the saved model weight pickle.
    data_path : str
        Path to the processed feature CSV (used for feature names and shape).
    le_path : str
        Path to the LabelEncoder pickle.
    num_clients : int
        Number of federated clients.
    data_dir : str
        Directory containing client partition pickles.
    bg_size : int
        Number of background samples used by KernelSHAP.  Documented for
        reproducibility: default = 100.
    explain_size : int
        Number of test samples explained per client.  Default = 50.
    """

    def __init__(
        self,
        weights_path: str,
        data_path: str,
        le_path: str,
        num_clients: int = 4,
        data_dir: str = "data",
        bg_size: int = 100,
        explain_size: int = 50,
    ) -> None:
        self.weights_path = weights_path
        self.data_path = data_path
        self.le_path = le_path
        self.num_clients = num_clients
        self.data_dir = data_dir
        self.bg_size = bg_size
        self.explain_size = explain_size

        self.model = self._load_model()
        self.feature_names: List[str] = self._load_feature_names()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_model(self):
        """Load DNN model and restore weights."""
        logger.info("Loading model from %s", self.weights_path)

        with open(self.le_path, "rb") as fh:
            le = pickle.load(fh)
        self.le = le

        df = pd.read_csv(self.data_path, nrows=5)
        target_col = next(
            c for c in df.columns if c.lower() in ("label", "class", "attack_cat")
        )
        x_dim = df.drop(columns=[target_col]).shape[1]

        model = create_dnn_model(x_dim, len(le.classes_))
        with open(self.weights_path, "rb") as fh:
            weights = pickle.load(fh)
        model.set_weights(weights)
        return model

    def _load_feature_names(self) -> List[str]:
        """Return ordered list of feature column names."""
        df = pd.read_csv(self.data_path, nrows=1)
        return [
            c for c in df.columns
            if c.lower() not in ("label", "class", "attack_cat")
        ]

    def _load_client_data(self, cid: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load a client's test set."""
        path = os.path.join(self.data_dir, f"client_partition_{cid}.pkl")
        with open(path, "rb") as fh:
            (_, _), (X_test, y_test) = pickle.load(fh)
        return X_test.astype(np.float32), y_test

    def _select_attack_shap(self, shap_values) -> np.ndarray:
        """Pick the correct SHAP slice for attack classes.

        KernelExplainer returns one array per output class.  Instead of
        always picking class index 1, we average over *all* attack classes
        (every class that is not 'Benign'), giving a more representative
        picture of feature importance for malicious traffic.
        """
        attack_class_indices = [
            i for i, c in enumerate(self.le.classes_)
            if c.lower() != "benign"
        ]

        if isinstance(shap_values, list):
            # List of (n_samples, n_features) — one per class
            if attack_class_indices:
                parts = [shap_values[i] for i in attack_class_indices]
                return np.mean(parts, axis=0)
            return shap_values[1] if len(shap_values) > 1 else shap_values[0]

        # 3-D ndarray (n_samples, n_features, n_classes)
        if shap_values.ndim == 3:
            if attack_class_indices:
                return np.mean(shap_values[:, :, attack_class_indices], axis=2)
            return shap_values[:, :, 1]

        # Already 2-D
        return shap_values

    # ── Public interface ──────────────────────────────────────────────────────

    def explain_client(
        self,
        cid: int,
        prefix: str = "",
    ) -> np.ndarray:
        """Run KernelSHAP on client *cid*'s test data.

        Privacy note
        ------------
        Only the *mean absolute SHAP* vector (shape: n_features,) is
        returned and shared with the server.  The full per-sample SHAP
        matrix stays local to this function and is never persisted.

        Parameters
        ----------
        cid : int
            Client index.
        prefix : str
            Label prefix for saved plot file names.

        Returns
        -------
        np.ndarray, shape (n_features,)
            Mean absolute SHAP importance vector (privacy-preserving).
        """
        logger.info("[Client %d] Running KernelSHAP (bg=%d, explain=%d) ...",
                    cid, self.bg_size, self.explain_size)

        X_test, _ = self._load_client_data(cid)

        # Sample background and explanation sets
        rng = np.random.default_rng(RANDOM_SEED + cid)
        bg_idx = rng.choice(len(X_test), min(self.bg_size, len(X_test)), replace=False)
        bg = X_test[bg_idx]
        samples = X_test[: min(self.explain_size, len(X_test))]

        def predict_fn(x: np.ndarray) -> np.ndarray:
            return self.model.predict(x, verbose=0)

        explainer = shap.KernelExplainer(predict_fn, bg)
        raw_shap = explainer.shap_values(samples, silent=True)
        shap_vals = self._select_attack_shap(raw_shap)  # (n_samples, n_features)

        # ── Local SHAP plot ───────────────────────────────────────────────────
        os.makedirs("results", exist_ok=True)
        out_path = f"results/{prefix}_client_{cid}_shap.png"
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, samples, feature_names=self.feature_names, show=False)
        plt.title(f"SHAP Feature Importance — Client {cid} ({prefix})")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("[Client %d] Local SHAP plot saved -> %s", cid, out_path)

        # ── Return privacy-preserving importance vector ────────────────────────
        importance_vector = np.mean(np.abs(shap_vals), axis=0)
        return importance_vector

    def aggregate_global(
        self,
        client_vectors: List[np.ndarray],
        prefix: str = "",
    ) -> np.ndarray:
        """Aggregate per-client importance vectors into a global ranking.

        Parameters
        ----------
        client_vectors : list of np.ndarray
            Each element is a (n_features,) importance vector from one client.
        prefix : str
            Label prefix for the saved plot file name.

        Returns
        -------
        np.ndarray, shape (n_features,)
            Normalised global importance vector (0-100 scale).
        """
        logger.info("[Server] Aggregating federated SHAP values from %d clients ...",
                    len(client_vectors))

        stacked = np.stack(client_vectors, axis=0)   # (n_clients, n_features)
        global_imp = np.mean(stacked, axis=0)         # simple federated mean
        global_imp = global_imp.flatten()

        # Normalise to percentage scale
        max_val = np.max(global_imp)
        if max_val > 0:
            global_imp = 100.0 * global_imp / max_val

        # Top-15 features
        idx = np.argsort(global_imp)[::-1][:15]
        labels = [self.feature_names[int(i)] for i in idx]
        values = global_imp[idx]

        # ── Global SHAP bar plot ──────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 7))
        bars = ax.barh(range(len(values)), values, color="#2563EB", alpha=0.85)
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels(labels, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel("Aggregated SHAP Importance (%)", fontsize=12)
        ax.set_title(
            f"Federated Global Feature Importance — {prefix}\n"
            f"(Privacy-preserving aggregation, n={len(client_vectors)} clients)",
            fontsize=13, fontweight="bold",
        )
        ax.axvline(x=0, color="black", linewidth=0.8)
        plt.tight_layout()

        out_path = f"results/{prefix}_global_shap.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("[Server] Global SHAP plot saved -> %s", out_path)

        return global_imp

    def compute_explanation_agreement(
        self,
        client_vectors: List[np.ndarray],
        top_k: int = 10,
    ) -> dict:
        """Compute inter-client explanation agreement metrics.

        Metrics
        -------
        kendall_tau_mean  : Mean Kendall's Tau across all client pairs.
        spearman_rho_mean : Mean Spearman's rho across all client pairs.
        stability_score   : Mean Kendall's Tau re-scaled to [0, 1].
        jaccard_top_k     : Mean Jaccard similarity of top-K feature sets.

        Parameters
        ----------
        client_vectors : list of np.ndarray
            Per-client importance vectors.
        top_k : int
            Number of top features to consider for Jaccard similarity.
        """
        n = len(client_vectors)
        if n < 2:
            return {"error": "Need >= 2 clients for agreement metrics."}

        taus, rhos, jaccards = [], [], []

        for i in range(n):
            for j in range(i + 1, n):
                vi, vj = client_vectors[i], client_vectors[j]

                # Kendall's Tau
                tau, _ = kendalltau(vi, vj)
                taus.append(tau)

                # Spearman's rho
                rho, _ = spearmanr(vi, vj)
                rhos.append(rho)

                # Jaccard on top-K
                top_i = set(np.argsort(vi)[::-1][:top_k])
                top_j = set(np.argsort(vj)[::-1][:top_k])
                jaccard = len(top_i & top_j) / len(top_i | top_j)
                jaccards.append(jaccard)

        results = {
            "num_clients": n,
            "top_k": top_k,
            "kendall_tau_mean": float(np.mean(taus)),
            "kendall_tau_std": float(np.std(taus)),
            "spearman_rho_mean": float(np.mean(rhos)),
            "spearman_rho_std": float(np.std(rhos)),
            # Stability: rescale Tau from [-1,1] to [0,1]
            "stability_score": float((np.mean(taus) + 1.0) / 2.0),
            "jaccard_top_k_mean": float(np.mean(jaccards)),
        }

        logger.info(
            "[XAI] Agreement — Kendall Tau=%.3f | Spearman rho=%.3f | "
            "Stability=%.3f | Jaccard@%d=%.3f",
            results["kendall_tau_mean"],
            results["spearman_rho_mean"],
            results["stability_score"],
            top_k,
            results["jaccard_top_k_mean"],
        )
        return results


# ──────────────────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────────────────

def run(
    round_num: int = 10,
    num_clients: int = 4,
    data_dir: str = "data",
    bg_size: int = 100,
    explain_size: int = 50,
) -> None:
    """Run federated SHAP explanation for FedAvg and FedProx."""
    import json

    models = {
        "FedAvg":  f"results/fedavgeachround/round-{round_num}-weights.pkl",
        "FedProx": f"results/fedproxeachround/round-{round_num}-weights.pkl",
    }

    all_results: dict = {}

    for name, path in models.items():
        if not os.path.exists(path):
            logger.warning("Weights not found for %s at '%s'. Skipping.", name, path)
            continue

        logger.info("\n==== Explaining %s (round %d) ====", name, round_num)

        explainer = FederatedSHAP(
            weights_path=path,
            data_path=os.path.join(data_dir, "processed_data.csv"),
            le_path=os.path.join(data_dir, "label_encoder.pkl"),
            num_clients=num_clients,
            data_dir=data_dir,
            bg_size=bg_size,
            explain_size=explain_size,
        )

        # Per-client local explanations (privacy-preserving)
        client_vectors: List[np.ndarray] = []
        for cid in range(num_clients):
            vec = explainer.explain_client(cid, prefix=name)
            client_vectors.append(vec)

        # Server-side global aggregation
        global_imp = explainer.aggregate_global(client_vectors, prefix=name)

        # Explanation agreement metrics
        agreement = explainer.compute_explanation_agreement(client_vectors, top_k=10)

        # Top features summary
        idx = np.argsort(global_imp)[::-1][:10]
        top_features = [
            {"rank": r + 1, "feature": explainer.feature_names[int(i)],
             "importance_pct": round(float(global_imp[i]), 2)}
            for r, i in enumerate(idx)
        ]

        all_results[name] = {
            "round": round_num,
            "num_clients": num_clients,
            "bg_size": bg_size,
            "explain_size": explain_size,
            "top_10_features": top_features,
            "agreement_metrics": agreement,
        }

    # Save JSON summary
    summary_path = "results/shap_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=4)
    logger.info("SHAP summary saved -> %s", summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated SHAP Explainability for FLEX-ID")
    parser.add_argument("--round",        type=int, default=10,  help="Training round to explain.")
    parser.add_argument("--num_clients",  type=int, default=4,   help="Number of clients.")
    parser.add_argument("--data_dir",     type=str, default="data", help="Data directory.")
    parser.add_argument("--bg_size",      type=int, default=100, help="KernelSHAP background samples.")
    parser.add_argument("--explain_size", type=int, default=50,  help="Samples to explain per client.")
    args = parser.parse_args()

    run(
        round_num=args.round,
        num_clients=args.num_clients,
        data_dir=args.data_dir,
        bg_size=args.bg_size,
        explain_size=args.explain_size,
    )


if __name__ == "__main__":
    main()
