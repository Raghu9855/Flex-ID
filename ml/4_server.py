"""
4_server.py — FLEX-ID Federated Learning Server.

Supports six aggregation strategies selectable at runtime:
  fedavg       — Federated Averaging (McMahan et al., 2017)
  fedprox      — Proximal-term regularisation (Li et al., 2020)
  krum         — Byzantine-robust single-best selection (Blanchard et al., 2017)
  multikrum    — Byzantine-robust multi-selection (Blanchard et al., 2017)
  trimmed_mean — Coordinate-wise trimmed mean (Yin et al., 2018)
  median       — Coordinate-wise median (Yin et al., 2018)

Usage
-----
    python 4_server.py --strategy fedavg   --rounds 30 --num_clients 4
    python 4_server.py --strategy fedprox  --rounds 30 --num_clients 4 --proximal_mu 0.1
    python 4_server.py --strategy krum     --rounds 30 --num_clients 8 --num_byzantine 1
    python 4_server.py --strategy median   --rounds 30 --num_clients 16
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import pandas as pd

# ── Fix Windows Unicode console output ─────────────────────────────────────────
sys.stdout.reconfigure(encoding="utf-8")

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("flex_id.server")

# ── Local aggregation registry ─────────────────────────────────────────────────
try:
    from aggregation import (
        AGGREGATION_REGISTRY,
        coordinate_median_aggregate,
        krum_aggregate,
        multi_krum_aggregate,
        trimmed_mean_aggregate,
    )
    _CUSTOM_AGG_AVAILABLE = True
except ImportError:
    _CUSTOM_AGG_AVAILABLE = False
    logger.warning(
        "aggregation.py not found.  Only fedavg/fedprox will work correctly."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Port utility
# ──────────────────────────────────────────────────────────────────────────────

def force_release_port(port: int) -> None:
    """Kill any process listening on *port* to prevent binding errors."""
    try:
        if platform.system() == "Windows":
            result = subprocess.check_output(
                f"netstat -ano | findstr :{port}",
                shell=True,
                stderr=subprocess.DEVNULL,
            ).decode()
            for line in result.strip().splitlines():
                if "LISTENING" in line:
                    pid = line.split()[-1]
                    logger.warning(
                        "Found zombie process PID=%s on port %d. Killing it.", pid, port
                    )
                    subprocess.run(
                        f"taskkill /PID {pid} /F",
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
        else:
            subprocess.run(
                f"fuser -k {port}/tcp",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        time.sleep(1)
    except subprocess.CalledProcessError:
        pass   # No process found — nothing to kill
    except Exception as exc:
        logger.warning("Port check failed: %s", exc)


# ──────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_history(obj: Any, filename: str) -> None:
    """Pickle *obj* to *filename*."""
    with open(filename, "wb") as fh:
        pickle.dump(obj, fh)


def save_parameters(params: fl.common.Parameters, filename: str) -> None:
    """Convert Flower Parameters to ndarrays and pickle them."""
    with open(filename, "wb") as fh:
        pickle.dump(fl.common.parameters_to_ndarrays(params), fh)


# ──────────────────────────────────────────────────────────────────────────────
# Metric aggregation
# ──────────────────────────────────────────────────────────────────────────────

def weighted_average(
    metrics: List[Tuple[int, Dict[str, Any]]]
) -> Dict[str, Any]:
    """Weighted average of per-client evaluation metrics."""
    if not metrics:
        return {}
    total_examples = sum(n for n, _ in metrics)
    if total_examples == 0:
        return {}
    acc = sum(n * m.get("accuracy", 0.0) for n, m in metrics) / total_examples
    f1  = sum(n * m.get("f1", 0.0)       for n, m in metrics) / total_examples
    return {"accuracy": acc, "f1": f1}


# ──────────────────────────────────────────────────────────────────────────────
# Custom Flower strategy — dispatches to chosen aggregation function
# ──────────────────────────────────────────────────────────────────────────────

class FlexIDStrategy(fl.server.strategy.FedAvg):
    """Flower strategy that supports multiple robust aggregation algorithms.

    Parameters
    ----------
    aggregation : str
        One of ``fedavg``, ``fedprox``, ``krum``, ``multikrum``,
        ``trimmed_mean``, ``median``.
    save_folder : str
        Directory where per-round weight pickles are stored.
    num_byzantine : int
        Assumed number of Byzantine clients (used by Krum variants).
    trim_ratio : float
        Fraction trimmed from each tail (used by Trimmed Mean).
    krum_m : int | None
        Number of clients selected by Multi-Krum.  Defaults to n - f.
    """

    def __init__(
        self,
        *,
        aggregation: str = "fedavg",
        save_folder: str = "results/fedavgeachround",
        num_byzantine: int = 1,
        trim_ratio: float = 0.1,
        krum_m: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.aggregation = aggregation
        self.save_folder = save_folder
        self.num_byzantine = num_byzantine
        self.trim_ratio = trim_ratio
        self.krum_m = krum_m
        self.round_history: List[Dict[str, Any]] = []

    # ── Flower hook ───────────────────────────────────────────────────────────

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Any],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, Any]]:
        """Aggregate client updates using the chosen strategy."""
        if not results:
            return None, {}

        # ── 1. Collect weights from clients ───────────────────────────────────
        weights_list: List[List[np.ndarray]] = []
        sample_counts: List[int] = []
        losses_weighted: List[float] = []
        loss_examples: List[int] = []

        for _, fit_res in results:
            w = fl.common.parameters_to_ndarrays(fit_res.parameters)
            weights_list.append(w)
            sample_counts.append(fit_res.num_examples)

            if "train_loss" in fit_res.metrics:
                losses_weighted.append(
                    fit_res.metrics["train_loss"] * fit_res.num_examples
                )
                loss_examples.append(fit_res.num_examples)

        # ── 2. Aggregate ──────────────────────────────────────────────────────
        agg_weights: List[np.ndarray]

        if self.aggregation in ("fedavg", "fedprox"):
            # Use Flower's built-in weighted averaging
            aggregated_params, agg_metrics = super().aggregate_fit(
                server_round, results, failures
            )
            if aggregated_params is None:
                return None, {}
            agg_weights = fl.common.parameters_to_ndarrays(aggregated_params)

        elif self.aggregation == "krum" and _CUSTOM_AGG_AVAILABLE:
            agg_weights = krum_aggregate(weights_list, self.num_byzantine)
            agg_metrics = {}

        elif self.aggregation == "multikrum" and _CUSTOM_AGG_AVAILABLE:
            agg_weights = multi_krum_aggregate(
                weights_list, self.num_byzantine, self.krum_m
            )
            agg_metrics = {}

        elif self.aggregation == "trimmed_mean" and _CUSTOM_AGG_AVAILABLE:
            agg_weights = trimmed_mean_aggregate(weights_list, self.trim_ratio)
            agg_metrics = {}

        elif self.aggregation == "median" and _CUSTOM_AGG_AVAILABLE:
            agg_weights = coordinate_median_aggregate(weights_list)
            agg_metrics = {}

        else:
            logger.warning(
                "Unknown aggregation '%s'. Falling back to FedAvg.", self.aggregation
            )
            aggregated_params, agg_metrics = super().aggregate_fit(
                server_round, results, failures
            )
            agg_weights = fl.common.parameters_to_ndarrays(aggregated_params)

        # ── 3. Convert back to Flower Parameters & save ───────────────────────
        aggregated_parameters = fl.common.ndarrays_to_parameters(agg_weights)
        os.makedirs(self.save_folder, exist_ok=True)
        weight_path = os.path.join(
            self.save_folder, f"round-{server_round}-weights.pkl"
        )
        save_parameters(aggregated_parameters, weight_path)
        logger.info(
            "Round %d (%s) weights saved → %s", server_round, self.aggregation, weight_path
        )

        # ── 4. Record training loss ───────────────────────────────────────────
        loss_val: Optional[float] = None
        if loss_examples and sum(loss_examples) > 0:
            loss_val = sum(losses_weighted) / sum(loss_examples)
        self.round_history.append({"round": server_round, "loss": loss_val})

        return aggregated_parameters, agg_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Data metadata
# ──────────────────────────────────────────────────────────────────────────────

def get_metadata(data_dir: str = "data") -> Tuple[int, int]:
    """Return (input_shape, num_classes) from saved artefacts."""
    csv_path = os.path.join(data_dir, "processed_data.csv")
    le_path  = os.path.join(data_dir, "label_encoder.pkl")

    if not os.path.exists(csv_path):
        return 0, 0

    df = pd.read_csv(csv_path, nrows=5)
    input_shape = df.shape[1] - 1   # subtract label column

    if os.path.exists(le_path):
        with open(le_path, "rb") as fh:
            le = pickle.load(fh)
        num_classes = len(le.classes_)
    else:
        num_classes = 2

    return input_shape, num_classes


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FLEX-ID Federated Learning Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--strategy", type=str, default="fedavg",
        choices=["fedavg", "fedprox"],
        help="High-level strategy name (sets FedProx proximal term when 'fedprox').",
    )
    parser.add_argument(
        "--aggregation", type=str, default=None,
        choices=["fedavg", "fedprox", "krum", "multikrum", "trimmed_mean", "median"],
        help=(
            "Aggregation algorithm.  Defaults to --strategy value.  "
            "Use this to run krum/median/etc. on top of the FedAvg communication protocol."
        ),
    )
    parser.add_argument(
        "--rounds", type=int, default=30,
        help="Number of federated learning rounds.",
    )
    parser.add_argument(
        "--num_clients", type=int, default=4,
        help="Total number of clients (4, 8, or 16).",
    )
    parser.add_argument(
        "--proximal_mu", type=float, default=0.1,
        help="FedProx proximal term coefficient (μ).  Used only when strategy=fedprox.",
    )
    parser.add_argument(
        "--num_byzantine", type=int, default=1,
        help="Assumed number of Byzantine clients for Krum variants.",
    )
    parser.add_argument(
        "--trim_ratio", type=float, default=0.1,
        help="Fraction of clients to trim from each tail (Trimmed Mean).",
    )
    parser.add_argument(
        "--krum_m", type=int, default=None,
        help="Number of clients selected by Multi-Krum.  Defaults to n - f.",
    )
    parser.add_argument(
        "--attack", action="store_true",
        help="Use attack-mode result folder names.",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data",
        help="Directory containing processed_data.csv and label_encoder.pkl.",
    )
    args = parser.parse_args()

    # Resolve effective aggregation name
    effective_agg = args.aggregation if args.aggregation is not None else args.strategy

    # ── Pre-flight ─────────────────────────────────────────────────────────────
    force_release_port(8080)
    input_shape, num_classes = get_metadata(args.data_dir)
    logger.info(
        "Metadata — input_shape=%d, num_classes=%d", input_shape, num_classes
    )

    # ── Config callback ────────────────────────────────────────────────────────
    def on_fit_config(rnd: int) -> Dict[str, Any]:
        mu = args.proximal_mu if args.strategy == "fedprox" else 0.0
        return {
            "round": rnd,
            "proximal_mu": mu,
            "input_shape": input_shape,
            "num_classes": num_classes,
        }

    # ── Determine save folder ─────────────────────────────────────────────────
    if args.attack:
        folder_name = (
            "fedproxunderattack" if args.strategy == "fedprox" else "fedunderattack"
        )
    else:
        folder_name = f"{effective_agg}eachround"

    save_folder = f"results/{folder_name}"

    # ── Build strategy ─────────────────────────────────────────────────────────
    strategy_kwargs: Dict[str, Any] = dict(
        aggregation=effective_agg,
        save_folder=save_folder,
        num_byzantine=args.num_byzantine,
        trim_ratio=args.trim_ratio,
        krum_m=args.krum_m,
        min_fit_clients=args.num_clients,
        min_evaluate_clients=args.num_clients,
        min_available_clients=args.num_clients,
        on_fit_config_fn=on_fit_config,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # FedProx needs the proximal_mu passed to the parent FedProx class
    if args.strategy == "fedprox":
        strategy_kwargs["proximal_mu"] = args.proximal_mu

    # Choose parent class
    if args.strategy == "fedprox" and effective_agg in ("fedavg", "fedprox"):
        # Use Flower's FedProx as parent for correct proximal handling
        class _Strategy(FlexIDStrategy, fl.server.strategy.FedProx):
            pass
    else:
        _Strategy = FlexIDStrategy  # type: ignore[assignment]

    strategy = _Strategy(**strategy_kwargs)

    logger.info(
        "Starting server — strategy=%s, aggregation=%s, rounds=%d, num_clients=%d",
        args.strategy, effective_agg, args.rounds, args.num_clients,
    )

    # ── Start server ───────────────────────────────────────────────────────────
    history_obj = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    # ── Assemble combined history ──────────────────────────────────────────────
    train_loss_map = {item["round"]: item["loss"] for item in strategy.round_history}
    all_rounds = sorted(train_loss_map.keys())
    combined_history: List[Dict[str, Any]] = []

    for r in all_rounds:
        entry: Dict[str, Any] = {"round": r, "train_loss": train_loss_map.get(r)}

        if "accuracy" in history_obj.metrics_distributed:
            acc_list = history_obj.metrics_distributed["accuracy"]
            entry["accuracy"] = next((v for rnd, v in acc_list if rnd == r), None)

        if "f1" in history_obj.metrics_distributed:
            f1_list = history_obj.metrics_distributed["f1"]
            entry["f1"] = next((v for rnd, v in f1_list if rnd == r), None)

        combined_history.append(entry)

    os.makedirs("results", exist_ok=True)
    hist_suffix = "underattack" if args.attack else ""
    history_filename = (
        f"results/{args.strategy}{'_' + hist_suffix if hist_suffix else ''}_history.pkl"
    )
    save_history(combined_history, history_filename)
    logger.info("History saved → %s", history_filename)


if __name__ == "__main__":
    main()