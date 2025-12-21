# server.py
import argparse
import sys
# Fix for Windows Unicode errors
sys.stdout.reconfigure(encoding='utf-8')
import os
import pickle
import pandas as pd
import numpy as np
import flwr as fl
from typing import Optional, Tuple, List, Dict, Any
import subprocess
import platform
import time

def force_release_port(port):
    """Kills any process listening on the specified port to prevent binding errors."""
    try:
        if platform.system() == "Windows":
            cmd_find = f"netstat -ano | findstr :{port}"
            try:
                # Use shell=True for Windows pipes
                result = subprocess.check_output(cmd_find, shell=True, stderr=subprocess.DEVNULL).decode()
                if result:
                    lines = result.strip().split("\n")
                    for line in lines:
                        if "LISTENING" in line:
                            parts = line.split()
                            pid = parts[-1]
                            print(f"⚠️ Found zombie process {pid} on port {port}. Killing it...")
                            subprocess.run(f"taskkill /PID {pid} /F", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                pass # No process found is good
        else:
            subprocess.run(f"fuser -k {port}/tcp", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1) 
    except Exception as e:
        print(f"Warning: Port check failed: {e}")

# Save Helpers
def save_history(obj, filename):
    with open(filename, "wb") as f: pickle.dump(obj, f)

def save_parameters(params, filename):
    with open(filename, "wb") as f: pickle.dump(fl.common.parameters_to_ndarrays(params), f)

# Metric Aggregation Function
def weighted_average(metrics: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, Any]:
    # Aggregate accuracy and F1 using weighted average
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    f1s = [num_examples * m.get("f1", 0) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    return {
        "accuracy": sum(accuracies) / sum(examples),
        "f1": sum(f1s) / sum(examples),
    }

# Aggregation Strategy
class SafeAggregationMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_history = []

    def aggregate_fit(self, rnd, results, failures):
        if not results: return None, {}
        
        # 1. Standard Parameter Aggregation
        aggregated_parameters, agg_metrics = super().aggregate_fit(rnd, results, failures)
        
        # 2. Save Weights to Disk
        if aggregated_parameters:
            folder = getattr(self, 'save_folder', f"results/{'fedproxeachround' if isinstance(self, fl.server.strategy.FedProx) else 'fedavgeachround'}")
            os.makedirs(folder, exist_ok=True)
            save_parameters(aggregated_parameters, f"{folder}/round-{rnd}-weights.pkl")
            print(f"Round {rnd} weights saved to {folder}")

        # 3. FIX: Manually Aggregate Training Loss
        # We extract 'train_loss' from every client's result
        losses_weighted = []
        examples = []
        
        for _, fit_res in results:
            # Check if client returned train_loss
            if "train_loss" in fit_res.metrics:
                losses_weighted.append(fit_res.metrics["train_loss"] * fit_res.num_examples)
                examples.append(fit_res.num_examples)
        
        # Calculate weighted average
        if examples and sum(examples) > 0:
            loss_val = sum(losses_weighted) / sum(examples)
        else:
            loss_val = None

        # 4. Save to History
        self.round_history.append({"round": rnd, "loss": loss_val})
        
        return aggregated_parameters, agg_metrics

class SaveFedAvg(SafeAggregationMixin, fl.server.strategy.FedAvg): pass
class SaveFedProx(SafeAggregationMixin, fl.server.strategy.FedProx): pass

# Metadata Helper
def get_metadata():
    if not os.path.exists("data/processed_data.csv"): return 0, 0
    df = pd.read_csv("data/processed_data.csv", nrows=100)
    input_shape = df.shape[1] - 1
    if os.path.exists("data/label_encoder.pkl"):
        with open("data/label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        num_classes = len(le.classes_)
    else:
        num_classes = 2
    return input_shape, num_classes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="fedavg", choices=["fedavg", "fedprox"])
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--proximal_mu", type=float, default=0.1)
    parser.add_argument("--attack", action="store_true", help="Enable attack mode storage")
    args = parser.parse_args()

    # Pre-flight check: Ensure port 8080 is free to prevent binding errors
    force_release_port(8080)

    input_shape, num_classes = get_metadata()

    def on_fit_config_fn(rnd):
        print(f"Round {rnd} starting...")
        mu = 0.0 if args.strategy == "fedavg" else args.proximal_mu
        return {
            "round": rnd, "proximal_mu": mu,
            "input_shape": input_shape, "num_classes": num_classes
        }

    MIN_CLIENTS = 4
    StrategyClass = SaveFedAvg if args.strategy == "fedavg" else SaveFedProx
    strategy_args = {
        "min_fit_clients": MIN_CLIENTS, 
        "min_evaluate_clients": MIN_CLIENTS,
        "min_available_clients": MIN_CLIENTS,
        "on_fit_config_fn": on_fit_config_fn,
        "evaluate_metrics_aggregation_fn": weighted_average  # <--- Vital for plot_history.py
    }
    if args.strategy == "fedprox": strategy_args["proximal_mu"] = args.proximal_mu

    strategy = StrategyClass(**strategy_args)

    # Determine save folder based on strategy and attack mode
    if args.strategy == "fedavg":
        folder_name = "fedunderattack" if args.attack else "fedavgeachround"
    else: # fedprox
        folder_name = "fedproxunderattack" if args.attack else "fedproxeachround"
    
    # Patch the strategy to use the correct folder
    # We dynamically bind the folder path to the instance so aggregate_fit uses it
    strategy.save_folder = f"results/{folder_name}"

    print(f"Starting {args.strategy} Server... (Storage: {strategy.save_folder})")
    history_obj = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy
    )
    
    # COMBINE LOSS FROM FIT_ROUND AND ACCURACY FROM EVALUATE_ROUND
    # history_obj.metrics_distributed['accuracy'] contains [(round, acc), ...]
    
    combined_history = []
    # 1. Training Loss (collected in our custom aggregate_fit)
    train_loss_map = {item['round']: item['loss'] for item in strategy.round_history}
    
    # 2. Validation Metrics (collected by Flower's start_server -> history_obj)
    # history_obj.metrics_distributed maps "accuracy" -> [(1, 0.5), (2, 0.6)...]
    
    # Get total rounds from args or max round
    all_rounds = sorted(list(train_loss_map.keys()))
    
    for r in all_rounds:
        entry = {"round": r, "train_loss": train_loss_map.get(r)}
        
        # Find matching accuracy
        if "accuracy" in history_obj.metrics_distributed:
            acc_list = history_obj.metrics_distributed["accuracy"]
            # Find tuple with round r
            match = next((val for rnd, val in acc_list if rnd == r), None)
            entry["accuracy"] = match
            
        if "f1" in history_obj.metrics_distributed:
            f1_list = history_obj.metrics_distributed["f1"]
            match = next((val for rnd, val in f1_list if rnd == r), None)
            entry["f1"] = match
            
        combined_history.append(entry)

    os.makedirs("results", exist_ok=True)
    
    # Determine history filename
    if args.attack:
        history_filename = f"results/{args.strategy}_underattack_history.pkl"
    else:
        history_filename = f"results/{args.strategy}_history.pkl"
        
    save_history(combined_history, history_filename)
    print(f"History saved to {history_filename}")

if __name__ == "__main__":
    main()