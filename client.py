#!/usr/bin/env python3
import argparse
import pickle
import os
import sys
from typing import Tuple, Dict

# Use a non-interactive backend for matplotlib (prevents errors on servers)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import flwr as fl
import tensorflow as tf
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

# Import your model definition
# Ensure model.py is in the same directory
try:
    from model import create_dnn_model
except ImportError:
    sys.exit("❌ Critical Error: 'model.py' not found. Please create it with 'create_dnn_model' function.")

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def load_partition(cid: int):
    """Loads the specific partition for this client ID."""
    filename = f"client_partition_{cid}.pkl"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"❌ Partition file not found: {filename}. Run create_partitions.py first.")
    
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_class_distribution_plot(y: np.ndarray, title: str, filename: str, le: LabelEncoder = None):
    """Generates a bar chart comparing class counts and saves it as PNG."""
    unique, counts = np.unique(y, return_counts=True)
    
    # Convert numeric labels to string names if LE is provided
    if le is not None:
        try:
            # Handle cases where LE classes might be strings or ints
            labels = le.inverse_transform(unique)
            labels = [str(x) for x in labels]
        except Exception:
            labels = [str(u) for u in unique]
    else:
        labels = [str(u) for u in unique]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, counts, color='#4CAF50') # Green bars
    plt.xlabel("Classes")
    plt.ylabel("Count")
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    
    # Add numbers on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom', fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"📊 Plot saved: {filename}")

# -----------------------------------------------------------------------------
# FLOWER CLIENT
# -----------------------------------------------------------------------------

class FLClient(fl.client.NumPyClient):
    def __init__(self, cid: int, is_malicious: bool = False, batch_size: int = 32, fast_run: bool = False):
        self.cid = cid
        self.is_malicious = is_malicious
        self.batch_size = batch_size
        self.fast_run = fast_run
        print(f"--- Client {cid} Initializing (Batch Size: {batch_size}, Fast Run: {fast_run}) ---")
        
        # 1. Load Data
        (self.X_train, self.y_train), (self.X_test, self.y_test) = load_partition(cid)

        # FAST RUN: Subsample data
        if self.fast_run:
            print(f"[Client {cid}] ⚡ FAST RUN ENABLED: Using 10% of data")
            # Take first 10%
            limit_train = int(len(self.y_train) * 0.1)
            limit_test = int(len(self.y_test) * 0.1)
            self.X_train = self.X_train[:limit_train]
            self.y_train = self.y_train[:limit_train]
            self.X_test = self.X_test[:limit_test]
            self.y_test = self.y_test[:limit_test]

        # Ensure correct shapes (1D labels)
        self.y_train = np.array(self.y_train).reshape(-1)
        self.y_test = np.array(self.y_test).reshape(-1)

        # 2. Load Global Label Encoder (Crucial for correct input/output shape)
        self.global_le = None
        self.num_classes = 0
        
        if os.path.exists("label_encoder.pkl"):
            try:
                with open("label_encoder.pkl", "rb") as f:
                    self.global_le = pickle.load(f)
                self.num_classes = len(self.global_le.classes_)
                print(f"[Client {cid}] Global Encoder Loaded. Total Classes: {self.num_classes}")
            except Exception as e:
                print(f"[Client {cid}] ⚠️ Warning: Failed to load label_encoder.pkl: {e}")
        
        # Fallback if global encoder missing
        if self.num_classes == 0:
            self.num_classes = len(np.unique(self.y_train))
            print(f"[Client {cid}] ⚠️ Using local class count: {self.num_classes} (Ensure this matches Server!)")

        # 3. Filter Unseen Classes (Safety check)
        # If the client has data that the Global Encoder doesn't know about, it causes crashes.
        if self.global_le:
            valid_classes = set(range(len(self.global_le.classes_)))
            # Assume data is already integer encoded by partition script. 
            # If valid_classes are [0,1,2,3,4] and client has [5], remove [5].
            mask_train = np.isin(self.y_train, list(valid_classes))
            mask_test = np.isin(self.y_test, list(valid_classes))
            
            self.X_train = self.X_train[mask_train]
            self.y_train = self.y_train[mask_train]
            self.X_test = self.X_test[mask_test]
            self.y_test = self.y_test[mask_test]

        print(f"[Client {cid}] Data Loaded -> Train: {len(self.y_train)}, Test: {len(self.y_test)}")

        # 4. PLOT BEFORE SMOTE
        save_class_distribution_plot(
            self.y_train, 
            f"Client {cid} - Distribution BEFORE SMOTE", 
            f"client_{cid}_dist_before_smote.png", 
            self.global_le
        )

        # -------------------- PRE-PROCESS: SMOTE & CLASS WEIGHTS --------------------
        # Perform this ONCE here, not every round in fit()
        self.X_train_final, self.y_train_final = self.X_train, self.y_train
        self.class_weight_dict = None

        print(f"\n[Client {self.cid}] Preparing Data (SMOTE)...")
        if self.is_malicious:
             # Malicious Client: Poisoning Attack (Label Flipping) happens in fit() or we prep it here?
             # If we want to support dynamic attacks we might keep attack logic in fit or prep here.
             # For simpler architecture, let's keep attack logic simple or handle "Flip" here if static.
             # BUT `client_attack.py` overrides fit(). This file is `client.py`.
             # Standard client doesn't attack.
             
             pass # Standard client logic below
        
        # Standard SMOTE Logic
        try:
            unique_cls, counts = np.unique(self.y_train, return_counts=True)
            min_samples = np.min(counts)
            if min_samples > 1:
                k = min(5, min_samples - 1)
                # Custom Strategy: Cap upsampling at 5000 to avoid excessive noise from rare classes (e.g. 3 -> 100k)
                TARGET_COUNT = 5000
                sampling_strategy = {}
                for cls, count in zip(unique_cls, counts):
                    if count < TARGET_COUNT:
                        sampling_strategy[cls] = TARGET_COUNT
                    # Else (e.g. Benign has 100k): Do nothing, keep original count
                
                # Only run SMOTE if we actually have classes to upsample
                if sampling_strategy:
                    sm = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k, random_state=42)
                    X_res, y_res = sm.fit_resample(self.X_train, self.y_train)
                    print(f"[Client {self.cid}] ✅ SMOTE Applied (Capped at {TARGET_COUNT}). Size: {len(self.X_train)} -> {len(X_res)}")
                    self.X_train_final, self.y_train_final = X_res, y_res
                else:
                    print(f"[Client {self.cid}] ℹ️ SMOTE skipped (All classes > {TARGET_COUNT})")
                
                # PLOT AFTER SMOTE
                save_class_distribution_plot(
                    y_res, 
                    f"Client {self.cid} - Distribution AFTER SMOTE", 
                    f"client_{self.cid}_dist_after_smote.png", 
                    self.global_le
                )
            else:
                 print(f"[Client {self.cid}] ⚠️ Skipping SMOTE (Class too small).")
        except Exception as e:
            print(f"[Client {self.cid}] ⚠️ SMOTE Failed: {e}")

        # Class Weights
            # Only use class weights if SMOTE was NOT applied or failed
            if len(self.X_train_final) == len(self.X_train): 
                class_weights_vals = compute_class_weight(
                    class_weight="balanced", 
                    classes=np.unique(self.y_train_final), 
                    y=self.y_train_final
                )
                self.class_weight_dict = dict(zip(np.unique(self.y_train_final), class_weights_vals))
                print(f"[Client {self.cid}] ⚖️ Class Weights Enabled (Balanced)")
            else:
                 self.class_weight_dict = None
                 print(f"[Client {self.cid}] ⚖️ Class Weights Disabled (Data is already balanced via SMOTE)")
        except:
            self.class_weight_dict = None


        # 5. Build Model
        # Input shape = features, Output shape = global total classes
        self.model = create_dnn_model(self.X_train.shape[1], self.num_classes)
        self.model.compile(
            optimizer="adam", 
            loss="sparse_categorical_crossentropy", 
            metrics=["accuracy"]
        )

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        
        # Check if malicious (legacy check, though client.py is usually benign)
        # If this checks for attributes set by client_attack.py subclasses, we need to be careful.
        # But standard client just uses self.X_train_final which is pre-processed.
        
        # Quick check for empty data
        if len(self.y_train_final) == 0:
             return self.model.get_weights(), 0, {}

        from sklearn.utils import shuffle
        self.X_train_final, self.y_train_final = shuffle(self.X_train_final, self.y_train_final, random_state=42)

        history = self.model.fit(
            self.X_train_final, self.y_train_final,
            epochs=3, # Reduced from 10 to prevent overfitting
            batch_size=self.batch_size,
            validation_data=(self.X_test, self.y_test), # Validate on REAL test data, not SMOTE split
            verbose=1,
            class_weight=self.class_weight_dict
        )
        
        loss = history.history['loss'][-1]
        # Check accuracy if available
        acc = history.history['accuracy'][-1] if 'accuracy' in history.history else 0.0
        
        print(f"[Client {self.cid}] Round complete. Loss: {loss:.4f} | Acc: {acc:.4f}")
        
        # --- SANITY CHECK: Local Generalization ---
        # Evaluate "Just Trained" model on Local Test Set
        # If this is High (e.g. 80%) but Global Eval is Low (16%), then Aggregation is hurting.
        # If this is Low (16%), then the model is Overfitting to Train Data.
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"[Client {self.cid}] 🔍 Local Test Sanity Check => Acc: {test_acc:.4f} (vs Train: {acc:.4f})")

        return self.model.get_weights(), len(self.X_train_final), {"train_loss": float(loss)}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        
        if len(self.y_test) == 0:
            return float('nan'), 0, {}

        loss, acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        
        # Calculate F1 Score
        y_pred = np.argmax(self.model.predict(self.X_test, verbose=0), axis=1)
        
        # Use 'macro' for imbalanced data, 'weighted' is also good
        f1 = f1_score(self.y_test, y_pred, average="macro")

        print(f"[Client {self.cid}] Eval => Loss: {loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")
        return float(loss), len(self.y_test), {"accuracy": float(acc), "f1": float(f1)}

# -----------------------------------------------------------------------------
# MAIN ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True, help="Client ID (0-3)")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--fast_run", action="store_true", help="Use 10% of data for debugging")
    args = parser.parse_args()

    # Connect to the server
    # 0.0.0.0 is for server binding; clients connect to localhost (127.0.0.1) or specific IP
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080", 
        client=FLClient(args.cid, batch_size=args.batch_size, fast_run=args.fast_run)
    )