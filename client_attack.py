import argparse
import pickle
import os
import sys
import numpy as np
import flwr as fl
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Use non-interactive backend for plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import model
try:
    from model import create_dnn_model
except ImportError:
    sys.exit("❌ Critical Error: 'model.py' not found.")

# --- HELPERS ---
def load_partition(cid: int):
    filename = f"client_partition_{cid}.pkl"
    with open(filename, "rb") as f:
        return pickle.load(f)

# --- MALICIOUS CLIENT ---
class MaliciousClient(fl.client.NumPyClient):
    def __init__(self, cid, attack_type="none", scale=1.0):
        self.cid = cid
        self.attack_type = attack_type
        self.scale = scale
        
        print(f"--- Client {cid} Initializing [Attack: {self.attack_type}, Scale: {self.scale}] ---")

        # 1. Load Data
        (self.X_train, self.y_train), (self.X_test, self.y_test) = load_partition(cid)
        self.y_train = np.array(self.y_train).reshape(-1)
        self.y_test = np.array(self.y_test).reshape(-1)

        # 2. Global Encoder Logic
        self.num_classes = 0
        self.benign_class_idx = 0 
        
        if os.path.exists("label_encoder.pkl"):
            with open("label_encoder.pkl", "rb") as f:
                le = pickle.load(f)
            self.num_classes = len(le.classes_)
            
            # Find which integer represents 'Benign'
            if 'Benign' in le.classes_:
                self.benign_class_idx = int(le.transform(['Benign'])[0])
        else:
            self.num_classes = len(np.unique(self.y_train))

        # 3. Build Model
        self.model = create_dnn_model(self.X_train.shape[1], self.num_classes)
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        
        X_train_final, y_train_final = self.X_train, self.y_train

        # --- 😈 ATTACK 1: DATA POISONING (Label Flipping) 😈 ---
        if self.attack_type == "flip":
            print(f"[Client {self.cid}] ⚠️ Executing Label Flipping Attack...")
            # Flip a portion of labels based on scale (scale=1.0 means 100% flip)
            num_samples = len(y_train_final)
            num_flip = int(num_samples * min(self.scale, 1.0))
            
            indices = np.random.choice(num_samples, num_flip, replace=False)
            y_train_final[indices] = self.benign_class_idx
            print(f"[Client {self.cid}] Flipped {num_flip}/{num_samples} labels to Class {self.benign_class_idx} (Benign).")
            
        else:
            # Normal behavior or partial benign behavior needed for other attacks?
            # For Model Poisoning, we train normally FIRST, then poison weights.
            pass

        # Train
        self.model.fit(X_train_final, y_train_final, epochs=5, batch_size=32, verbose=0)
        
        # --- 😈 ATTACK 2: MODEL POISONING (Noise Injection) 😈 ---
        final_weights = self.model.get_weights()
        
        if self.attack_type == "noise":
            print(f"[Client {self.cid}] ⚠️ Executing Model Poisoning (Noise) Attack...")
            poisoned_weights = []
            for w in final_weights:
                # Add Gaussian noise
                noise = np.random.normal(0, self.scale, w.shape)
                poisoned_weights.append(w + noise)
            
            final_weights = poisoned_weights
            print(f"[Client {self.cid}] Added Gaussian Noise (std={self.scale}) to weights.")

        return final_weights, len(X_train_final), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"[Client {self.cid}] Eval => Loss: {loss:.4f}, Acc: {acc:.4f}")
        return float(loss), len(self.y_test), {"accuracy": float(acc)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True)
    parser.add_argument("--attack_type", type=str, default="none", choices=["none", "flip", "noise"], help="Type of attack")
    parser.add_argument("--scale", type=float, default=1.0, help="Intensity of attack (Flip ratio 0-1, or Noise std dev)")
    
    # Backward compatibility
    parser.add_argument("--malicious", action='store_true', help="Legacy flag for flip attack")
    
    args = parser.parse_args()

    # Handle legacy flag
    if args.malicious and args.attack_type == "none":
        args.attack_type = "flip"

    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=MaliciousClient(args.cid, args.attack_type, args.scale))