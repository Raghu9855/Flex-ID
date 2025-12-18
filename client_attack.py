import argparse
import pickle
import os
import sys
import numpy as np
import flwr as fl
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

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
        
        # DEBUG: Print class distribution
        unique, counts = np.unique(self.y_train, return_counts=True)
        print(f"[Client {cid}] Label Distribution: {dict(zip(unique, counts))}")

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
            
        # -------------------- PRE-PROCESS: SMOTE & CLASS WEIGHTS --------------------
        self.X_train_final, self.y_train_final = self.X_train, self.y_train
        self.class_weight_dict = None

        print(f"\n[Client {self.cid}] Preparing Data (SMOTE)...")
        # Standard SMOTE Logic (Adapted from client.py)
        try:
            unique_cls, counts = np.unique(self.y_train, return_counts=True)
            min_samples = np.min(counts)
            
            # Only apply if we have enough samples (k_neighbors=5 requires >=6)
            if min_samples > 5:
                # Custom Strategy: Cap upsampling at 5000
                TARGET_COUNT = 5000
                sampling_strategy = {}
                for cls, count in zip(unique_cls, counts):
                    if count < TARGET_COUNT:
                        sampling_strategy[cls] = TARGET_COUNT
                    # Else (e.g. Benign has 100k): Do nothing
                
                # Only run SMOTE if we actually have classes to upsample
                if sampling_strategy:
                    sm = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=5, random_state=42)
                    X_res, y_res = sm.fit_resample(self.X_train, self.y_train)
                    print(f"[Client {self.cid}] ✅ SMOTE Applied. Size: {len(self.X_train)} -> {len(X_res)}")
                    self.X_train_final, self.y_train_final = X_res, y_res
                else:
                    print(f"[Client {self.cid}] ℹ️ SMOTE skipped (All relevant classes > {TARGET_COUNT})")
            else:
                 print(f"[Client {self.cid}] ⚠️ Skipping SMOTE (Classes too small per client: min={min_samples}).")
                 
        except Exception as e:
            print(f"[Client {self.cid}] ⚠️ SMOTE Failed: {e}")

        # Class Weights (Fallback if SMOTE skipped or failed)
        if len(self.X_train_final) == len(self.X_train): 
            try:
                class_weights_vals = compute_class_weight(
                    class_weight="balanced", 
                    classes=np.unique(self.y_train_final), 
                    y=self.y_train_final
                )
                self.class_weight_dict = dict(zip(np.unique(self.y_train_final), class_weights_vals))
                print(f"[Client {self.cid}] ⚖️ Class Weights Enabled (Balanced)")
            except Exception as e:
                print(f"[Client {self.cid}] ⚠️ Failed to compute class weights: {e}")
                self.class_weight_dict = None
        else:
             self.class_weight_dict = None
             print(f"[Client {self.cid}] ⚖️ Class Weights Disabled (Data Balanced via SMOTE)")

        # 3. Build Model
        # 3. Build Model (Use X_train_final for shape)
        self.model = create_dnn_model(self.X_train_final.shape[1], self.num_classes)
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        
        # Use prepared data
        X_train_final, y_train_final = self.X_train_final, self.y_train_final

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

        # Custom FedProx Training Loop or Standard Fit
        proximal_mu = float(config.get("proximal_mu", 0.0))
        
        if proximal_mu > 0.0:
            print(f"[Client {self.cid}] 🦅 Training with FedProx (mu={proximal_mu})")
            
            # Fix for Shape Mismatch: 
            # 'parameters' contains ALL weights (including non-trainable BN stats).
            # 'trainable_variables' contains ONLY trainable weights.
            # Directly zipping them causes misalignment. 
            # Since we just called set_weights(parameters), the model's current trainable_variables ARE the global weights.
            global_trainable_weights = [tf.identity(v) for v in self.model.trainable_variables]
            
            # Prepare dataset
            batch_size = 32
            epochs = 5
            dataset = tf.data.Dataset.from_tensor_slices((X_train_final, y_train_final))
            dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
            
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
            optimizer = self.model.optimizer

            # Optimization: Use tf.function for graph execution (Much faster)
            @tf.function
            def train_step(data, labels, global_w):
                with tf.GradientTape() as tape:
                    predictions = self.model(data, training=True)
                    loss_value = loss_fn(labels, predictions)
                    
                    proximal_term = 0.0
                    for local_var, global_var in zip(self.model.trainable_variables, global_w):
                        proximal_term += tf.reduce_sum(tf.square(local_var - global_var))
                        
                    total_loss = loss_value + (proximal_mu / 2.0) * proximal_term
                    
                grads = tape.gradient(total_loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                return total_loss

            print(f"[Client {self.cid}] ⏳ Starting training (Graph Mode)...")
            for epoch in range(epochs):
                epoch_loss = 0.0
                num_batches = 0
                for batch_X, batch_y in dataset:
                    loss = train_step(batch_X, batch_y, global_trainable_weights)
                    epoch_loss += loss
                    num_batches += 1
                
                # Optional: Print progress per epoch to show it's alive
                # print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/num_batches:.4f}")
            print(f"[Client {self.cid}] ✅ Training Complete.")
                
                # Optional: print epoch loss?
                # print(f"Epoch {epoch+1}/{epochs} loss: {total_loss.numpy():.4f}")
                
        else:
            # Standard Local Training
            # Pass class_weight if available (only works if not SMOTE-ed)
            self.model.fit(
                X_train_final, y_train_final, 
                epochs=5, batch_size=32, verbose=0,
                class_weight=self.class_weight_dict
            )
        
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
        # Calculate F1 Score
        y_pred = np.argmax(self.model.predict(self.X_test, verbose=0), axis=1)
        from sklearn.metrics import f1_score
        f1 = f1_score(self.y_test, y_pred, average="macro")

        print(f"[Client {self.cid}] Eval => Loss: {loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")
        return float(loss), len(self.y_test), {"accuracy": float(acc), "f1": float(f1)}

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