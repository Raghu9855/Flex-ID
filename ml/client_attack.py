import argparse
import pickle
import os
import sys
# Fix for Windows Unicode errors in terminal
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import flwr as fl
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# Import model
try:
    from model import create_dnn_model
except ImportError:
    sys.exit("Critical Error: 'model.py' not found.")

# --- HELPERS ---
def load_partition(cid: int):
    """Load a client's data partition from the data/ subfolder."""
    filename = f"data/client_partition_{cid}.pkl"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Partition file not found: {filename}. Run 2_create_partitions.py first.")
    with open(filename, "rb") as f:
        return pickle.load(f)

# --- MALICIOUS CLIENT ---
class MaliciousClient(fl.client.NumPyClient):
    def __init__(self, cid, attack_type="none", scale=1.0, batch_size=32, fast_run=False,
                 trigger_feature_idx=0, trigger_value=999.0):
        """
        Malicious federated learning client supporting five attack types:

        - "none"      : Honest client (no attack).
        - "flip"      : Data poisoning via label flipping. Flips `scale * 100%` of
                        training labels to the Benign class, causing the global model
                        to under-detect attacks.
        - "noise"     : Model poisoning via Gaussian weight perturbation. Injects
                        Gaussian noise (std = scale) into the trained model weights
                        before sending them to the server.
        - "backdoor"  : Targeted data poisoning. Stamps a fixed trigger
                        (feature[trigger_feature_idx] = trigger_value) onto
                        `scale * 100%` of training samples and flips their label to
                        Benign. At inference, any traffic with the trigger is
                        misclassified as benign while the model behaves normally
                        on clean data.
        - "byzantine": Model replacement / scaling attack. Multiplies the locally
                        trained weight update by `scale` before submission, aiming
                        to dominate the FedAvg aggregate and steer the global model
                        toward the attacker's objective.
        - "adaptive"  : Constrain-and-scale adaptive poisoning. Performs gradient
                        ascent on the cross-entropy loss for `scale * 10` extra
                        steps after normal training, then clips the weight update
                        to the L2-norm of an honest update so the attack stays
                        within typical model-update bounds (stealthy).
        """
        self.cid = cid
        self.attack_type = attack_type
        self.scale = scale
        self.batch_size = batch_size
        self.fast_run = fast_run
        self.trigger_feature_idx = trigger_feature_idx   # Feature column used as backdoor trigger
        self.trigger_value = trigger_value               # Value stamped on that feature

        print(f"--- Client {cid} Initializing [Attack: {self.attack_type}, Scale: {self.scale}, Batch: {self.batch_size}] ---")

        # 1. Load Data
        (self.X_train, self.y_train), (self.X_test, self.y_test) = load_partition(cid)

        # FAST RUN: Subsample data
        if self.fast_run:
            print(f"[Client {cid}] [FAST] FAST RUN ENABLED: Using 10% of data")
            limit_train = int(len(self.y_train) * 0.1)
            limit_test = int(len(self.y_test) * 0.1)
            self.X_train = self.X_train[:limit_train]
            self.y_train = self.y_train[:limit_train]
            self.X_test = self.X_test[:limit_test]
            self.y_test = self.y_test[:limit_test]

        self.y_train = np.array(self.y_train).reshape(-1)
        self.y_test = np.array(self.y_test).reshape(-1)
        
        # DEBUG: Print class distribution
        unique, counts = np.unique(self.y_train, return_counts=True)
        print(f"[Client {cid}] Label Distribution: {dict(zip(unique, counts))}")

        # 2. Global Encoder Logic
        self.num_classes = 0
        self.benign_class_idx = 0 
        
        if os.path.exists("data/label_encoder.pkl"):
            with open("data/label_encoder.pkl", "rb") as f:
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

        print(f"\n[Client {self.cid}] Preparing Data (Hybrid Balancing)...")
        
        # ---------------- HYBRID STRATEGY ----------------
        try:
            # Current data
            X_curr, y_curr = self.X_train, self.y_train
            unique_cls, counts = np.unique(y_curr, return_counts=True)
            dist = dict(zip(unique_cls, counts))
            
            # 1. DOWNSAMPLE BENIGN (Index 0 usually, but we have self.benign_class_idx)
            # Cap at 50,000 (5:1 Ratio with attacks: 50k vs 10k)
            BENIGN_CAP = 50000
            benign_count = dist.get(self.benign_class_idx, 0)
            
            if benign_count > BENIGN_CAP:
                print(f"[Client {self.cid}] Downsampling Benign from {benign_count} to {BENIGN_CAP}...")
                rus = RandomUnderSampler(sampling_strategy={self.benign_class_idx: BENIGN_CAP}, random_state=42)
                X_curr, y_curr = rus.fit_resample(X_curr, y_curr)
                # Update counts
                unique_cls, counts = np.unique(y_curr, return_counts=True)
                dist = dict(zip(unique_cls, counts))

            # 2. BOOTSTRAP TINY CLASSES (Count < 6)
            # SMOTE failes if neighbors < 6. RandomOverSampler to safe margin (e.g. 20)
            TINY_THRESHOLD = 6
            SAFE_MARGIN = 20
            ros_strategy = {}
            for cls, count in dist.items():
                if count < TINY_THRESHOLD:
                    ros_strategy[cls] = SAFE_MARGIN
            
            if ros_strategy:
                print(f"[Client {self.cid}] Bootstrapping tiny classes: {list(ros_strategy.keys())}")
                ros = RandomOverSampler(sampling_strategy=ros_strategy, random_state=42)
                X_curr, y_curr = ros.fit_resample(X_curr, y_curr)
                # Update counts
                unique_cls, counts = np.unique(y_curr, return_counts=True)
                dist = dict(zip(unique_cls, counts))
                
            # 3. SMOTE
            # Upsample everything else to TARGET_COUNT (10,000)
            TARGET_COUNT = 10000
            smote_strategy = {}
            for cls, count in dist.items():
                # Don't touch Benign (it's already handled or large)
                if cls == self.benign_class_idx:
                    continue
                # If smaller than target, boost it
                if count < TARGET_COUNT:
                    smote_strategy[cls] = TARGET_COUNT
            
            if smote_strategy:
                print(f"[Client {self.cid}] Applying SMOTE to minority classes...")
                sm = SMOTE(sampling_strategy=smote_strategy, k_neighbors=5, random_state=42)
                X_curr, y_curr = sm.fit_resample(X_curr, y_curr)
            
            self.X_train_final, self.y_train_final = X_curr, y_curr
            print(f"[Client {self.cid}] Hybrid Balancing Complete. Final Size: {len(self.X_train_final)}")
            
            # Print new distribution
            unique, counts = np.unique(self.y_train_final, return_counts=True)
            print(f"[Client {self.cid}] New Distribution: {dict(zip(unique, counts))}")

        except Exception as e:
            print(f"[Client {self.cid}] Balancing Failed: {e}")
            # Fallback
            self.X_train_final, self.y_train_final = self.X_train, self.y_train

        # Class Weights (Fallback if SMOTE skipped or failed)
        if len(self.X_train_final) == len(self.X_train): 
            try:
                class_weights_vals = compute_class_weight(
                    class_weight="balanced", 
                    classes=np.unique(self.y_train_final), 
                    y=self.y_train_final
                )
                self.class_weight_dict = dict(zip(np.unique(self.y_train_final), class_weights_vals))
                print(f"[Client {self.cid}] Class Weights Enabled (Balanced)")
            except Exception as e:
                print(f"[Client {self.cid}] Failed to compute class weights: {e}")
                self.class_weight_dict = None
        else:
             self.class_weight_dict = None
             print(f"[Client {self.cid}] Class Weights Disabled (Data Balanced via SMOTE)")

        # 3. Build Model
        # 3. Build Model (Use X_train_final for shape)
        self.model = create_dnn_model(self.X_train_final.shape[1], self.num_classes)
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)

        # Work on copies so the original buffers are not permanently mutated
        X_train_final = self.X_train_final.copy()
        y_train_final = self.y_train_final.copy()

        # ------------------------------------------------------------------
        # ATTACK 1: DATA POISONING — Label Flipping
        # Flip `scale * 100%` of training labels to the Benign class so the
        # global model learns to misclassify attacks as benign traffic.
        # ------------------------------------------------------------------
        if self.attack_type == "flip":
            print(f"[Client {self.cid}] [Attack] Executing Label Flipping Attack...")
            num_samples = len(y_train_final)
            num_flip = int(num_samples * min(self.scale, 1.0))
            indices = np.random.choice(num_samples, num_flip, replace=False)
            y_train_final[indices] = self.benign_class_idx
            print(f"[Client {self.cid}] Flipped {num_flip}/{num_samples} labels -> Class {self.benign_class_idx} (Benign).")

        # ------------------------------------------------------------------
        # ATTACK 2: DATA POISONING — Backdoor (Trigger Injection)
        # Stamp a fixed trigger pattern onto `scale * 100%` of training
        # samples and mislabel them as Benign. After aggregation, any
        # inference sample carrying the trigger will be silently passed as
        # benign while the model performs normally on clean traffic.
        #
        # Trigger design: set feature[trigger_feature_idx] = trigger_value
        # (a statistically anomalous value, e.g., 999.0 for a 0-1 feature).
        # ------------------------------------------------------------------
        elif self.attack_type == "backdoor":
            print(f"[Client {self.cid}] [Attack] Executing Backdoor (Trigger Injection) Attack...")
            num_samples = len(y_train_final)
            num_poison = int(num_samples * min(self.scale, 1.0))
            # Only poison samples that are NOT already Benign
            attack_indices = np.where(y_train_final != self.benign_class_idx)[0]
            if len(attack_indices) == 0:
                attack_indices = np.arange(num_samples)   # Fallback: poison any sample
            chosen = np.random.choice(attack_indices,
                                      min(num_poison, len(attack_indices)),
                                      replace=False)
            X_train_final[chosen, self.trigger_feature_idx] = self.trigger_value
            y_train_final[chosen] = self.benign_class_idx
            print(f"[Client {self.cid}] Backdoor: poisoned {len(chosen)} samples "
                  f"(feature[{self.trigger_feature_idx}]={self.trigger_value} -> Benign).")

        # All other attacks train normally first; weight manipulation happens AFTER training.

        # ------------------------------------------------------------------
        # LOCAL TRAINING (shared by all attack types)
        # ------------------------------------------------------------------
        proximal_mu = float(config.get("proximal_mu", 0.0))

        if proximal_mu > 0.0:
            # ---- FedProx custom training loop ----
            print(f"[Client {self.cid}] Training with FedProx (mu={proximal_mu})")
            global_trainable_weights = [tf.identity(v) for v in self.model.trainable_variables]

            dataset = tf.data.Dataset.from_tensor_slices((X_train_final, y_train_final))
            dataset = dataset.shuffle(buffer_size=1024).batch(self.batch_size)

            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
            optimizer = self.model.optimizer

            @tf.function
            def train_step(data, labels, global_w):
                with tf.GradientTape() as tape:
                    predictions = self.model(data, training=True)
                    loss_value = loss_fn(labels, predictions)
                    proximal_term = tf.add_n([
                        tf.reduce_sum(tf.square(lv - gv))
                        for lv, gv in zip(self.model.trainable_variables, global_w)
                    ])
                    total_loss = loss_value + (proximal_mu / 2.0) * proximal_term
                grads = tape.gradient(total_loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                return total_loss

            print(f"[Client {self.cid}] Starting FedProx training (Graph Mode)...")
            for epoch in range(5):
                epoch_loss = 0.0
                num_batches = 0
                for batch_X, batch_y in dataset:
                    loss = train_step(batch_X, batch_y, global_trainable_weights)
                    epoch_loss += loss
                    num_batches += 1
            print(f"[Client {self.cid}] FedProx Training Complete.")

        else:
            # ---- Standard FedAvg local training ----
            self.model.fit(
                X_train_final, y_train_final,
                epochs=5, batch_size=self.batch_size, verbose=2,
                class_weight=self.class_weight_dict
            )

        # ------------------------------------------------------------------
        # WEIGHT-LEVEL ATTACKS (applied after local training)
        # ------------------------------------------------------------------
        final_weights = self.model.get_weights()

        # ------------------------------------------------------------------
        # ATTACK 3: MODEL POISONING — Gaussian Noise Injection
        # Add zero-mean Gaussian noise (std = scale) to every weight tensor.
        # Degrades model quality proportional to scale; easily detected by
        # norm-bounding defences at large scale values.
        # ------------------------------------------------------------------
        if self.attack_type == "noise":
            print(f"[Client {self.cid}] [Attack] Executing Gaussian Noise Injection (std={self.scale})...")
            final_weights = [w + np.random.normal(0, self.scale, w.shape)
                             for w in final_weights]
            print(f"[Client {self.cid}] Noise injected into {len(final_weights)} weight tensors.")

        # ------------------------------------------------------------------
        # ATTACK 4: MODEL REPLACEMENT — Byzantine / Scaling Attack
        # The attacker multiplies its weight *update* (delta from the global
        # model) by `scale` before submission. A large scale factor causes
        # the attacker's update to dominate the FedAvg aggregate, effectively
        # steering the global model toward the attacker's objective.
        #
        # delta_i = w_local - w_global
        # submitted = w_global + scale * delta_i
        # ------------------------------------------------------------------
        elif self.attack_type == "byzantine":
            print(f"[Client {self.cid}] [Attack] Executing Byzantine / Model Replacement Attack (scale={self.scale})...")
            global_weights = parameters   # original global weights received at round start
            amplified_weights = []
            for w_local, w_global in zip(final_weights, global_weights):
                delta = w_local - w_global
                amplified_weights.append(w_global + self.scale * delta)
            final_weights = amplified_weights
            print(f"[Client {self.cid}] Update delta amplified by x{self.scale}.")

        # ------------------------------------------------------------------
        # ATTACK 5: ADAPTIVE POISONING — Constrain-and-Scale
        # Extends the local update with a few extra gradient-ascent steps on
        # the cross-entropy loss (maximising classification error). The final
        # update is then clipped to the L2-norm of an honest update so it
        # stays within typical update bounds and evades norm-based defences.
        #
        # Reference: Bagdasaryan et al., "How To Backdoor Federated Learning"
        #            (AISTATS 2020) — constrain-and-scale framework.
        # ------------------------------------------------------------------
        elif self.attack_type == "adaptive":
            print(f"[Client {self.cid}] [Attack] Executing Adaptive Poisoning (Constrain-and-Scale)...")
            extra_steps = max(1, int(self.scale * 10))  # e.g. scale=1.0 -> 10 ascent steps

            # Compute honest update norm for the clipping bound
            global_weights = parameters
            honest_deltas = [w_local - w_global
                             for w_local, w_global in zip(final_weights, global_weights)]
            honest_norm = float(np.sqrt(sum(
                np.sum(d ** 2) for d in honest_deltas
            )))

            # Gradient ASCENT: maximise cross-entropy to corrupt the global model
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
            optimizer_adv = tf.keras.optimizers.Adam(learning_rate=1e-4)
            adv_dataset = (
                tf.data.Dataset.from_tensor_slices((X_train_final, y_train_final))
                .shuffle(1024).batch(self.batch_size).take(extra_steps)
            )
            for batch_X, batch_y in adv_dataset:
                with tf.GradientTape() as tape:
                    preds = self.model(batch_X, training=True)
                    # Negative loss = gradient ASCENT
                    loss_val = -loss_fn(batch_y, preds)
                grads = tape.gradient(loss_val, self.model.trainable_variables)
                optimizer_adv.apply_gradients(
                    zip(grads, self.model.trainable_variables)
                )

            adv_weights = self.model.get_weights()
            adv_deltas = [w_adv - w_global
                          for w_adv, w_global in zip(adv_weights, global_weights)]
            adv_norm = float(np.sqrt(sum(np.sum(d ** 2) for d in adv_deltas)))

            # Clip adversarial update to honest norm (stealthiness constraint)
            if adv_norm > honest_norm and adv_norm > 0:
                clip_ratio = honest_norm / adv_norm
                adv_deltas = [d * clip_ratio for d in adv_deltas]
                print(f"[Client {self.cid}] Adaptive: clipped update norm "
                      f"{adv_norm:.4f} -> {honest_norm:.4f} (ratio={clip_ratio:.4f}).")

            final_weights = [w_global + d
                             for w_global, d in zip(global_weights, adv_deltas)]
            print(f"[Client {self.cid}] Adaptive poisoning complete ({extra_steps} ascent steps).")

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
    parser = argparse.ArgumentParser(
        description="Malicious Federated Learning Client for FLEX-ID adversarial evaluation."
    )
    parser.add_argument("--cid", type=int, required=True,
                        help="Client ID (integer, 0-indexed).")
    parser.add_argument(
        "--attack_type", type=str, default="none",
        choices=["none", "flip", "noise", "backdoor", "byzantine", "adaptive"],
        help=(
            "Attack strategy to execute:\n"
            "  none      - Honest client (no attack).\n"
            "  flip      - Label flipping: relabels scale*100%% of samples as Benign.\n"
            "  noise     - Gaussian weight noise: adds N(0, scale) to model weights.\n"
            "  backdoor  - Trigger injection: stamps a feature trigger and relabels\n"
            "              scale*100%% of attack samples as Benign.\n"
            "  byzantine - Model replacement: amplifies weight update delta by scale.\n"
            "  adaptive  - Constrain-and-scale: gradient ascent then norm clipping."
        )
    )
    parser.add_argument("--scale", type=float, default=1.0,
                        help=("Attack intensity. Meaning varies by attack_type:\n"
                              "  flip/backdoor -> fraction of samples to poison (0.0-1.0)\n"
                              "  noise         -> Gaussian noise std deviation\n"
                              "  byzantine     -> delta amplification factor\n"
                              "  adaptive      -> number of gradient ascent steps = scale*10"))
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Mini-batch size for local training.")
    parser.add_argument("--fast_run", action="store_true",
                        help="Use 10%% of data for quick debugging runs.")
    # Backdoor-specific parameters
    parser.add_argument("--trigger_feature_idx", type=int, default=0,
                        help="Feature column index to use as the backdoor trigger.")
    parser.add_argument("--trigger_value", type=float, default=999.0,
                        help="Value stamped onto trigger_feature_idx for backdoor poisoning.")
    # Backward compatibility
    parser.add_argument("--malicious", action="store_true",
                        help="Legacy flag — equivalent to --attack_type flip.")

    args = parser.parse_args()

    # Handle legacy flag
    if args.malicious and args.attack_type == "none":
        args.attack_type = "flip"

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=MaliciousClient(
            cid=args.cid,
            attack_type=args.attack_type,
            scale=args.scale,
            batch_size=args.batch_size,
            fast_run=args.fast_run,
            trigger_feature_idx=args.trigger_feature_idx,
            trigger_value=args.trigger_value,
        )
    )