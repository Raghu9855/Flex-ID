import shap
import numpy as np
import pandas as pd
import pickle
import os
import sys
import matplotlib
matplotlib.use('Agg') # Safe for non-interactive environments
import matplotlib.pyplot as plt

# Import your model
try:
    from model import create_dnn_model
except ImportError:
    sys.exit("❌ Critical Error: 'model.py' not found.")

# --- CONFIGURATION ---
import argparse

# --- CONFIGURATION (Defaults) ---
DEFAULT_WEIGHTS_PATH = "fedavgeachround/round-5-weights.pkl" 
DATA_PATH = "processed_data.csv"
LE_PATH = "label_encoder.pkl"

def run_explanation(round_num=None, weights_file=None):
    # Determine weights path
    if weights_file:
        weights_path = weights_file
    elif round_num:
        # Try both folders
        p1 = f"fedavgeachround/round-{round_num}-weights.pkl"
        p2 = f"fedproxeachround/round-{round_num}-weights.pkl"
        if os.path.exists(p1): weights_path = p1
        elif os.path.exists(p2): weights_path = p2
        else:
            print(f"❌ Weights for round {round_num} not found in fedavg/fedprox folders.")
            return
    else:
        weights_path = DEFAULT_WEIGHTS_PATH

    print(f"--- 1. Loading Global Model from: {weights_path} ---")
    if not os.path.exists(DATA_PATH) or not os.path.exists(LE_PATH):
        sys.exit("❌ Error: Data or Label Encoder not found.")
    
    if not os.path.exists(weights_path):
        sys.exit(f"❌ Error: Weights file not found: {weights_path}")

    df = pd.read_csv(DATA_PATH)
    
    # Load Encoder
    with open(LE_PATH, "rb") as f:
        le = pickle.load(f)
    print(f"✔ Classes: {len(le.classes_)}")

    # Identify Label Column
    target_col = None
    for name in ['label', 'Label', 'class', 'Class', 'attack_cat']:
        if name in df.columns:
            target_col = name
            break
    if not target_col: target_col = df.columns[-1]

    # Clean Data (Match Encoder)
    df[target_col] = df[target_col].astype(str)
    valid_labels = set([str(x) for x in le.classes_])
    df = df[df[target_col].isin(valid_labels)]
    
    # Safe Transform
    def safe_transform(x):
        try: return le.transform([x])[0]
        except: return 0
    df[target_col] = df[target_col].apply(safe_transform)

    # Prepare X (Features) and y (Labels)
    X = df.drop(columns=[target_col])
    feature_names = X.columns.tolist()
    X_values = X.values.astype(np.float32)

    # Sample Data (Background = 50, Test = 10)
    # Using small numbers because KernelExplainer is slow but accurate
    background = X_values[np.random.choice(X_values.shape[0], 50, replace=False)]
    to_explain = X_values[np.random.choice(X_values.shape[0], 10, replace=False)]

    print(f"--- 2. Building Model & Loading Weights ---")
    model = create_dnn_model(X.shape[1], len(le.classes_))
    with open(weights_path, "rb") as f:
        weights = pickle.load(f)
    model.set_weights(weights)

    # --- 3. Wrapper Function (The Fix for TF Issues) ---
    def model_predict(data):
        return model.predict(data, verbose=0)

    print("--- 3. Running SHAP (KernelExplainer) ---")
    print("ℹ️ Using KernelExplainer (Robust mode).")
    
    explainer = shap.KernelExplainer(model_predict, background)
    shap_values = explainer.shap_values(to_explain)

    # --- 4. Generate Plot ---
    print("\n📊 Generating Summary Plot...")
    
    # Handle Multi-class output (SHAP returns list of arrays)
    if isinstance(shap_values, list):
         # Pick the class with highest importance or just the Attack class (usually index 1)
         class_idx = 1 if len(shap_values) > 1 else 0
         shap_data = shap_values[class_idx]
         print(f"   -> Plotting for Class Index {class_idx} ({le.classes_[class_idx]})")
    else:
        shap_data = shap_values

    plt.figure()
    shap.summary_plot(shap_data, to_explain, feature_names=feature_names, show=False)
    
    out_name = "shap_summary_plot.png"
    if round_num:
        out_name = f"shap_summary_round_{round_num}.png"
        
    plt.savefig(out_name, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved XAI Plot: {out_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, help="Round number to explain (looks in fedavg/fedprox folders)")
    parser.add_argument("--weights", type=str, help="Direct path to weights file")
    args = parser.parse_args()
    
    run_explanation(args.round, args.weights)