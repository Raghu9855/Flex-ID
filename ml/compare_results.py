import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

# --- 1. IMPORT YOUR MODEL FUNCTION ---
try:
    from model import create_dnn_model
except ImportError:
    print("[ERR] Error: model.py not found. Please place it in the same directory.")
    exit()

# --- 2. CONFIGURATION ---
# UPDATE THESE PATHS TO MATCH YOUR ACTUAL FOLDER STRUCTURE
# Note: Server saved them in 'fedavgeachround', not 'fedavg'
# Defaults (Overridden by CLI args)
# FEDAVG_PATH = "fedavgeachround/round-10-weights.pkl"   
# FEDPROX_PATH = "fedproxeachround/round-10-weights.pkl"
DATA_PATH = "data/processed_data.csv"
LE_PATH = "data/label_encoder.pkl"

def load_and_process_data():
    """
    Loads data and applies the EXACT same label encoding as training.
    """
    print("Loading data from CSV...")
    if not os.path.exists(DATA_PATH):
        print(f"[ERR] Error: {DATA_PATH} not found.")
        exit()
        
    df = pd.read_csv(DATA_PATH)
    
    # Identify the target column (string labels)
    string_cols = df.select_dtypes(include=['object']).columns
    if len(string_cols) == 0:
        print("[ERR] Error: No string label column found in CSV.")
        exit()
        
    target_col = string_cols[0] 
    print(f"Target column identified: '{target_col}'")

    # --- CRITICAL FIX: Load existing LabelEncoder ---
    if os.path.exists(LE_PATH):
        print(f"[OK] Loading existing LabelEncoder from {LE_PATH}")
        with open(LE_PATH, 'rb') as f:
            le = pickle.load(f)
    else:
        print("[WARN] Warning: label_encoder.pkl not found. Creating a new one (Risky!).")
        le = LabelEncoder()
        le.fit(df[target_col])

    # Transform labels
    # Handle unseen labels safely
    try:
        df[target_col] = le.transform(df[target_col])
    except ValueError:
        # If evaluation data has labels the model never saw, filter them out
        valid_labels = set(le.classes_)
        df = df[df[target_col].isin(valid_labels)]
        df[target_col] = le.transform(df[target_col])

    class_names = le.classes_
    print(f"Classes: {class_names}")

    y = df[target_col].values
    X = df.drop(columns=[target_col]).values
    
    # Split into training and testing (use a fixed seed to match previous logic logic)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_test.astype('float32'), y_test.astype('float32'), class_names

def plot_confusion_matrix(model, X_test, y_test, class_names, title="Confusion Matrix", suffix=""):
    print(f"\n--- Generating {title} ---")
    
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate metrics
    cm = confusion_matrix(y_test, y_pred, normalize='true') # Normalize to show percentages
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{title}\n(Confusion Matrix)')
    plt.tight_layout()
    
    filename = f"results/confusion_matrix_{title.replace(' ', '_').lower()}_{suffix}.png"
    plt.savefig(filename)
    print(f"[OK] Saved: {filename}")
    plt.close()

    print("\nClassification Report:")
    # Use output_dict=False to print string, or True to get data
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in class_names], zero_division=0))
    return y_pred

def evaluate_weights(weights_path, X_test, y_test, class_names, algorithm_name, suffix=""):
    print(f"\n--- Evaluating {algorithm_name} ---")
    
    if not os.path.exists(weights_path):
        print(f"[SKIP] Skipped: File not found ({weights_path})")
        return {"accuracy": 0, "report": None}

    try:
        # Load Weights
        with open(weights_path, 'rb') as f:
            weights = pickle.load(f)
            
        # Recreate Model
        input_dim = X_test.shape[1]
        num_classes = len(class_names)
        
        model = create_dnn_model(input_shape=input_dim, num_classes=num_classes)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Set Weights
        model.set_weights(weights)

        # Evaluate
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"[OK] {algorithm_name} Global Accuracy: {accuracy * 100:.2f}%")
        
        y_pred = plot_confusion_matrix(model, X_test, y_test, class_names, title=algorithm_name, suffix=suffix)
        # Get dict for JSON saving
        report_dict = classification_report(y_test, y_pred, target_names=[str(c) for c in class_names], zero_division=0, output_dict=True)
        
        return {"accuracy": accuracy * 100, "report": report_dict}

    except Exception as e:
        print(f"[ERR] Error during evaluation: {e}")
        return {"accuracy": 0, "report": None}

# --- MAIN ---
if __name__ == "__main__":
    import argparse
    import json
    import sys

    # Reduce log noise
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    tf.get_logger().setLevel('ERROR')

    parser = argparse.ArgumentParser()
    parser.add_argument("--fedavg", type=str, required=True, help="Path to FedAvg weights")
    parser.add_argument("--fedprox", type=str, required=True, help="Path to FedProx weights")
    parser.add_argument("--mode", type=str, default="custom", help="Mode name for context (no_attack / under_attack)")
    args = parser.parse_args()

    mode_suffix = args.mode.replace(" ", "_").lower()
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # Redefine plot helper inside main or just ensure it uses correct path
    # Actually, verify evaluate_weights calls plot_confusion_matrix with a filename argument or title?
    # I need to pass the suffix or modify plot_confusion_matrix to take a save path.
    # To minimize diff, I'll monkey patch or modify evaluate_weights call slightly? 
    # Better to modify the function.
    
    # ... I will modify the functions too in a separate block or here if I can view them. 
    # I can't easily modify the functions above from here without seeing them. 
    # But I can redefine them or just change the calls if I pass arguments?
    # Wait, functions are outside. I'll stick to modifying main here and ask for another edit for functions if needed.
    # Actually, I can use a global variable or change the signature.
    # Changing signature is cleaner. 
    
    # NOTE: I am ONLY modifying the MAIN block here. I will do a multi_replace for functions next.
    
    results = {}

    try:
        X_test, y_test, class_names = load_and_process_data()
        
        # We need to tell the functions where to save.
        
        # Evaluate FedAvg
        print(f"Evaluating FedAvg: {args.fedavg}")
        fa_res = evaluate_weights(args.fedavg, X_test, y_test, class_names, "FedAvg", mode_suffix)
        results["fedavg"] = {
             "accuracy": fa_res["accuracy"],
             "report": fa_res["report"],
             "path": args.fedavg
        }

        # Evaluate FedProx
        print(f"Evaluating FedProx: {args.fedprox}")
        fp_res = evaluate_weights(args.fedprox, X_test, y_test, class_names, "FedProx", mode_suffix)
        results["fedprox"] = {
             "accuracy": fp_res["accuracy"],
             "report": fp_res["report"],
             "path": args.fedprox
        }

        results["success"] = True
        results["timestamp"] = pd.Timestamp.now().isoformat()
        results["mode"] = args.mode

    except Exception as e:
        results["success"] = False
        results["error"] = str(e)
        print(f"Global Error: {e}", file=sys.stderr)

    # SAVE TO FILE
    output_filename = f"results/comparison_results_{mode_suffix}.json"
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"[OK] Results saved to {output_filename}")