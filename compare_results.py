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
    print("❌ Error: model.py not found. Please place it in the same directory.")
    exit()

# --- 2. CONFIGURATION ---
# UPDATE THESE PATHS TO MATCH YOUR ACTUAL FOLDER STRUCTURE
# Note: Server saved them in 'fedavgeachround', not 'fedavg'
FEDAVG_PATH = "fedavgeachround/round-5-weights.pkl"   
FEDPROX_PATH = "fedproxeachround/round-5-weights.pkl"
DATA_PATH = "processed_data.csv"
LE_PATH = "label_encoder.pkl"

def load_and_process_data():
    """
    Loads data and applies the EXACT same label encoding as training.
    """
    print("Loading data from CSV...")
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found.")
        exit()
        
    df = pd.read_csv(DATA_PATH)
    
    # Identify the target column (string labels)
    string_cols = df.select_dtypes(include=['object']).columns
    if len(string_cols) == 0:
        print("❌ Error: No string label column found in CSV.")
        exit()
        
    target_col = string_cols[0] 
    print(f"Target column identified: '{target_col}'")

    # --- CRITICAL FIX: Load existing LabelEncoder ---
    if os.path.exists(LE_PATH):
        print(f"✔ Loading existing LabelEncoder from {LE_PATH}")
        with open(LE_PATH, 'rb') as f:
            le = pickle.load(f)
    else:
        print("⚠ Warning: label_encoder.pkl not found. Creating a new one (Risky!).")
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

def plot_confusion_matrix(model, X_test, y_test, class_names, title="Confusion Matrix"):
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
    
    filename = f"confusion_matrix_{title.replace(' ', '_').lower()}.png"
    plt.savefig(filename)
    print(f"✔ Saved: {filename}")
    plt.close()

    print("\nClassification Report:")
    # Use output_dict=False to print string, or True to get data
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in class_names], zero_division=0))

def evaluate_weights(weights_path, X_test, y_test, class_names, algorithm_name):
    print(f"\n--- Evaluating {algorithm_name} ---")
    
    if not os.path.exists(weights_path):
        print(f"❌ Skipped: File not found ({weights_path})")
        return None

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
        print(f"✔ {algorithm_name} Global Accuracy: {accuracy * 100:.2f}%")
        
        plot_confusion_matrix(model, X_test, y_test, class_names, title=algorithm_name)
        return accuracy * 100

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return None

# --- MAIN ---
if __name__ == "__main__":
    X_test, y_test, class_names = load_and_process_data()
    
    evaluate_weights(FEDAVG_PATH, X_test, y_test, class_names, "FedAvg")
    
    # Only runs if FedProx file exists
    evaluate_weights(FEDPROX_PATH, X_test, y_test, class_names, "FedProx")