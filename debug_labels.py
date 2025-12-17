import pickle
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

def load_partition(cid):
    with open(f"client_partition_{cid}.pkl", "rb") as f:
        return pickle.load(f)

def check():
    print("--- Debugging Client 0 Data ---")
    if not os.path.exists("label_encoder.pkl"):
        print("❌ label_encoder.pkl missing!")
        return

    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    print(f"Global Classes ({len(le.classes_)}): {le.classes_}")

    (X_train, y_train), (X_test, y_test) = load_partition(0)
    y_test = np.array(y_test).reshape(-1)
    
    print(f"Test Set Size: {len(y_test)}")
    unique, counts = np.unique(y_test, return_counts=True)
    
    print("\nTest Set Distribution:")
    for u, c in zip(unique, counts):
        try:
            label_name = le.inverse_transform([u])[0]
        except:
            label_name = "UNKNOWN_TO_ENCODER"
        print(f"  Class {u} ({label_name}): {c} samples ({c/len(y_test)*100:.1f}%)")
        
    # Check if 'Benign' exists
    if 'Benign' in le.classes_:
        benign_idx = int(le.transform(['Benign'])[0])
        print(f"\nBenign Class Index: {benign_idx}")
    else:
        print("\n'Benign' label not found in Global Encoder.")

if __name__ == "__main__":
    check()
