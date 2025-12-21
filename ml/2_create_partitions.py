# create partitions
import sys
# Fix for Windows Unicode errors
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os

NUM_CLIENTS = 4
NON_IID_ALPHA = 0.4        # 40% -> client 0
SOFT_NON_IID = 0.2         # rest attacks distributed

print("Loading processed data...")

df = pd.read_csv('data/processed_data.csv')

label_col = 'label' if 'label' in df.columns else 'Label'
le = LabelEncoder()
df[label_col] = le.fit_transform(df[label_col].astype(str))

with open('data/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

benign_code = le.transform(['Benign'])[0]
df_benign = df[df[label_col] == benign_code]
df_attack = df[df[label_col] != benign_code]

df_benign = df_benign.sample(frac=1, random_state=42)
df_attack = df_attack.sample(frac=1, random_state=42)

print("Benign:", len(df_benign))
print("Attack:", len(df_attack))

# --------------- NEW NON-IID STRATEGY --------------------

partitions = []

# client 0 gets NON_IID_ALPHA share
primary_attacks = int(len(df_attack) * NON_IID_ALPHA)
client0_attacks = df_attack.iloc[:primary_attacks]

remaining_attacks = df_attack.iloc[primary_attacks:]
remaining_parts = np.array_split(remaining_attacks, NUM_CLIENTS - 1)

attack_parts = [client0_attacks] + list(remaining_parts)
benign_parts = np.array_split(df_benign, NUM_CLIENTS)

# --------------- SAVE CLIENT PARTITIONS --------------------

for i in range(NUM_CLIENTS):
    client_df = pd.concat([attack_parts[i], benign_parts[i]])
    client_df = client_df.sample(frac=1, random_state=42)

    y = client_df[label_col].values
    X = client_df.drop(columns=[label_col]).values

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    filename = f"data/client_partition_{i}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(((X_train, y_train), (X_test, y_test)), f)

    print(f"Saved {filename}")

print("DONE – Balanced NON-IID partitions created.")

# --------------- REPORT & PLOTS --------------------
import matplotlib.pyplot as plt

# Ensure results directory exists
if not os.path.exists("results"):
    os.makedirs("results")

report_lines = []
report_lines.append("Federated Learning Data Partition Report")
report_lines.append("======================================\n")

for i in range(NUM_CLIENTS):
    with open(f'data/client_partition_{i}.pkl', 'rb') as f:
        (X_train, y_train), (X_test, y_test) = pickle.load(f)

    # 1. Report Data
    total_samples = len(y_train) + len(y_test)
    report_lines.append(f"Client {i}:")
    report_lines.append(f"  - Total Samples: {total_samples}")
    report_lines.append(f"  - Train Set: {len(y_train)}")
    report_lines.append(f"  - Test Set: {len(y_test)}")

    # Distribution
    y_all = np.concatenate([y_train, y_test])
    unique, counts = np.unique(y_all, return_counts=True)
    names = le.inverse_transform(unique)
    dist_str = ", ".join([f"{n}: {c}" for n, c in zip(names, counts)])
    report_lines.append(f"  - Label Distribution: {dist_str}")
    report_lines.append("-" * 30 + "\n")

    # 2. Plot
    plt.figure(figsize=(10,5))
    plt.bar(names, counts, color='skyblue')
    plt.xticks(rotation=45)
    plt.title(f"Client {i} - Label Distribution")
    plt.tight_layout()
    plt.savefig(f"results/client_{i}_distribution.png")
    plt.close()

# Save Report
with open("results/partition_report.txt", "w") as f:
    f.write("\n".join(report_lines))

print("Report saved to results/partition_report.txt")
print("Plots saved to results/client_*_distribution.png")
