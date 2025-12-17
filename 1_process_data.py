import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import os

print("Starting CSE-CIC-IDS2018 preprocessing...")
start_time = time.time()

# --- CONFIGURATION ---
INPUT_FILE = 'combined_ids2018_raw.csv'
OUTPUT_FILE = 'processed_data.csv'

# --- 1. Load Data ---
if not os.path.exists(INPUT_FILE):
    print(f"Error: '{INPUT_FILE}' not found.")
    exit()

# Read CSV (Use low_memory=False to handle mixed types warning)
df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"Original shape: {df.shape}")

# --- 2. Clean Column Names ---
df.columns = df.columns.str.strip()

# --- 3. Feature Selection ---
# Keeping the most relevant features for Cloud IDS
important_features = [
    'Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 
    'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Mean', 
    'Bwd Pkt Len Max', 'Bwd Pkt Len Mean', 'Flow Byts/s', 'Flow Pkts/s', 
    'Flow IAT Mean', 'Flow IAT Max', 'Fwd IAT Mean', 'Bwd IAT Mean', 
    'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s', 
    'Pkt Len Mean', 'Pkt Len Max', 'Pkt Len Var', 'SYN Flag Cnt', 
    'RST Flag Cnt', 'ACK Flag Cnt', 'Init Fwd Win Byts', 'Init Bwd Win Byts', 
    'Label' 
]

# Filter dataset
existing_cols = [col for col in important_features if col in df.columns]
df = df[existing_cols]
print(f"Shape after Feature Selection: {df.shape}")

# --- 4. Robust Data Cleaning (THE FIX) ---

# A. Handle Label Column Name
# Sometimes it's 'Label' or 'label'
label_col = 'Label'
if 'Label' not in df.columns:
    if 'label' in df.columns:
        label_col = 'label'
        df.rename(columns={'label': 'Label'}, inplace=True)

# B. Force Numeric Conversion (Coerce errors to NaN)
print("Converting columns to numeric...")
cols_to_normalize = [col for col in df.columns if col != 'Label']

for col in cols_to_normalize:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# C. Explicitly Replace Infinity with NaN
print("Sanitizing Infinity values...")
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# D. Drop NaN values
print("Dropping NaN values...")
df.dropna(inplace=True)

# E. Final Safety Check
# Ensure all data in features is finite (not infinite)
if not np.all(np.isfinite(df[cols_to_normalize])):
    print("⚠️ Warning: Infinite values still detected! Attempting second cleanup...")
    df = df[np.isfinite(df[cols_to_normalize]).all(1)]

print(f"Shape after Cleaning: {df.shape}")

if df.shape[0] == 0:
    print("❌ Error: All data was removed during cleaning. Check input file.")
    exit()

# --- 5. Normalization (MinMax Scaling) ---
y = df['Label']
X = df.drop(columns=['Label'])

print("Scaling data...")
scaler = MinMaxScaler()

# This is where it failed before. Now X is guaranteed clean.
try:
    X_scaled = scaler.fit_transform(X)
except ValueError as e:
    print(f"❌ Scaling Failed: {e}")
    # Last ditch debugging
    print("Max values in X:\n", X.max())
    exit()

# Re-create DataFrame with scaled features
df_processed = pd.DataFrame(X_scaled, columns=X.columns)

# Add the Label column back (reset index to match)
df_processed['label'] = y.values 

# --- 6. Save ---
df_processed.to_csv(OUTPUT_FILE, index=False)

end_time = time.time()
print(f"--- Preprocessing Complete ({end_time - start_time:.2f}s) ---")
print(f"Saved to: {OUTPUT_FILE}")
print(f"Final Class Distribution:\n{df_processed['label'].value_counts()}")