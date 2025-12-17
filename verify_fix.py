from client import FLClient
import numpy as np
import os

# Mock the arguments
CID = 3
BATCH_SIZE = 1024

print(f"--- 🧪 Starting Verification for Client {CID} ---")

try:
    # Initialize Client
    client = FLClient(cid=CID, batch_size=BATCH_SIZE)
    
    # Get initial weights
    print("\n[VERIFY] Getting initial parameters...")
    mock_config = {}
    params = client.get_parameters(mock_config)
    
    # Run fit() locally
    print("\n[VERIFY] Running fit() (Training Loop)...")
    # We expect 3 epochs and validation on REAL test data
    updated_params, num_examples, metrics = client.fit(params, mock_config)
    
    print("\n[VERIFY] ✅ fit() completed successfully.")
    print(f"[VERIFY] Metrics: {metrics}")
    
    # Check if sanity check matches validation accuracy (rough check)
    # The logs will show 'val_accuracy' (on X_test) and 'Local Test Sanity Check' (on X_test)
    # They should be IDENTICAL or very close.
    
except Exception as e:
    print(f"\n[VERIFY] ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
