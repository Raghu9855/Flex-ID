import pickle
import os

path = "results/fedavg_history.pkl"
if os.path.exists(path):
    try:
        with open(path, "rb") as f:
            history = pickle.load(f)
        
        # history is a list of dicts: {'round': r, 'train_loss': l, 'accuracy': a, 'f1': f}
        # Check if list is empty
        if not history:
             print("History file exists but is empty.")
             exit()

        # Extract accuracies that are not None
        accuracies = [h.get('accuracy') for h in history if h.get('accuracy') is not None]
        
        if accuracies:
            print(f"\n✅ Training Results Found:")
            print(f"   Max Accuracy: {max(accuracies)*100:.2f}%")
            print(f"   Final Accuracy: {accuracies[-1]*100:.2f}%")
            
            # Also show F1 if available
            f1s = [h.get('f1') for h in history if h.get('f1') is not None]
            if f1s:
                print(f"   Final F1 Score: {f1s[-1]:.4f}")
        else:
            print("⚠️ Training finished but no Global Accuracy data found in history.")
            print("   (This might happen if 'evaluate' steps were skipped or failed).")
            
    except Exception as e:
        print(f"❌ Error reading history: {e}")
else:
    print("⏳ Training still in progress (or failed). History file not found yet.")
    print("   Please wait for the server window to close.")
