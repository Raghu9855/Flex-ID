# plot history
import matplotlib.pyplot as plt
import pickle
import os

# --- CONFIGURATION ---
# Normal History
FEDAVG_HIST = "results/fedavg_history.pkl"
FEDPROX_HIST = "results/fedprox_history.pkl"

# Under Attack History
FEDAVG_ATTACK_HIST = "results/fedavg_underattack_history.pkl"
FEDPROX_ATTACK_HIST = "results/fedprox_underattack_history.pkl"

def load_data(filepath):
    """
    Loads the history file and extracts (Round, Loss, Accuracy, F1).
    """
    if not os.path.exists(filepath):
        # Silent warning to avoid clutter if just one mode is run
        return [], [], [], []
    
    with open(filepath, "rb") as f:
        history = pickle.load(f)
        
    rounds = []
    losses = []
    accuracies = []
    f1s = []
    
    for entry in history:
        rounds.append(entry.get('round'))
        losses.append(entry.get('train_loss'))
        accuracies.append(entry.get('accuracy'))
        f1s.append(entry.get('f1'))
            
    return rounds, losses, accuracies, f1s

def plot_graph(r1, l1, a1, f1, r2, l2, a2, f2, label1, label2, save_name, title_suffix=""):
    if not r1 and not r2:
        return

    # Create Subplots (Loss, Accuracy, F1)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Loss Plot
    if r1: axs[0].plot(r1, l1, label=label1, color='blue', marker='o')
    if r2: axs[0].plot(r2, l2, label=label2, color='red', linestyle='--', marker='x')
    axs[0].set_title(f'Training Loss {title_suffix}')
    axs[0].set_xlabel('Round')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)
    
    # 2. Accuracy Plot
    if r1: axs[1].plot(r1, a1, label=label1, color='blue', marker='o')
    if r2: axs[1].plot(r2, a2, label=label2, color='red', linestyle='--', marker='x')
    axs[1].set_title(f'Global Accuracy {title_suffix}')
    axs[1].set_xlabel('Round')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].grid(True)

    # 3. F1 Plot
    if r1: axs[2].plot(r1, f1, label=label1, color='blue', marker='o')
    if r2: axs[2].plot(r2, f2, label=label2, color='red', linestyle='--', marker='x')
    axs[2].set_title(f'Global F1 Score {title_suffix}')
    axs[2].set_xlabel('Round')
    axs[2].set_ylabel('F1 Score')
    axs[2].legend()
    axs[2].grid(True)
    
    plt.tight_layout()
    save_path = f"results/{save_name}"
    plt.savefig(save_path, dpi=300)
    print(f"[OK] Graph saved as: {save_path}")
    # plt.show() # backend usually runs this, so show() might block

def plot_comparison():
    print("Loading history files...")
    
    # 1. Plot Normal Comparison
    r_avg, l_avg, a_avg, f_avg = load_data(FEDAVG_HIST)
    r_prox, l_prox, a_prox, f_prox = load_data(FEDPROX_HIST)
    
    if r_avg or r_prox:
        plot_graph(
            r_avg, l_avg, a_avg, f_avg, 
            r_prox, l_prox, a_prox, f_prox, 
            "FedAvg", "FedProx", 
            "comparison_metrics.png"
        )
    
    # 2. Plot Under Attack Comparison
    r_avg_att, l_avg_att, a_avg_att, f_avg_att = load_data(FEDAVG_ATTACK_HIST)
    r_prox_att, l_prox_att, a_prox_att, f_prox_att = load_data(FEDPROX_ATTACK_HIST)
    
    if r_avg_att or r_prox_att:
        print("Found Under Attack history. Generating specific plot...")
        plot_graph(
            r_avg_att, l_avg_att, a_avg_att, f_avg_att, 
            r_prox_att, l_prox_att, a_prox_att, f_prox_att, 
            "FedAvg (Attack)", "FedProx (Attack)", 
            "comparison_metrics_underattack.png",
            title_suffix="(Under Attack)"
        )
    else:
        print("No 'Under Attack' history found. Skipping attack plot.")

    # 3. Save Summary to JSON
    summary = {}
    
    def extract_stats(r, l, a, f):
        if not r: return None
        return {
            "best_accuracy": f"{max(a)*100:.2f}%" if a else "N/A",
            "best_f1": f"{max(f):.4f}" if f else "N/A",
            "last_accuracy": f"{a[-1]*100:.2f}%" if a else "N/A",
            "last_loss": f"{l[-1]:.4f}" if l else "N/A",
            "last_f1": f"{f[-1]:.4f}" if f else "N/A",
            "rounds": len(r)
        }
    
    summary["fedavg"] = extract_stats(r_avg, l_avg, a_avg, f_avg)
    summary["fedprox"] = extract_stats(r_prox, l_prox, a_prox, f_prox)
    summary["fedavg_attack"] = extract_stats(r_avg_att, l_avg_att, a_avg_att, f_avg_att)
    summary["fedprox_attack"] = extract_stats(r_prox_att, l_prox_att, a_prox_att, f_prox_att)

    import json
    with open("results/metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    print(f"[OK] Metrics summary saved to results/metrics_summary.json")

if __name__ == "__main__":
    plot_comparison()