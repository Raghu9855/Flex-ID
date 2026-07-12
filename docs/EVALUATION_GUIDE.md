# FLEX-ID Evaluation Guide

---

## Metrics Reported

### Global Metrics

| Metric | Script | Reviewer concern |
|---|---|---|
| Accuracy | `compare_results.py` | Basic — can be misleading on imbalanced data |
| Balanced Accuracy | `compare_results.py` | Corrects for class imbalance |
| Weighted F1 | `compare_results.py` | Weighted by class support |
| Macro F1 | `compare_results.py` | Equal weight per class |
| ROC-AUC (macro, OvR) | `compare_results.py` | Separability across all classes |
| PR-AUC (macro) | `compare_results.py` | Precision-Recall trade-off |
| MCC | `compare_results.py` | Best single metric for imbalanced multi-class |
| Inference Time (ms) | `compare_results.py` | Practical deployment concern |

### Per-Class Metrics

`compare_results.py` outputs per-class Precision, Recall, and F1 for every
attack class into the JSON results file under `per_class_report`.

### Round-wise Metrics

`plot_history.py` saves `results/round_metrics_fedavg.csv` and
`results/round_metrics_fedprox.csv` with columns:

```
round, train_loss, accuracy, f1, strategy
```

These can be imported into LaTeX via `\input{round_metrics_fedavg.csv}` or
pasted into Excel for table generation.

---

## Running Evaluation

```bash
cd ml

python compare_results.py \
    --fedavg  results/fedavgeachround/round-30-weights.pkl \
    --fedprox results/fedproxeachround/round-30-weights.pkl \
    --mode    no_attack
```

Outputs:
- `results/comparison_results_no_attack.json` — full metric JSON
- `results/confusion_matrix_fedavg_no_attack.png` — 300 DPI heatmap
- `results/confusion_matrix_fedprox_no_attack.png`
- `results/comparison_metrics_no_attack.png` — bar chart

---

## Interpreting SHAP Results

```bash
python explain_model.py --round 30 --num_clients 4 --bg_size 100 --explain_size 50
```

Outputs:
- `results/FedAvg_client_*_shap.png` — Local per-client SHAP summary
- `results/FedAvg_global_shap.png` — Global aggregated feature importance
- `results/shap_summary.json` — Top-10 features + agreement metrics

### Agreement Metrics in `shap_summary.json`

| Metric | Meaning | Good Value |
|---|---|---|
| `kendall_tau_mean` | Mean Kendall's Tau across client pairs (-1 to 1) | > 0.5 |
| `spearman_rho_mean` | Mean Spearman's ρ across client pairs (-1 to 1) | > 0.6 |
| `stability_score` | Tau rescaled to [0, 1] | > 0.75 |
| `jaccard_top_k_mean` | Mean overlap of top-10 features across pairs | > 0.5 |

A high stability score (> 0.75) means clients agree on *which features matter*
even though they train on different local data distributions — supporting the
privacy-preserving federated SHAP claim.

---

## Communication Cost

`plot_history.py` estimates the communication cost per round:

```
cost_per_round = sum(layer.nbytes for layer in model_weights)
```

For 4 clients: `cost_per_round × 4` bytes uploaded + `cost_per_round`
bytes downloaded per round.

---

## Reproducibility Checklist

- [ ] `RANDOM_SEED = 42` set in all scripts
- [ ] `set_global_seeds(42)` called before any training
- [ ] Same `--num_clients` used in partition + server
- [ ] Same `--rounds` for all compared strategies
- [ ] Same `--proximal_mu` reported in paper (default: 0.1)
- [ ] SHAP `bg_size=100`, `explain_size=50` reported

---

## Metric Definitions

**Balanced Accuracy** = mean recall across all classes  
**MCC** = (TP×TN − FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))  
**ROC-AUC** (OvR macro) = mean AUC of each class treated as binary  
**PR-AUC** (macro) = mean average precision across classes  
**Macro F1** = mean F1 across classes (equal weight regardless of support)  
**Weighted F1** = F1 weighted by class support (number of true instances)
