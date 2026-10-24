# Results

This folder contains the raw results of all experiments reported in the paper.

| Folder | Content |
|---|---|
| `results_ok/` | Seeds 0–4: all configurations, controls (CE two-stage, frozen ImageNet) and re-trained architectures (ResNet50, DenseNet121, LE-CNN) |
| `results_run2/` | Seeds 5–9: baseline, SCL and CE two-stage (with their pruned versions) |

Each folder contains:

- `runs.csv`: one row per model and seed (accuracy, precision, recall, F1-score, sparsity, model sizes, training budget, and for the first seed FLOPs, latency and memory)
- `y_test.npy`: true labels of the 518 test images
- `pred_<config>_seed<N>.npy`: predicted labels of each model and seed
- `summary.md`: summary tables and statistical tests

Trained model files (`*.h5`) are not included because of their size.

To regenerate the tables and statistics:

```bash
python scripts/arts_multiseed.py --summarize --out results/results_ok --merge results/results_run2
```

To regenerate the confusion matrices and per-class TP/TN/FP/FN tables:

```bash
python scripts/arts_confusion.py --dirs results/results_ok results/results_run2 --out figures/confusion --show-zeros
```
