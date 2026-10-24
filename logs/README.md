# Training logs

This folder contains the console logs of the training runs, which are used by `scripts/plot_training_curves.py` to draw the training and validation curves (Figures 7 and 8 of the paper).

| File | Content |
|---|---|
| `log1.txt` | Seeds 0–4: all configurations, controls and re-trained architectures |
| `log2.txt` | Seeds 5–9: baseline, SCL and CE two-stage |

`log1.txt` begins with a smoke test and an interrupted run. The plotting script uses only the last complete run of each seed, so these entries are ignored automatically.

To reproduce the curves, run from the repository root (list the older log first):

```bash
python scripts/plot_training_curves.py logs/log1.txt logs/log2.txt --out figures/curves
```
