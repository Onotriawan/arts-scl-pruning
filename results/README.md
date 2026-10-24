# Results

Copy the result folders of the paper here:

```
results/results_ok/     seeds 0-4 (all configurations, controls, re-trained architectures)
results/results_run2/   seeds 5-9 (baseline, SCL, CE two-stage)
```

Each folder contains `runs.csv` (one row per model and seed), `y_test.npy`, `pred_<config>_seed<N>.npy` and `summary.md`. Model files (`*.h5`) are excluded by `.gitignore` because of their size.
