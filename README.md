# Supervised Contrastive Learning with Magnitude-Based Pruning for Arabic Traffic Sign Recognition

Code accompanying the paper

> T. Nugroho and M. Bilal, "Supervised Contrastive Learning with Magnitude-Based Pruning for Arabic Traffic Sign Recognition," *Applied Sciences*, 2026 (under review).

The repository contains everything needed to reproduce the multi-seed experiments, statistics, tables and figures of the revised manuscript on the Arabic Traffic Signs (ArTS) dataset.

## What is evaluated

| Configuration | Description |
|---|---|
| `baseline` | ResNet50V2 (ImageNet), encoder + classifier trained end-to-end with cross-entropy |
| `scl` | Two-stage supervised contrastive learning: stage 1 encoder + projection head (SupCon loss, τ = 0.1, 50 epochs); stage 2 frozen encoder + classifier |
| `ce_two_stage` | Control: same two-stage procedure and schedule as `scl`, but cross-entropy with a linear head in stage 1 |
| `frozen_imagenet` | Control: frozen ImageNet encoder, only the classifier is trained |
| `pruned_*` | Magnitude-based pruning of the Dense layers (30% → 80% sparsity over 20 epochs, 30 epochs in total, model selected by lowest validation loss after 80% sparsity) |
| `resnet_v1`, `densenet121`, `le_cnn` | Previously reported architectures re-trained under the same protocol |

All main configurations were run with ten random seeds (0–9); controls and re-trained architectures with five seeds (0–4).

## Repository structure

```
scripts/
  arts_multiseed.py         training, evaluation, size/FLOPs/latency/memory, statistics
  plot_training_curves.py   training/validation curves from the console logs
  arts_confusion.py         confusion matrices, per-class TP/TN/FP/FN, error examples
figures/diagrams/           editable draw.io diagrams of the framework figures
data/                       place the ArTS dataset here (not included, see data/README.md)
results/                    result folders produced by arts_multiseed.py
logs/                       console logs of the training runs (used for the curves)
```

## Installation

Tested with Python 3.9, TensorFlow 2.10 and TensorFlow Model Optimization 0.8 on Windows 11 with an NVIDIA RTX 3060 Laptop GPU.

```bash
conda create -n tfgpu python=3.9
conda activate tfgpu
pip install -r requirements.txt
```

## Dataset

Download the ArTS dataset and arrange it as described in [`data/README.md`](data/README.md). The training set (2,200 images) is split once, stratified with a fixed seed, into 1,650 training and 550 validation images; the 518 test images are kept unchanged.

## Reproducing the experiments

Run all commands from the repository root.

```bash
# 1. quick smoke test (1 seed, 1-2 epochs, checks that everything runs)
python scripts/arts_multiseed.py --smoke

# 2. main configurations and controls, seeds 0-4 (save the console output as a log!)
python scripts/arts_multiseed.py --seeds 0 1 2 3 4 --out results/results_ok > logs/log1.txt

# 3. additional seeds 5-9 for the main configurations
python scripts/arts_multiseed.py --seeds 5 6 7 8 9 --out results/results_run2 --groups baseline scl ce_two_stage --no-cost > logs/log2.txt

# 4. previously reported architectures under the same protocol
python scripts/arts_multiseed.py --seeds 0 1 2 3 4 --out results/results_ok --groups resnet_v1 densenet121 le_cnn
```

Results are appended to `<out>/runs.csv` after every model, so an interrupted run can be restarted with the same command. Latency, FLOPs and memory are measured for the first seed only.

## Tables, statistics and figures (no retraining)

```bash
# summary tables, paired t-tests, Wilcoxon tests, pruning effects -> results/results_ok/summary.md
python scripts/arts_multiseed.py --summarize --out results/results_ok --merge results/results_run2

# training and validation curves (Figures 7 and 8)
python scripts/plot_training_curves.py logs/log1.txt logs/log2.txt --out figures/curves

# confusion matrices and per-class TP/TN/FP/FN (Error Analysis)
python scripts/arts_confusion.py --dirs results/results_ok results/results_run2 --out figures/confusion --show-zeros
```

Optional: add `--examples --script scripts/arts_multiseed.py` to `arts_confusion.py` to create a figure with misclassified and correctly rejected test images.

## Main results (ten seeds, mean ± SD)

| Configuration | Accuracy (%) |
|---|---|
| End-to-end baseline | 98.47 ± 0.28 |
| Pruned baseline | 99.07 ± 0.27 |
| SCL two-stage | 98.90 ± 0.50 |
| Pruned SCL | 99.00 ± 0.54 |
| CE two-stage (control) | 98.98 ± 0.34 |

SCL outperformed the end-to-end baseline (+0.42 pp, paired t-test p = 0.016), but the CE two-stage control reached a similar accuracy (p = 0.53 versus SCL), indicating that the improvement is associated with the two-stage training procedure. Pruning 80% of the Dense-layer weights maintained accuracy but reduced the compressed model size by only 3.4% and did not reduce latency; int8 quantisation reduced the model size from 99.35 to 25.48 MB. See the paper for details.

## Citation

```bibtex
@article{nugroho2026arts,
  author  = {Nugroho, Triawan and Bilal, Muhammad},
  title   = {Supervised Contrastive Learning with Magnitude-Based Pruning for Arabic Traffic Sign Recognition},
  journal = {Applied Sciences},
  year    = {2026},
  note    = {Under review}
}
```

## License

This code is released under the MIT License (see `LICENSE`). The ArTS dataset is subject to its own license and is not redistributed here.
