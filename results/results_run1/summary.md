# Results over seeds

Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (n = 10);  test images: 518


## Classification, % (mean ± SD [95 % CI half-width] (min–max))

| Config | n | accuracy | precision | recall | f1 | f1_macro | errors |
|---|---|---|---|---|---|---|---|
| baseline | 10 | 98.47 ± 0.28 [CI ±0.20] (98.07–98.84) | 98.58 ± 0.25 [CI ±0.18] (98.19–98.93) | 98.47 ± 0.28 [CI ±0.20] (98.07–98.84) | 98.48 ± 0.28 [CI ±0.20] (98.07–98.85) | 98.46 ± 0.27 [CI ±0.20] (98.01–98.82) | 7.90 ± 1.45 [CI ±1.04] (6.00–10.00) |
| pruned_baseline | 10 | 99.07 ± 0.27 [CI ±0.19] (98.65–99.42) | 99.14 ± 0.24 [CI ±0.17] (98.76–99.46) | 99.07 ± 0.27 [CI ±0.19] (98.65–99.42) | 99.07 ± 0.27 [CI ±0.19] (98.66–99.42) | 99.02 ± 0.30 [CI ±0.21] (98.60–99.42) | 4.80 ± 1.40 [CI ±1.00] (3.00–7.00) |
| scl | 10 | 98.90 ± 0.50 [CI ±0.36] (98.07–99.61) | 98.96 ± 0.48 [CI ±0.35] (98.15–99.65) | 98.90 ± 0.50 [CI ±0.36] (98.07–99.61) | 98.90 ± 0.50 [CI ±0.36] (98.08–99.61) | 98.86 ± 0.53 [CI ±0.38] (97.95–99.58) | 5.70 ± 2.58 [CI ±1.85] (2.00–10.00) |
| pruned_scl | 10 | 99.00 ± 0.54 [CI ±0.38] (97.68–99.61) | 99.07 ± 0.46 [CI ±0.33] (98.00–99.65) | 99.00 ± 0.54 [CI ±0.38] (97.68–99.61) | 99.00 ± 0.52 [CI ±0.38] (97.73–99.61) | 98.94 ± 0.61 [CI ±0.44] (97.39–99.58) | 5.20 ± 2.78 [CI ±1.99] (2.00–12.00) |
| ce_two_stage | 10 | 98.98 ± 0.34 [CI ±0.24] (98.26–99.23) | 99.04 ± 0.33 [CI ±0.24] (98.33–99.31) | 98.98 ± 0.34 [CI ±0.24] (98.26–99.23) | 98.98 ± 0.34 [CI ±0.24] (98.26–99.23) | 98.96 ± 0.35 [CI ±0.25] (98.19–99.24) | 5.30 ± 1.77 [CI ±1.26] (4.00–9.00) |
| pruned_ce_two_stage | 10 | 98.96 ± 0.34 [CI ±0.25] (98.26–99.23) | 99.02 ± 0.31 [CI ±0.22] (98.38–99.30) | 98.96 ± 0.34 [CI ±0.25] (98.26–99.23) | 98.96 ± 0.34 [CI ±0.24] (98.27–99.23) | 98.93 ± 0.33 [CI ±0.24] (98.27–99.22) | 5.40 ± 1.78 [CI ±1.27] (4.00–9.00) |
| frozen_imagenet | 5 | 88.22 ± 0.61 [CI ±0.76] (87.26–88.80) | 88.58 ± 0.70 [CI ±0.88] (87.52–89.24) | 88.22 ± 0.61 [CI ±0.76] (87.26–88.80) | 88.12 ± 0.62 [CI ±0.78] (87.11–88.64) | 87.96 ± 0.56 [CI ±0.70] (86.97–88.38) | 61.00 ± 3.16 [CI ±3.93] (58.00–66.00) |
| pruned_frozen_imagenet | 5 | 88.84 ± 0.70 [CI ±0.87] (87.84–89.77) | 89.05 ± 0.72 [CI ±0.90] (88.08–90.04) | 88.84 ± 0.70 [CI ±0.87] (87.84–89.77) | 88.82 ± 0.70 [CI ±0.86] (87.84–89.76) | 88.40 ± 0.60 [CI ±0.74] (87.64–89.25) | 57.80 ± 3.63 [CI ±4.51] (53.00–63.00) |
| scl_balanced | 5 | 98.88 ± 0.40 [CI ±0.49] (98.46–99.23) | 98.95 ± 0.36 [CI ±0.44] (98.56–99.28) | 98.88 ± 0.40 [CI ±0.49] (98.46–99.23) | 98.88 ± 0.39 [CI ±0.48] (98.46–99.23) | 98.82 ± 0.43 [CI ±0.54] (98.34–99.20) | 5.80 ± 2.05 [CI ±2.54] (4.00–8.00) |
| pruned_scl_balanced | 5 | 99.00 ± 0.21 [CI ±0.26] (98.65–99.23) | 99.06 ± 0.21 [CI ±0.25] (98.73–99.28) | 99.00 ± 0.21 [CI ±0.26] (98.65–99.23) | 99.00 ± 0.21 [CI ±0.26] (98.65–99.23) | 98.94 ± 0.24 [CI ±0.30] (98.55–99.19) | 5.20 ± 1.10 [CI ±1.36] (4.00–7.00) |
| ce_two_stage_balanced | 5 | 98.96 ± 0.22 [CI ±0.27] (98.65–99.23) | 99.02 ± 0.22 [CI ±0.27] (98.69–99.29) | 98.96 ± 0.22 [CI ±0.27] (98.65–99.23) | 98.96 ± 0.22 [CI ±0.28] (98.64–99.23) | 98.93 ± 0.26 [CI ±0.32] (98.54–99.25) | 5.40 ± 1.14 [CI ±1.42] (4.00–7.00) |
| pruned_ce_two_stage_balanced | 5 | 99.19 ± 0.32 [CI ±0.39] (98.65–99.42) | 99.24 ± 0.32 [CI ±0.39] (98.69–99.46) | 99.19 ± 0.32 [CI ±0.39] (98.65–99.42) | 99.19 ± 0.32 [CI ±0.40] (98.64–99.42) | 99.16 ± 0.36 [CI ±0.44] (98.54–99.41) | 4.20 ± 1.64 [CI ±2.04] (3.00–7.00) |
| resnet_v1 | 5 | 98.42 ± 0.60 [CI ±0.75] (97.68–99.03) | 98.52 ± 0.53 [CI ±0.66] (97.90–99.08) | 98.42 ± 0.60 [CI ±0.75] (97.68–99.03) | 98.42 ± 0.59 [CI ±0.74] (97.70–99.03) | 98.37 ± 0.62 [CI ±0.77] (97.65–99.01) | 8.20 ± 3.11 [CI ±3.87] (5.00–12.00) |
| densenet121 | 5 | 98.65 ± 0.47 [CI ±0.59] (98.07–99.23) | 98.72 ± 0.46 [CI ±0.57] (98.14–99.28) | 98.65 ± 0.47 [CI ±0.59] (98.07–99.23) | 98.65 ± 0.47 [CI ±0.59] (98.07–99.23) | 98.55 ± 0.54 [CI ±0.67] (97.91–99.22) | 7.00 ± 2.45 [CI ±3.04] (4.00–10.00) |
| le_cnn | 5 | 92.51 ± 0.54 [CI ±0.67] (91.89–93.05) | 92.82 ± 0.47 [CI ±0.58] (92.24–93.27) | 92.51 ± 0.54 [CI ±0.67] (91.89–93.05) | 92.49 ± 0.55 [CI ±0.68] (91.85–93.07) | 92.01 ± 0.66 [CI ±0.82] (91.42–92.83) | 38.80 ± 2.77 [CI ±3.45] (36.00–42.00) |

## Training budget (mean over seeds)

                              stage1_epochs  stage2_epochs  grad_steps  train_time_s  best_epoch
config                                                                                          
baseline                                0.0           20.5      2132.0         466.6         NaN
pruned_baseline                         0.0           30.0      3120.0         674.3        25.2
scl                                    50.0           24.1      7706.4        1363.1         NaN
pruned_scl                              0.0           30.0      3120.0         383.2        22.4
ce_two_stage                           50.0           22.7      7560.8        1360.9         NaN
pruned_ce_two_stage                     0.0           30.0      3120.0         387.1        24.3
frozen_imagenet                         0.0           50.0      5200.0         644.1         NaN
pruned_frozen_imagenet                  0.0           30.0      3120.0         388.6        29.8
scl_balanced                           50.0           23.0      7592.0        1435.4         NaN
pruned_scl_balanced                     0.0           30.0      3120.0         397.9        22.2
ce_two_stage_balanced                  50.0           23.0      7592.0        1382.9         NaN
pruned_ce_two_stage_balanced            0.0           30.0      3120.0         395.8        23.8
resnet_v1                               0.0           20.2      2100.8         548.3         NaN
densenet121                             0.0           24.6      2558.4         737.8         NaN
le_cnn                                  0.0           33.8      3515.2          17.9         NaN

## Size (MB = 10^6 bytes) and inference cost (cost from first seed)

config                      baseline  pruned_baseline           scl    pruned_scl  ce_two_stage  pruned_ce_two_stage  frozen_imagenet  pruned_frozen_imagenet  scl_balanced  pruned_scl_balanced  ce_two_stage_balanced  pruned_ce_two_stage_balanced     resnet_v1   densenet121       le_cnn
total_params            2.475446e+07     2.475446e+07  2.475446e+07  2.475446e+07  2.475446e+07         2.475446e+07     2.475446e+07            2.475446e+07  2.475446e+07         2.475446e+07           2.475446e+07                  2.475446e+07  2.477738e+07  7.702879e+06  142424.0000
dense_param_share       4.790000e-02     4.790000e-02  4.790000e-02  4.790000e-02  4.790000e-02         4.790000e-02     4.790000e-02            4.790000e-02  4.790000e-02         4.790000e-02           4.790000e-02                  4.790000e-02  4.790000e-02  8.600000e-02       0.3453
dense_kernel_sparsity   0.000000e+00     8.000000e-01  0.000000e+00  8.000000e-01  0.000000e+00         8.000000e-01     0.000000e+00            8.000000e-01  0.000000e+00         8.000000e-01           0.000000e+00                  8.000000e-01  0.000000e+00  0.000000e+00       0.0000
model_sparsity          0.000000e+00     3.830000e-02  0.000000e+00  3.830000e-02  0.000000e+00         3.830000e-02     0.000000e+00            3.830000e-02  0.000000e+00         3.830000e-02           0.000000e+00                  3.830000e-02  0.000000e+00  0.000000e+00       0.0000
nonzero_params          2.475446e+07     2.380586e+07  2.475446e+07  2.380586e+07  2.475446e+07         2.380586e+07     2.475446e+07            2.380586e+07  2.475446e+07         2.380586e+07           2.475446e+07                  2.380586e+07  2.477736e+07  7.702874e+06  142424.0000
h5_with_opt_MB          2.974504e+02     9.935410e+01  1.088826e+02  9.935220e+01  1.088826e+02         9.935220e+01     1.088826e+02            9.935220e+01  1.088826e+02         9.935220e+01           1.088826e+02                  9.935220e+01  2.977650e+02  9.341180e+01       1.7623
h5_weights_MB           9.935400e+01     9.935400e+01  9.935210e+01  9.935210e+01  9.935210e+01         9.935210e+01     9.935210e+01            9.935210e+01  9.935210e+01         9.935210e+01           9.935210e+01                  9.935210e+01  9.946000e+01  3.153890e+01       0.6027
h5_weights_gz_MB        9.204290e+01     8.888280e+01  9.201490e+01  8.890920e+01  9.197960e+01         8.887480e+01     9.212070e+01            8.902470e+01  9.201960e+01         8.891480e+01           9.199820e+01                  8.889390e+01  9.200670e+01  2.859330e+01       0.5321
tflite_f32_MB           9.884720e+01     9.884720e+01  9.884720e+01  9.884720e+01  9.884720e+01         9.884720e+01     9.884720e+01            9.884720e+01  9.884720e+01         9.884720e+01           9.884720e+01                  9.884720e+01  9.872420e+01  3.055110e+01       0.5736
tflite_f32_gz_MB        9.186610e+01     8.871680e+01  9.184540e+01  8.873380e+01  9.182070e+01         8.871120e+01     9.192410e+01            8.881830e+01  9.184860e+01         8.873990e+01           9.183590e+01                  8.872630e+01  9.173660e+01  2.850600e+01       0.5305
tflite_int8_MB          2.548030e+01     2.548030e+01  2.548030e+01  2.548030e+01  2.548030e+01         2.548030e+01     2.548030e+01            2.548030e+01  2.548030e+01         2.548030e+01           2.548030e+01                  2.548030e+01  2.544530e+01  8.022400e+00       0.1534
tflite_int8_gz_MB       2.114040e+01     2.051430e+01  2.119170e+01  2.042950e+01  2.128670e+01         2.052220e+01     2.092930e+01            2.025410e+01  2.118310e+01         2.041950e+01           2.122630e+01                  2.046500e+01  2.143780e+01  6.490000e+00       0.1302
int8_subset_n           1.200000e+02     1.200000e+02  1.200000e+02  1.200000e+02           NaN                  NaN              NaN                     NaN           NaN                  NaN                    NaN                           NaN           NaN           NaN          NaN
float_acc_subset        9.833000e-01     9.750000e-01  1.000000e+00  1.000000e+00           NaN                  NaN              NaN                     NaN           NaN                  NaN                    NaN                           NaN           NaN           NaN          NaN
int8_acc_subset         9.750000e-01     9.750000e-01  1.000000e+00  1.000000e+00           NaN                  NaN              NaN                     NaN           NaN                  NaN                    NaN                           NaN           NaN           NaN          NaN
int8_float_agreement    9.917000e-01     1.000000e+00  1.000000e+00  1.000000e+00           NaN                  NaN              NaN                     NaN           NaN                  NaN                    NaN                           NaN           NaN           NaN          NaN
gflops                  6.972400e+00     6.972400e+00  6.972400e+00  6.972400e+00  6.972400e+00         6.972400e+00     6.972400e+00            6.972400e+00           NaN                  NaN                    NaN                           NaN           NaN           NaN          NaN
dense_head_mflops       2.371600e+00     2.371600e+00  2.371600e+00  2.371600e+00  2.371600e+00         2.371600e+00     2.371600e+00            2.371600e+00           NaN                  NaN                    NaN                           NaN           NaN           NaN          NaN
latency_gpu_ms          6.497500e+00     9.012500e+00  5.800400e+00  5.968300e+00  5.556800e+00         5.702800e+00     5.599500e+00            5.563800e+00           NaN                  NaN                    NaN                           NaN           NaN           NaN          NaN
latency_cpu_ms          8.618260e+01     1.768516e+02  9.721800e+01  1.783634e+02  8.547680e+01         1.859258e+02     1.795426e+02            1.304033e+02           NaN                  NaN                    NaN                           NaN           NaN           NaN          NaN
latency_tflite_f32_ms   2.483760e+01     3.179160e+01  1.870550e+01  1.861850e+01  1.949440e+01         1.866150e+01     1.879200e+01            1.824370e+01           NaN                  NaN                    NaN                           NaN           NaN           NaN          NaN
latency_tflite_int8_ms  2.684213e+03     2.973966e+03  2.701883e+03  2.748035e+03  2.635434e+03         2.622740e+03     2.659998e+03            2.631572e+03           NaN                  NaN                    NaN                           NaN           NaN           NaN          NaN
peak_gpu_mem_MB         7.764365e+02     1.022714e+03  7.195676e+02  8.001216e+02  6.932547e+02         8.046275e+02     6.311045e+02            6.295176e+02           NaN                  NaN                    NaN                           NaN           NaN           NaN          NaN
export_max_diff         0.000000e+00     0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00         0.000000e+00     0.000000e+00            0.000000e+00  0.000000e+00         0.000000e+00           0.000000e+00                  0.000000e+00  0.000000e+00  0.000000e+00          NaN

## LaTeX rows: accuracy / precision / recall / F1 (mean $\pm$ SD)
```latex
baseline & 98.47 $\pm$ 0.28 & 98.58 $\pm$ 0.25 & 98.47 $\pm$ 0.28 & 98.48 $\pm$ 0.28 \\
pruned_baseline & 99.07 $\pm$ 0.27 & 99.14 $\pm$ 0.24 & 99.07 $\pm$ 0.27 & 99.07 $\pm$ 0.27 \\
scl & 98.90 $\pm$ 0.50 & 98.96 $\pm$ 0.48 & 98.90 $\pm$ 0.50 & 98.90 $\pm$ 0.50 \\
pruned_scl & 99.00 $\pm$ 0.54 & 99.07 $\pm$ 0.46 & 99.00 $\pm$ 0.54 & 99.00 $\pm$ 0.52 \\
ce_two_stage & 98.98 $\pm$ 0.34 & 99.04 $\pm$ 0.33 & 98.98 $\pm$ 0.34 & 98.98 $\pm$ 0.34 \\
pruned_ce_two_stage & 98.96 $\pm$ 0.34 & 99.02 $\pm$ 0.31 & 98.96 $\pm$ 0.34 & 98.96 $\pm$ 0.34 \\
frozen_imagenet & 88.22 $\pm$ 0.61 & 88.58 $\pm$ 0.70 & 88.22 $\pm$ 0.61 & 88.12 $\pm$ 0.62 \\
pruned_frozen_imagenet & 88.84 $\pm$ 0.70 & 89.05 $\pm$ 0.72 & 88.84 $\pm$ 0.70 & 88.82 $\pm$ 0.70 \\
scl_balanced & 98.88 $\pm$ 0.40 & 98.95 $\pm$ 0.36 & 98.88 $\pm$ 0.40 & 98.88 $\pm$ 0.39 \\
pruned_scl_balanced & 99.00 $\pm$ 0.21 & 99.06 $\pm$ 0.21 & 99.00 $\pm$ 0.21 & 99.00 $\pm$ 0.21 \\
ce_two_stage_balanced & 98.96 $\pm$ 0.22 & 99.02 $\pm$ 0.22 & 98.96 $\pm$ 0.22 & 98.96 $\pm$ 0.22 \\
pruned_ce_two_stage_balanced & 99.19 $\pm$ 0.32 & 99.24 $\pm$ 0.32 & 99.19 $\pm$ 0.32 & 99.19 $\pm$ 0.32 \\
resnet_v1 & 98.42 $\pm$ 0.60 & 98.52 $\pm$ 0.53 & 98.42 $\pm$ 0.60 & 98.42 $\pm$ 0.59 \\
densenet121 & 98.65 $\pm$ 0.47 & 98.72 $\pm$ 0.46 & 98.65 $\pm$ 0.47 & 98.65 $\pm$ 0.47 \\
le_cnn & 92.51 $\pm$ 0.54 & 92.82 $\pm$ 0.47 & 92.51 $\pm$ 0.54 & 92.49 $\pm$ 0.55 \\
```

## Paired comparisons across seeds (accuracy, percentage points)

- scl − baseline (SCL vs end-to-end baseline (paper's comparison)): mean +0.42 pp, SD 0.45, paired t p = 0.016, Wilcoxon p = 0.033; better in 7/10 seeds, ties 1
- scl − ce_two_stage (SCL vs SAME procedure with CE  <- effect of SCL itself): mean -0.08 pp, SD 0.38, paired t p = 0.534, Wilcoxon p = 0.375; better in 4/10 seeds, ties 0
- scl − frozen_imagenet (SCL vs no stage 1 (frozen ImageNet encoder)): mean +10.69 pp, SD 0.62, paired t p = 0.000, Wilcoxon p = 0.062; better in 5/5 seeds, ties 0
- ce_two_stage − baseline (two-stage procedure vs end-to-end (procedure effect)): mean +0.50 pp, SD 0.39, paired t p = 0.003, Wilcoxon p = 0.006; better in 9/10 seeds, ties 0
- pruned_scl − pruned_baseline (after pruning: SCL vs baseline): mean -0.08 pp, SD 0.56, paired t p = 0.674, Wilcoxon p = 0.770; better in 5/10 seeds, ties 0
- pruned_scl − pruned_ce_two_stage (after pruning: SCL vs CE two-stage): mean +0.04 pp, SD 0.42, paired t p = 0.775, Wilcoxon p = 0.888; better in 4/10 seeds, ties 2
- scl_balanced − scl (balanced vs random batches for SCL): mean -0.04 pp, SD 0.48, paired t p = 0.866, Wilcoxon p = 1.000; better in 2/5 seeds, ties 0
- scl_balanced − ce_two_stage_balanced (balanced SCL vs balanced CE  <- effect of SCL): mean -0.08 pp, SD 0.52, paired t p = 0.757, Wilcoxon p = 0.715; better in 2/5 seeds, ties 1
- scl_balanced − baseline (balanced SCL vs end-to-end baseline): mean +0.27 pp, SD 0.44, paired t p = 0.245, Wilcoxon p = 0.312; better in 4/5 seeds, ties 0
- ce_two_stage_balanced − ce_two_stage (balanced vs random batches for CE): mean +0.04 pp, SD 0.50, paired t p = 0.871, Wilcoxon p = 0.854; better in 2/5 seeds, ties 1
- pruned_scl_balanced − pruned_baseline (after pruning: balanced SCL vs baseline): mean -0.08 pp, SD 0.26, paired t p = 0.541, Wilcoxon p = 0.625; better in 2/5 seeds, ties 0

## Accuracy change caused by pruning (pruned − unpruned, pp)

- baseline: +0.60 ± 0.31 pp (per seed: +0.19, +0.58, +0.00, +0.97, +0.58, +0.58, +0.77, +0.58, +0.77, +0.97)
- scl: +0.10 ± 0.25 pp (per seed: +0.00, -0.39, +0.00, +0.19, +0.00, +0.19, +0.58, +0.00, +0.19, +0.19)
- ce_two_stage: -0.02 ± 0.26 pp (per seed: -0.19, -0.19, +0.00, +0.58, -0.19, +0.00, -0.39, +0.00, +0.19, +0.00)
- frozen_imagenet: +0.62 ± 0.35 pp (per seed: +0.97, +0.97, +0.19, +0.58, +0.39)
- scl_balanced: +0.12 ± 0.29 pp (per seed: +0.19, +0.58, +0.00, +0.00, -0.19)
- ce_two_stage_balanced: +0.23 ± 0.16 pp (per seed: +0.19, +0.39, +0.19, +0.39, +0.00)
- drop(scl) − drop(baseline) (pruning robustness difference): mean -0.50 pp, SD 0.30, paired t p = 0.001, Wilcoxon p = 0.008; better in 0/10 seeds, ties 1
- drop(scl) − drop(ce_two_stage) (pruning robustness difference): mean +0.12 pp, SD 0.35, paired t p = 0.329, Wilcoxon p = 0.574; better in 5/10 seeds, ties 2
- drop(scl) − drop(frozen_imagenet) (pruning robustness difference): mean -0.66 pp, SD 0.48, paired t p = 0.039, Wilcoxon p = 0.062; better in 0/5 seeds, ties 0
- drop(scl_balanced) − drop(baseline) (pruning robustness difference): mean -0.35 pp, SD 0.48, paired t p = 0.181, Wilcoxon p = 0.197; better in 1/5 seeds, ties 1
- drop(scl_balanced) − drop(ce_two_stage_balanced) (pruning robustness difference): mean -0.12 pp, SD 0.22, paired t p = 0.305, Wilcoxon p = 0.141; better in 1/5 seeds, ties 1

## Exact McNemar test per seed: baseline vs scl

- seed 0: only scl correct = 3, only baseline correct = 3, p = 1.000
- seed 1: only scl correct = 4, only baseline correct = 5, p = 1.000
- seed 2: only scl correct = 3, only baseline correct = 0, p = 0.250
- seed 3: only scl correct = 4, only baseline correct = 3, p = 1.000
- seed 4: only scl correct = 5, only baseline correct = 0, p = 0.062
- seed 5: only scl correct = 7, only baseline correct = 2, p = 0.180
- seed 6: only scl correct = 4, only baseline correct = 5, p = 1.000
- seed 7: only scl correct = 4, only baseline correct = 0, p = 0.125
- seed 8: only scl correct = 7, only baseline correct = 3, p = 0.344
- seed 9: only scl correct = 4, only baseline correct = 2, p = 0.688

## Exact McNemar test per seed: ce_two_stage vs scl

- seed 0: only scl correct = 2, only ce_two_stage correct = 4, p = 0.688
- seed 1: only scl correct = 4, only ce_two_stage correct = 7, p = 0.549
- seed 2: only scl correct = 2, only ce_two_stage correct = 1, p = 1.000
- seed 3: only scl correct = 6, only ce_two_stage correct = 4, p = 0.754
- seed 4: only scl correct = 2, only ce_two_stage correct = 0, p = 0.500
- seed 5: only scl correct = 1, only ce_two_stage correct = 2, p = 1.000
- seed 6: only scl correct = 4, only ce_two_stage correct = 6, p = 0.754
- seed 7: only scl correct = 3, only ce_two_stage correct = 1, p = 0.625
- seed 8: only scl correct = 2, only ce_two_stage correct = 3, p = 1.000
- seed 9: only scl correct = 0, only ce_two_stage correct = 2, p = 0.500

## Exact McNemar test per seed: baseline vs scl_balanced

- seed 0: only scl_balanced correct = 2, only baseline correct = 4, p = 0.688
- seed 1: only scl_balanced correct = 5, only baseline correct = 4, p = 1.000
- seed 2: only scl_balanced correct = 3, only baseline correct = 2, p = 1.000
- seed 3: only scl_balanced correct = 5, only baseline correct = 1, p = 0.219
- seed 4: only scl_balanced correct = 3, only baseline correct = 0, p = 0.250

## Exact McNemar test per seed: ce_two_stage_balanced vs scl_balanced

- seed 0: only scl_balanced correct = 0, only ce_two_stage_balanced correct = 4, p = 0.125
- seed 1: only scl_balanced correct = 3, only ce_two_stage_balanced correct = 5, p = 0.727
- seed 2: only scl_balanced correct = 2, only ce_two_stage_balanced correct = 2, p = 1.000
- seed 3: only scl_balanced correct = 2, only ce_two_stage_balanced correct = 1, p = 1.000
- seed 4: only scl_balanced correct = 4, only ce_two_stage_balanced correct = 1, p = 0.375