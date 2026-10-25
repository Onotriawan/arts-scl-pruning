import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_CONFIGS = ["baseline", "scl", "ce_two_stage",
                   "pruned_baseline", "pruned_scl", "pruned_ce_two_stage"]
ARTS_NAMES = ["Front or Left", "Front or Right", "Hump", "Left Turn", "Narrows from Right",
              "Narrows from Left", "No Horn", "No Parking", "No U Turn",
              "Overtaking Forbidden", "Parking", "Pedestrian Crossing", "Right or Left",
              "Right Turn", "Rotor", "Slow", "Speed 100", "Speed 30", "Speed 40",
              "Speed 50", "Speed 60", "Speed 80", "Stop", "U Turn"]
PRETTY = {"baseline": "End-to-end baseline", "scl": "SCL two-stage",
          "ce_two_stage": "CE two-stage", "pruned_baseline": "Pruned baseline",
          "pruned_scl": "Pruned SCL", "pruned_ce_two_stage": "Pruned CE two-stage"}


def load_runs(dirs, cfg):
    """Return {seed: (y_true, y_pred)} for one config across all result folders."""
    runs = {}
    for d in dirs:
        yt_path = os.path.join(d, "y_test.npy")
        if not os.path.exists(yt_path):
            print(f"  [skip] {yt_path} not found")
            continue
        y_true = np.load(yt_path)
        for f in glob.glob(os.path.join(d, f"pred_{cfg}_seed*.npy")):
            seed = int(re.search(r"seed(\d+)\.npy$", f).group(1))
            y_pred = np.load(f)
            if len(y_pred) != len(y_true):
                print(f"  [skip] {f}: {len(y_pred)} predictions vs {len(y_true)} labels")
                continue
            runs[seed] = (y_true, y_pred)
    return dict(sorted(runs.items()))


def confusion(y_true, y_pred, labels):
    idx = {c: i for i, c in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm


def per_class(cm):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - tp - fp - fn
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        rec = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), np.nan)
    return tp, fp, fn, tn, prec, rec, f1


plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
DIAG_COLOR = "#dfe6ee"      # correct predictions: light grey-blue background
ERR_CMAP = "Reds"           # misclassifications: coloured by count


def draw_cm(ax, cm, names, vmax_err, fs=7.5, label_fs=9):
    """Error-focused confusion matrix: diagonal shown as plain counts on a light
    background, off-diagonal (errors) coloured so that even 1-2 errors are visible."""
    n = len(names)
    off = cm.astype(float).copy()
    np.fill_diagonal(off, np.nan)
    off[off == 0] = np.nan
    ax.set_facecolor("white")
    for k in range(n):                                   # diagonal background
        ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1, color=DIAG_COLOR, lw=0))
    im = ax.imshow(off, cmap=ERR_CMAP, vmin=0, vmax=max(vmax_err, 1),
                   interpolation="nearest")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=90 if max(len(s) for s in names) > 2 else 0,
                       fontsize=fs)
    ax.set_yticklabels(names, fontsize=fs)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="#d0d0d0", lw=0.4)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("Predicted class", fontsize=label_fs)
    ax.set_ylabel("True class", fontsize=label_fs)
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            if v == 0:
                continue
            if i == j:
                ax.text(j, i, f"{v}", ha="center", va="center", fontsize=fs - 1.5,
                        color="#1f3b57")
            else:
                ax.text(j, i, f"{v}", ha="center", va="center", fontsize=fs,
                        fontweight="bold",
                        color="white" if v > 0.55 * vmax_err else "black")
    return im



CLASSIC_CMAP = "Blues"
SHOW_ERRCOUNT = False        # set by --show-error-count
ERR_TEXT = "#c0392b"          # misclassification counts in red so they stand out
ERR_BG = "#fde2e0"            # light red cell behind each misclassification


def draw_cm_classic(ax, cm, names, vmax, fs=8, label_fs=10, show_zeros=False):
    """Classic blue confusion matrix (as in the original paper), with bold, larger
    numbers. Off-diagonal errors get a light-red cell and red bold text so they remain
    visible even though they are much smaller than the diagonal counts."""
    n = len(names)
    im = ax.imshow(cm, cmap=CLASSIC_CMAP, vmin=0, vmax=vmax, interpolation="nearest")
    for i in range(n):
        for j in range(n):
            v = int(cm[i, j])
            if i != j and v > 0:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=ERR_BG,
                                           lw=0, zorder=2))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, fontsize=fs, fontweight="bold",
                       rotation=90 if max(len(x) for x in names) > 2 else 0)
    ax.set_yticklabels(names, fontsize=fs, fontweight="bold")
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.6)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=2, pad=1.5)
    ax.set_xlabel("Predicted Label", fontsize=label_fs, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=label_fs, fontweight="bold")
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
    for i in range(n):
        for j in range(n):
            v = int(cm[i, j])
            if v == 0 and not show_zeros:
                continue
            size = fs if len(str(v)) <= 2 else fs * 0.7
            if i == j:
                col = "white" if v > 0.5 * vmax else "#08306b"
                ax.text(j, i, str(v), ha="center", va="center", fontsize=size,
                        fontweight="bold", color=col, zorder=3)
            elif v > 0:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=size,
                        fontweight="bold", color=ERR_TEXT, zorder=3)
            else:
                ax.text(j, i, "0", ha="center", va="center", fontsize=fs - 1.5,
                        color="#b0b0b0", zorder=3)
    return im


def plot_cm(cm, names, title, path, normalize=False, style="classic", show_zeros=False):
    """Single matrix -> PNG (600 dpi) and vector PDF. No title inside the figure:
    put the description in the LaTeX caption instead."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    if style == "classic":
        im = draw_cm_classic(ax, cm, names, cm.max(), fs=8.5, label_fs=11,
                             show_zeros=show_zeros)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("Number of test images", fontsize=10, fontweight="bold")
    else:
        im = draw_cm(ax, cm, names, off.max())
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("Misclassified images (off-diagonal)", fontsize=8)
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    base = os.path.splitext(path)[0]
    fig.savefig(base + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def plot_panel(cms_by_cfg, names, path, n_seeds, style="classic", show_zeros=False,
               subtitle=None):
    """2x2 panel (a)-(d) with one shared colour scale -> PDF + PNG.
    Sized for a full-width MDPI figure (17.5 cm), so fonts print at their real size."""
    cfgs = [c for c in ["baseline", "pruned_baseline", "scl", "pruned_scl"]
            if c in cms_by_cfg]
    if len(cfgs) < 2:
        return
    subtitle = subtitle or {}
    if style == "classic":
        vmax = max(cms_by_cfg[c].max() for c in cfgs)
    else:
        vmax = 0
        for c in cfgs:
            off = cms_by_cfg[c].copy()
            np.fill_diagonal(off, 0)
            vmax = max(vmax, off.max())
    rows = int(np.ceil(len(cfgs) / 2))
    W = 17.5 / 2.54
    fig, axes = plt.subplots(rows, 2, figsize=(W, W * 0.5 * rows + 0.5))
    axes = np.atleast_2d(axes)
    im = None
    for k, c in enumerate(cfgs):
        ax = axes[k // 2, k % 2]
        if style == "classic":
            im = draw_cm_classic(ax, cms_by_cfg[c], names, vmax, fs=5.6, label_fs=8,
                                 show_zeros=show_zeros)
        else:
            im = draw_cm(ax, cms_by_cfg[c], names, vmax, fs=6.5, label_fs=8.5)
        off = cms_by_cfg[c].copy()
        np.fill_diagonal(off, 0)
        extra = f" [{int(off.sum())} errors]" if (style == "classic" and SHOW_ERRCOUNT) else ""
        ax.set_title(f"({chr(97 + k)}) {PRETTY.get(c, c)}{subtitle.get(c, '')}{extra}",
                     fontsize=8.5, loc="left", fontweight="bold")
    for k in range(len(cfgs), rows * 2):
        axes[k // 2, k % 2].axis("off")
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=1.2, h_pad=1.0)
    cax = fig.add_axes([0.25, 0.025, 0.5, 0.014])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    lab = (f"Number of test images (sum of {n_seeds} seeds)" if n_seeds > 1
           else "Number of test images")
    if style != "classic":
        lab = f"Misclassified images (sum of {n_seeds} seeds)"
    cb.set_label(lab, fontsize=8, fontweight="bold")
    cb.ax.tick_params(labelsize=7)
    base = os.path.splitext(path)[0]
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["results_ok", "results_run2"])
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--out", default="results_confusion")
    ap.add_argument("--names", default=None, help="text file, one class name per line")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--examples", action="store_true",
                    help="also make a figure with real misclassified / correctly rejected images")
    ap.add_argument("--examples-config", default="scl")
    ap.add_argument("--n-pairs", type=int, default=3)
    ap.add_argument("--script", default="arts_multiseed.py",
                    help="path to arts_multiseed.py (for the data paths and image loader)")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--style", choices=["classic", "errors"], default="classic",
                    help="classic = blue matrix like the original paper; "
                         "errors = grey diagonal + red errors")
    ap.add_argument("--panel-source", choices=["summed", "median"], default="summed",
                    help="2x2 panel from the matrix summed over all seeds, or from the "
                         "median-accuracy seed of each configuration")
    ap.add_argument("--show-error-count", action="store_true",
                    help="add '[N errors]' to the panel titles (off by default)")
    ap.add_argument("--show-zeros", action="store_true",
                    help="also print the zeros (light grey), as in the original figure")
    args = ap.parse_args()
    global SHOW_ERRCOUNT
    SHOW_ERRCOUNT = args.show_error_count
    os.makedirs(args.out, exist_ok=True)

    summary = ["# Confusion-matrix summary\n"]
    agg_rows = []
    panel_cms, panel_info = {}, {}
    panel_med, panel_medseed = {}, {}
    names_for_examples = None
    if args.names:
        with open(args.names, encoding="utf-8") as fh:
            names_for_examples = [l.strip() for l in fh if l.strip()]
    for cfg in args.configs:
        runs = load_runs(args.dirs, cfg)
        if not runs:
            print(f"[{cfg}] no prediction files found, skipped")
            continue
        all_true = np.concatenate([r[0] for r in runs.values()])
        labels = sorted(np.unique(all_true).tolist())
        if args.names:
            with open(args.names, encoding="utf-8") as fh:
                names = [l.strip() for l in fh if l.strip()]
            if len(names) != len(labels):
                print(f"  {len(names)} names but {len(labels)} classes; using numbers")
                names = [str(c) for c in labels]
        else:
            names = [str(c) for c in labels]

        seeds = list(runs)
        cms = {s: confusion(*runs[s], labels) for s in seeds}
        cm_sum = sum(cms.values())
        n_seeds = len(seeds)
        panel_cms[cfg] = cm_sum
        panel_info["names"], panel_info["n_seeds"] = names, n_seeds
        n_test = len(next(iter(runs.values()))[0])

        # ---- summed matrix (all seeds) ----
        pd.DataFrame(cm_sum, index=[f"true_{n}" for n in names],
                     columns=[f"pred_{n}" for n in names]).to_csv(
            os.path.join(args.out, f"confusion_{cfg}_summed.csv"))
        plot_cm(cm_sum, names,
                f"{PRETTY.get(cfg, cfg)}: confusion matrix summed over {n_seeds} seeds "
                f"({n_seeds} x {n_test} test predictions)",
                os.path.join(args.out, f"confusion_{cfg}_summed.png"),
                style=args.style, show_zeros=args.show_zeros)

        # ---- representative (median-accuracy) seed ----
        accs = {s: np.trace(cms[s]) / cms[s].sum() for s in seeds}
        med_seed = sorted(accs, key=accs.get)[(n_seeds - 1) // 2]
        plot_cm(cms[med_seed], names,
                f"{PRETTY.get(cfg, cfg)}: seed {med_seed} "
                f"(median accuracy {100 * accs[med_seed]:.2f}%)",
                os.path.join(args.out, f"confusion_{cfg}_seed{med_seed}.png"),
                style=args.style, show_zeros=args.show_zeros)
        panel_med[cfg] = cms[med_seed]
        panel_medseed[cfg] = med_seed

        # ---- per-class TP/FP/FN/TN (mean per run) ----
        tp, fp, fn, tn, prec, rec, f1 = per_class(cm_sum)
        df = pd.DataFrame({
            "class": names,
            "support_per_run": cm_sum.sum(axis=1) / n_seeds,
            "TP_mean": tp / n_seeds, "FP_mean": fp / n_seeds,
            "FN_mean": fn / n_seeds, "TN_mean": tn / n_seeds,
            "precision_%": 100 * prec, "recall_%": 100 * rec, "F1_%": 100 * f1,
        })
        df.round(2).to_csv(os.path.join(args.out, f"perclass_{cfg}.csv"), index=False)
        with open(os.path.join(args.out, f"perclass_{cfg}.tex"), "w", encoding="utf-8") as fh:
            fh.write("% Class & Support & TP & FP & FN & TN & Precision & Recall & F1 \\\\\n")
            for _, r in df.iterrows():
                fh.write(f"{r['class']} & {r['support_per_run']:.0f} & {r['TP_mean']:.1f} & "
                         f"{r['FP_mean']:.1f} & {r['FN_mean']:.1f} & {r['TN_mean']:.1f} & "
                         f"{r['precision_%']:.2f} & {r['recall_%']:.2f} & {r['F1_%']:.2f} \\\\\n")

        # ---- most frequent confusions ----
        off = cm_sum.copy()
        np.fill_diagonal(off, 0)
        pairs = [(names[i], names[j], int(off[i, j]),
                  sum(1 for s in seeds if cms[s][i, j] > 0))
                 for i in range(len(labels)) for j in range(len(labels)) if off[i, j] > 0]
        pairs.sort(key=lambda x: -x[2])
        tops = pd.DataFrame(pairs, columns=["true_class", "predicted_as",
                                            "count_all_seeds", "seeds_with_error"])
        tops["mean_per_run"] = (tops["count_all_seeds"] / n_seeds).round(2)
        tops.to_csv(os.path.join(args.out, f"top_confusions_{cfg}.csv"), index=False)

        total_err = int(off.sum())
        summary.append(f"## {PRETTY.get(cfg, cfg)} ({cfg})\n")
        summary.append(f"- Seeds: {seeds} (n = {n_seeds}); test images per run: {n_test}")
        summary.append(f"- Mean errors per run: {total_err / n_seeds:.1f}; "
                       f"median-accuracy seed: {med_seed} ({100 * accs[med_seed]:.2f}%)")
        worst = df.sort_values("recall_%").head(3)
        summary.append("- Lowest-recall classes: " + ", ".join(
            f"{r['class']} ({r['recall_%']:.1f}%)" for _, r in worst.iterrows()))
        summary.append(f"- Most frequent confusions (top {args.top}, all seeds):")
        for _, r in tops.head(args.top).iterrows():
            summary.append(f"    - {r['true_class']} -> {r['predicted_as']}: "
                           f"{r['count_all_seeds']} times ({r['mean_per_run']} per run, "
                           f"in {r['seeds_with_error']}/{n_seeds} seeds)")
        summary.append("")
        print(f"[{cfg}] {n_seeds} seeds, {total_err / n_seeds:.1f} errors/run -> done")

        # ---- aggregate TP/FP/FN/TN per run (for the new table, R1 C5) ----
        rows = []
        for s in seeds:
            t, f_p, f_n, t_n, p_, r_, f1_ = per_class(cms[s])
            rows.append({"seed": s, "TP": t.sum(), "FP": f_p.sum(), "FN": f_n.sum(),
                         "TN": t_n.sum(), "macro_precision": 100 * np.nanmean(p_),
                         "macro_recall": 100 * np.nanmean(r_), "macro_F1": 100 * np.nanmean(f1_),
                         "accuracy": 100 * t.sum() / cms[s].sum()})
        agg_rows.append((cfg, pd.DataFrame(rows)))

    with open(os.path.join(args.out, "summary_confusion.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(summary))

    if panel_cms:
        if args.panel_source == "median":
            plot_panel(panel_med, panel_info["names"],
                       os.path.join(args.out, "confusion_panel.pdf"), 1,
                       style=args.style, show_zeros=args.show_zeros,
                       subtitle={c: f", seed {s}" for c, s in panel_medseed.items()})
        else:
            plot_panel(panel_cms, panel_info["names"],
                       os.path.join(args.out, "confusion_panel.pdf"),
                       panel_info["n_seeds"], style=args.style,
                       show_zeros=args.show_zeros)

    if agg_rows:
        out_rows, tex = [], []
        for cfg, d in agg_rows:
            m, sd = d.mean(numeric_only=True), d.std(numeric_only=True, ddof=1)
            rec = {"method": PRETTY.get(cfg, cfg), "n_seeds": len(d)}
            for k in ["TP", "FP", "FN", "TN", "accuracy", "macro_precision",
                      "macro_recall", "macro_F1"]:
                rec[f"{k}_mean"], rec[f"{k}_sd"] = round(m[k], 2), round(sd[k], 2)
            out_rows.append(rec)
            tex.append(f"{PRETTY.get(cfg, cfg)} & {m['TP']:.1f} $\\pm$ {sd['TP']:.1f} & "
                       f"{m['FP']:.1f} $\\pm$ {sd['FP']:.1f} & {m['FN']:.1f} $\\pm$ {sd['FN']:.1f} & "
                       f"{m['TN']:.1f} $\\pm$ {sd['TN']:.1f} \\\\")
            d.to_csv(os.path.join(args.out, f"tp_fp_fn_tn_per_seed_{cfg}.csv"), index=False)
        pd.DataFrame(out_rows).to_csv(os.path.join(args.out, "tp_fp_fn_tn_by_method.csv"),
                                      index=False)
        with open(os.path.join(args.out, "tp_fp_fn_tn_by_method.tex"), "w", encoding="utf-8") as fh:
            fh.write("% Method & TP & FP & FN & TN  (per run, summed over the 24 classes, "
                     "one-vs-rest; mean $\\pm$ SD over seeds)\n" + "\n".join(tex) + "\n")

    if args.examples:
        make_examples(args, names_for_examples or ARTS_NAMES)
    print(f"\nAll outputs written to: {os.path.abspath(args.out)}")


def make_examples(args, names):
    """Figure with real test images: misclassified (FP/FN) and correctly rejected (TN)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("arts", args.script)
    arts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arts)
    te_dir, te_csv = arts.TEST_DIR, arts.TEST_CSV
    if args.data_root:
        te_dir = os.path.join(args.data_root, "test")
        te_csv = os.path.join(args.data_root, os.path.basename(te_csv.replace("\\", "/")))
    print(f"\nLoading test images from {te_csv} ...")
    X, y = arts.read_csv_data(te_csv, te_dir)

    cfg = args.examples_config
    runs = load_runs(args.dirs, cfg)
    if not runs:
        print(f"  no predictions for {cfg}; examples skipped")
        return
    y_true = next(iter(runs.values()))[0]
    if len(y) != len(y_true) or not np.array_equal(y, y_true):
        print("  WARNING: test images/labels do not match y_test.npy; examples skipped")
        return
    P = np.stack([r[1] for r in runs.values()])          # seeds x images
    n_seeds = P.shape[0]
    wrong = P != y_true[None, :]
    err_count = wrong.sum(axis=0)
    labels = sorted(np.unique(y_true).tolist())
    name_of = {c: (names[i] if names else str(c)) for i, c in enumerate(labels)}

    # most frequent confusion pairs (true a -> predicted b)
    pair_counts = {}
    for s in range(n_seeds):
        for i in np.where(wrong[s])[0]:
            key = (int(y_true[i]), int(P[s, i]))
            pair_counts[key] = pair_counts.get(key, 0) + 1
    pairs = sorted(pair_counts, key=pair_counts.get, reverse=True)[:args.n_pairs]
    if not pairs:
        print("  no errors at all; nothing to show")
        return

    def show(img):
        img = np.asarray(img, dtype=float)
        if img.max() > 1.0:
            img = img / 255.0
        return np.clip(img, 0, 1)

    col_titles = ["Misclassified (FP / FN)", "Correctly recognised (TN)",
                  "Reference of predicted class"]
    colours = ["#c0392b", "#1e8449", "#555555"]
    fig, axes = plt.subplots(len(pairs), 3, figsize=(8.0, 2.9 * len(pairs)))
    axes = np.atleast_2d(axes)
    for r, (a, b) in enumerate(pairs):
        cand = np.where(y_true == a)[0]
        k_ab = np.array([(P[:, i] == b).sum() for i in cand])
        i_fp = cand[np.argmax(k_ab)]                      # true a, predicted b
        ok = cand[err_count[cand] == 0]
        i_tn = ok[0] if len(ok) else None                 # true a, always correct
        refs = np.where((y_true == b) & (err_count == 0))[0]
        i_ref = refs[0] if len(refs) else np.where(y_true == b)[0][0]
        texts = [f"True: {name_of[a]}\nPredicted: {name_of[b]}\n"
                 f"(wrong in {int(k_ab.max())}/{n_seeds} seeds)",
                 f"True: {name_of[a]}\nPredicted: {name_of[a]}\n"
                 f"(TN for {name_of[b]})",
                 f"{name_of[b]}\n(correct in all seeds)"]
        for c, idx in enumerate([i_fp, i_tn, i_ref]):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            if idx is None:
                ax.text(0.5, 0.5, "no example", ha="center", va="center", fontsize=9)
            else:
                ax.imshow(show(X[idx]))
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(colours[c]); sp.set_linewidth(2.5)
            ax.set_xlabel(texts[c], fontsize=8.5, fontweight="bold", color=colours[c],
                          labelpad=4)
            if r == 0:
                ax.set_title(f"({chr(97 + c)}) {col_titles[c]}", fontsize=10,
                             fontweight="bold")
    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    path = os.path.join(args.out, f"error_examples_{cfg}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)

    # list of every misclassified test image, for the discussion
    rows = [{"test_index": int(i), "true": name_of[int(y_true[i])],
             "wrong_in_seeds": int(err_count[i]),
             "predicted_as": ", ".join(sorted({name_of[int(p)] for p in P[wrong[:, i], i]}))}
            for i in np.where(err_count > 0)[0]]
    pd.DataFrame(rows).sort_values("wrong_in_seeds", ascending=False).to_csv(
        os.path.join(args.out, f"misclassified_images_{cfg}.csv"), index=False)
    print(f"  example figure: {path}")


if __name__ == "__main__":
    main()
