"""
Training / validation curves from the console logs of arts_multiseed.py
(no retraining needed).

Usage (Windows, Anaconda Prompt):
    python plot_training_curves.py log1.txt log2.txt --out "D:\\my\\folder\\curves"
The --out folder is created if it does not exist.

The script
  * splits each log into "===== seed S | group =====" blocks,
  * splits each block into its training phases ("Epoch 1/..." restarts):
      baseline groups : [main training, pruning fine-tuning]
      two-stage groups: [stage 1 (no validation), stage 2, pruning fine-tuning]
  * saves every curve to CSV and draws, for baseline and SCL:
      (a) accuracy, (b) loss                    -> unpruned model
      (c) accuracy, (d) loss                    -> pruning fine-tuning
    All seeds are drawn as thin lines; the median-accuracy seed is drawn bold.
  * draws the SCL / CE stage-1 training loss separately.
"""
import argparse
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HEADER = re.compile(r"^=+\s*seed\s+(\d+)\s*\|\s*([\w]+)\s*=+")
EPOCH = re.compile(r"^Epoch\s+(\d+)/(\d+)")
METRIC = re.compile(r"\b(val_loss|val_accuracy|loss|accuracy):\s*([0-9.]+(?:e[-+]?\d+)?)")
RESULT = re.compile(r"\[(\w+)\s*\|\s*seed\s+(\d+)\]\s*acc=([0-9.]+)")
TWO_STAGE = {"scl", "ce_two_stage", "scl_balanced", "ce_two_stage_balanced"}
MARK_RESTORED = False
PRETTY = {"baseline": "End-to-end baseline", "scl": "SCL two-stage",
          "ce_two_stage": "CE two-stage"}


def parse(paths):
    """Return {(group, seed): [phase, ...]} with phase = list of metric dicts,
    and {(config, seed): test accuracy}."""
    runs, accs = {}, {}
    for p in paths:
        key, phases, cur = None, [], None
        with open(p, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                m = HEADER.match(line)
                if m:
                    if key and phases:
                        runs[key] = phases
                    key, phases, cur = (m.group(2), int(m.group(1))), [], None
                    continue
                m = EPOCH.match(line)
                if m and key:
                    if int(m.group(1)) == 1:
                        cur = []
                        phases.append(cur)
                    continue
                if cur is not None and "loss:" in line and "/step" in line:
                    vals = dict((k, float(v)) for k, v in METRIC.findall(line))
                    if vals:
                        cur.append(vals)
                    continue
                m = RESULT.search(line)
                if m:
                    accs[(m.group(1), int(m.group(2)))] = float(m.group(3))
        if key and phases:
            runs[key] = phases
    return runs, accs


def split_phases(group, phases):
    """Map phases to names, using the presence of validation metrics."""
    out = {}
    with_val = [ph for ph in phases if ph and "val_loss" in ph[0]]
    without_val = [ph for ph in phases if ph and "val_loss" not in ph[0]]
    if group in TWO_STAGE and without_val:
        out["stage1"] = without_val[0]
    if len(with_val) >= 1:
        out["main"] = with_val[0]
    if len(with_val) >= 2:
        out["pruning"] = with_val[1]
    return out


def save_csv(runs, out):
    rows = []
    for (group, seed), phases in sorted(runs.items()):
        for name, ph in split_phases(group, phases).items():
            for ep, vals in enumerate(ph, 1):
                rows.append({"group": group, "seed": seed, "phase": name, "epoch": ep, **vals})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "training_curves_all.csv"), index=False)
    return df


def median_seed(accs, cfg, seeds):
    vals = [(accs[(cfg, s)], s) for s in seeds if (cfg, s) in accs]
    if not vals:
        return seeds[0] if seeds else None
    vals.sort()
    return vals[(len(vals) - 1) // 2][1]


def seed_label(seeds):
    seeds = sorted(seeds)
    if seeds == list(range(seeds[0], seeds[-1] + 1)) and len(seeds) > 2:
        return f"{len(seeds)} seeds ({seeds[0]}\u2013{seeds[-1]})"
    return f"{len(seeds)} seeds (" + ", ".join(map(str, seeds)) + ")"


def panel(ax, curves, bold_seed, metric, ylabel, mark_epoch=None, legend_loc="best"):
    tr_c, va_c = "#1f5aa6", "#d1495b"
    seeds = [s for s, ph in curves.items() if ph and metric in ph[0]]
    slab = seed_label(seeds) if seeds else ""
    first_thin = True
    for seed, ph in curves.items():
        if not ph or metric not in ph[0]:
            continue
        ep = np.arange(1, len(ph) + 1)
        tr = [v.get(metric, np.nan) for v in ph]
        va = [v.get("val_" + metric, np.nan) for v in ph]
        bold = seed == bold_seed
        if bold:
            ax.plot(ep, tr, color=tr_c, lw=1.8, marker="o", ms=2.5, zorder=4,
                    label="Training, median-accuracy seed")
            ax.plot(ep, va, color=va_c, lw=1.8, marker="s", ms=2.5, zorder=4,
                    label="Validation, median-accuracy seed")
        else:
            ax.plot(ep, tr, color=tr_c, alpha=0.25, lw=0.8,
                    label=f"Training, all {slab}" if first_thin else None)
            ax.plot(ep, va, color=va_c, alpha=0.25, lw=0.8,
                    label=f"Validation, all {slab}" if first_thin else None)
            first_thin = False
        if bold and mark_epoch is None and MARK_RESTORED:
            # epoch restored by early stopping (lowest validation loss), shown as a
            # thin dotted vertical line (optional, --mark-restored)
            vl = [v.get("val_loss", np.nan) for v in ph]
            be = int(np.nanargmin(vl))
            ax.axvline(be + 1, color="#333333", ls=":", lw=1.0,
                       label="Lowest validation loss (restored weights)")
    if mark_epoch:
        ax.axvline(mark_epoch, color="#555555", ls="--", lw=0.9,
                   label="80% target sparsity reached")
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.grid(alpha=0.3)
    return ax.get_legend_handles_labels()


def figure_for(group, runs, accs, out, prune_ramp=20):
    seeds = sorted(s for (g, s) in runs if g == group)
    if not seeds:
        print(f"  no runs for {group}")
        return
    split = {s: split_phases(group, runs[(group, s)]) for s in seeds}
    main = {s: split[s].get("main") for s in seeds}
    prun = {s: split[s].get("pruning") for s in seeds}
    b_main = median_seed(accs, group, seeds)
    b_prun = median_seed(accs, "pruned_" + group, seeds)
    name = PRETTY.get(group, group)
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.6))
    h1, l1 = panel(axs[0, 0], main, b_main, "accuracy", "Accuracy")
    panel(axs[0, 1], main, b_main, "loss", "Loss")
    h2, l2 = panel(axs[1, 0], prun, b_prun, "accuracy", "Accuracy", mark_epoch=prune_ramp)
    panel(axs[1, 1], prun, b_prun, "loss", "Loss", mark_epoch=prune_ramp)
    titles = {"a": f"(a) Unpruned model: accuracy (bold = seed {b_main})",
              "b": f"(b) Unpruned model: loss (bold = seed {b_main})",
              "c": f"(c) Pruning fine-tuning: accuracy (bold = seed {b_prun})",
              "d": f"(d) Pruning fine-tuning: loss (bold = seed {b_prun})"}
    for ax, lab in zip(axs.flat, "abcd"):
        ax.set_title(titles[lab], loc="left", fontweight="bold", fontsize=10.5)
    # one shared legend below the panels, so no data are hidden
    hs, ls = [], []
    for h, l in list(zip(h1, l1)) + list(zip(h2, l2)):
        if l not in ls:
            hs.append(h); ls.append(l)
    order = sorted(range(len(ls)), key=lambda i: (0 if "median" in ls[i] else
                                                   1 if "all" in ls[i] else 2))
    fig.legend([hs[i] for i in order], [ls[i] for i in order], loc="lower center",
               ncol=3, fontsize=9.5, frameon=True, bbox_to_anchor=(0.5, 0.0))
    # no title inside the figure: the LaTeX caption carries this information
    print(f"  {name}: bold seed (unpruned) = {b_main}, bold seed (pruned) = {b_prun}")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    base = os.path.join(out, f"curves_{group}")
    fig.savefig(base + ".png", dpi=300)
    fig.savefig(base + ".pdf")
    plt.close(fig)
    print(f"  {base}.png  (seeds {seeds})")


def stage1_figure(runs, out):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    any_ = False
    for group, col in [("scl", "#6a3d9a"), ("ce_two_stage", "#33a02c")]:
        seeds = sorted(s for (g, s) in runs if g == group)
        for i, s in enumerate(seeds):
            st1 = split_phases(group, runs[(group, s)]).get("stage1")
            if not st1:
                continue
            any_ = True
            ax.plot(np.arange(1, len(st1) + 1), [v["loss"] for v in st1], color=col,
                    alpha=0.5, lw=1, label=f"{PRETTY[group]} (stage 1)" if i == 0 else None)
    if not any_:
        plt.close(fig)
        return
    ax.set_xlabel("Epoch"); ax.set_ylabel("Training loss"); ax.set_yscale("log")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("First-stage training loss (all seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "curves_stage1.png"), dpi=300)
    fig.savefig(os.path.join(out, "curves_stage1.pdf"))
    plt.close(fig)
    print(f"  {os.path.join(out, 'curves_stage1.png')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="console log file(s) of arts_multiseed.py")
    ap.add_argument("--out", default="curves")
    ap.add_argument("--groups", nargs="+", default=["baseline", "scl", "ce_two_stage"])
    ap.add_argument("--mark-restored", action="store_true",
                    help="draw a dotted vertical line at the epoch restored by early stopping "
                         "(off by default)")
    args = ap.parse_args()
    global MARK_RESTORED
    MARK_RESTORED = args.mark_restored
    args.out = os.path.abspath(args.out)
    os.makedirs(args.out, exist_ok=True)
    runs, accs = parse(args.logs)
    print(f"parsed {len(runs)} (group, seed) blocks from {len(args.logs)} log file(s)")
    df = save_csv(runs, args.out)
    if not df.empty:
        print(df.groupby(["group", "phase"]).seed.nunique().rename("n_seeds").to_string())
    for g in args.groups:
        figure_for(g, runs, accs, args.out)
    stage1_figure(runs, args.out)
    print(f"\nAll figures and the CSV are saved in: {args.out}")


if __name__ == "__main__":
    main()
