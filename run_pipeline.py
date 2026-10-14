"""End-to-end CWRU bearing study.

    python run_pipeline.py                 # full study (about 2-4 minutes on a laptop CPU)
    python run_pipeline.py --quick         # fewer windows, for a fast smoke test

Writes results/metrics.json, results/*.csv and results/figures/*.png.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from bearing.data import CLASSES, FS, LOADS, load_all, windows  # noqa: E402
from bearing.evaluate import leave_one_load_out, leave_one_size_out, random_window_split  # noqa: E402
from bearing.features import FEATURE_GROUPS, envelope_spectrum, extract  # noqa: E402


def build_table(data_dir: Path, size: int, step: int, max_windows: int | None) -> pd.DataFrame:
    rows = []
    for rec in load_all(data_dir):
        W = windows(rec.signal, size=size, step=step)
        if max_windows:
            W = W[:max_windows]
        for i, w in enumerate(W):
            f = extract(w, FS, rec.rpm)
            f.update(label=rec.label, load=rec.load, window=i, rpm=rec.rpm)
            rows.append(f)
    return pd.DataFrame(rows)


def models(seed: int):
    return {
        "RandomForest": RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=seed),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=seed),
        "SVM-RBF": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")),
    }


def plot_confusion(cm: np.ndarray, labels: list[str], path: Path, title: str):
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cmn[i, j] >= 0.01:
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                        color="white" if cmn[i, j] > 0.6 else "black", fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_envelopes(data_dir: Path, path: Path):
    from bearing.data import load_recording
    from bearing.features import FAULT_ORDERS
    fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
    for ax, lab in zip(axes, ["Normal", "IR021", "OR021@6", "B021"]):
        rec = load_recording(data_dir, lab, 0)
        f, a = envelope_spectrum(rec.signal[: FS * 2], FS)
        ax.plot(f, a, lw=0.8)
        fr = rec.rpm / 60
        for name, order in [("BPFI", FAULT_ORDERS["BPFI"]), ("BPFO", FAULT_ORDERS["BPFO"]), ("BSF", FAULT_ORDERS["BSF"])]:
            ax.axvline(order * fr, ls="--", lw=0.8, color="tab:red" if name == "BPFI" else "tab:green" if name == "BPFO" else "tab:purple")
        ax.set_xlim(0, 500); ax.set_ylabel(lab)
    axes[0].set_title("Envelope spectra (2-5 kHz band), 0 hp. Dashed: BPFI red, BPFO green, BSF purple")
    axes[-1].set_xlabel("Frequency (Hz)")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "cwru"))
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--step", type=int, default=1024, help="window hop; 1024 = 50% overlap")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = ROOT / "results"; figs = out / "figures"; figs.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    table = build_table(Path(args.data_dir), args.window, args.step, 25 if args.quick else None)
    table.to_csv(out / "features.csv", index=False)
    feat_cols = [c for c in table.columns if c not in {"label", "load", "window", "rpm"}]
    X, y, loads = table[feat_cols], table["label"], table["load"]
    print(f"{len(table):,} windows x {len(feat_cols)} features from {table.groupby(['label','load']).ngroups} recordings ({time.time()-t0:.1f}s)")

    results = {"n_windows": int(len(table)), "n_features": len(feat_cols), "window": args.window, "step": args.step,
               "classes": CLASSES, "loads_hp": LOADS, "protocols": {}}
    fold_tables, size_tables, best = [], [], (None, -1, None, None)
    for name, model in models(args.seed).items():
        rnd = random_window_split(model, X, y, seed=args.seed)
        lolo, folds, cm = leave_one_load_out(model, X, y, loads, CLASSES)
        folds.insert(0, "model", name); fold_tables.append(folds)
        loso, sfolds, scm, slabels = leave_one_size_out(model, X, y, loads)
        sfolds.insert(0, "model", name); size_tables.append(sfolds)
        results["protocols"][name] = {"random_window_split": rnd, "leave_one_load_out": lolo,
                                      "leave_one_fault_size_out_4class": loso}
        print(f"{name:22s} random {rnd['accuracy']:.4f} | load-out {lolo['accuracy_mean']:.4f} | size-out (4-class) {loso['accuracy_mean']:.4f} (worst {loso['accuracy_min']:.4f})")
        if loso["accuracy_mean"] > best[1]:
            best = (name, loso["accuracy_mean"], scm, cm)
    pd.concat(fold_tables).to_csv(out / "leave_one_load_out_folds.csv", index=False)
    pd.concat(size_tables).to_csv(out / "leave_one_fault_size_out_folds.csv", index=False)

    # Feature-group ablation with the best model under the honest protocol.
    ablation = {}
    best_model = models(args.seed)[best[0]]
    for group, keep in FEATURE_GROUPS.items():
        cols = [c for c in feat_cols if keep(c)]
        s, _, _, _ = leave_one_size_out(best_model, X[cols], y, loads)
        ablation[group] = {"n_features": len(cols), **s}
        print(f"  ablation {group:9s} ({len(cols):2d} feats): size-out acc {s['accuracy_mean']:.4f}")
    results["ablation_leave_one_fault_size_out"] = {"model": best[0], **ablation}
    results["best_model"] = best[0]

    plot_confusion(best[3], CLASSES, figs / "confusion_leave_one_load_out.png",
                   f"{best[0]}: leave-one-load-out, 10 classes (pooled)")
    plot_confusion(best[2], ["Normal", "Ball", "InnerRace", "OuterRace"], figs / "confusion_leave_one_fault_size_out.png",
                   f"{best[0]}: unseen fault size, 4 classes (pooled)")
    plot_envelopes(Path(args.data_dir), figs / "envelope_spectra.png")
    names = list(results["protocols"])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    xs = np.arange(len(names))
    for off, key, lab, sub in [(-0.27, "random_window_split", "random window split (leaky)", "accuracy"),
                               (0.0, "leave_one_load_out", "unseen motor load", "accuracy_mean"),
                               (0.27, "leave_one_fault_size_out_4class", "unseen fault size (bearing)", "accuracy_mean")]:
        ax.bar(xs + off, [results["protocols"][n][key][sub] for n in names], 0.27, label=lab)
    ax.set_xticks(xs, names); ax.set_ylim(0.0, 1.05); ax.set_ylabel("accuracy"); ax.legend(loc="lower right")
    ax.set_title("Same models, two evaluation protocols"); fig.tight_layout(); fig.savefig(figs / "protocol_comparison.png", dpi=140); plt.close(fig)

    results["runtime_s"] = round(time.time() - t0, 1)
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"done in {results['runtime_s']}s -> {out}")


if __name__ == "__main__":
    main()
