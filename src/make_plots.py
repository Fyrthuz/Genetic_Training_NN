"""Genera plots comparativos a partir de los resultados guardados.

Uso:
  uv run python -m src.make_plots
"""
import json
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

METHOD_LABEL = {"bp": "Backprop", "ga": "GA"}
ARCH_LABEL = {"mlp": "MLP", "cnn": "CNN"}
DATASET_TITLE = {"mnist": "MNIST", "cifar10": "CIFAR-10"}


def load_histories():
    hist = {}
    for ds in config.DATASETS:
        for arch in config.ARCHS:
            for method in config.METHODS:
                for seed in config.SEEDS:
                    p = config.RESULTS_DIR / ds / f"{arch}_{method}_seed{seed}" / "metrics.json"
                    if not p.exists():
                        continue
                    with open(p) as f:
                        r = json.load(f)
                    key = (ds, arch, method, seed)
                    hist[key] = r
    return hist


def plot_convergence(hist, out_dir, ds=None):
    """Curvas de convergencia por dataset y arquitectura (BP vs GA)."""
    datasets = [ds] if ds else config.DATASETS
    for ds in datasets:
        for arch in config.ARCHS:
            fig, ax = plt.subplots(figsize=(8, 5))
            for method in config.METHODS:
                for seed in config.SEEDS:
                    r = hist.get((ds, arch, method, seed))
                    if not r:
                        continue
                    if method == "bp":
                        x = [h["epoch"] for h in r["history"]]
                        y = [h["val_acc"] for h in r["history"]]
                    else:
                        x = [h["generation"] for h in r["history"]]
                        y = [h["best"] for h in r["history"]]
                    ax.plot(x, y, alpha=0.4, color="C0" if method == "bp" else "C1")
                    ax.plot([], [], color="C0" if method == "bp" else "C1",
                            label=METHOD_LABEL[method])
            ax.set_xlabel("Época (BP) / Generación (GA)")
            ax.set_ylabel("Accuracy (val/fitness)")
            ax.set_title(f"{DATASET_TITLE[ds]} - {ARCH_LABEL[arch]}: convergencia")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / f"convergence_{ds}_{arch}.png", dpi=120)
            plt.close(fig)


def plot_final_bars(hist, out_dir, ds=None):
    """Barras de accuracy test final (media de semillas) por dataset."""
    datasets = [ds] if ds else config.DATASETS
    for ds in datasets:
        fig, ax = plt.subplots(figsize=(8, 5))
        groups = []
        means, stds = [], []
        for arch in config.ARCHS:
            for method in config.METHODS:
                vals = [hist[(ds, arch, method, s)]["test_accuracy"]
                        for s in config.SEEDS if (ds, arch, method, s) in hist]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
                groups.append(f"{ARCH_LABEL[arch]}\n{METHOD_LABEL[method]}")
        x = np.arange(len(groups))
        bars = ax.bar(x, means, yerr=stds, capsize=5,
                      color=["#4C72B0", "#DD8452", "#4C72B0", "#DD8452"])
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{m:.2f}", ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_ylabel("Test accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{DATASET_TITLE[ds]}: accuracy test final")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"final_accuracy_{ds}.png", dpi=120)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=config.DATASETS)
    args = parser.parse_args()
    hist = load_histories()
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        plot_convergence(hist, config.PLOTS_DIR, ds)
        plot_final_bars(hist, config.PLOTS_DIR, ds)
    print("Plots guardados en", config.PLOTS_DIR)


if __name__ == "__main__":
    main()
