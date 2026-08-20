"""Orquestador del experimento: ejecuta las corridas BP/GA y recopila resultados.

Uso:
  uv run python -m src.run_experiments                      # las 8 corridas
  uv run python -m src.run_experiments --datasets mnist      # solo MNIST
  uv run python -m src.run_experiments --archs mlp --seed 0  # selección fina
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

import config
from src.ga import run_ga
from src.train_backprop import run_bp


def run_single(dataset, arch, method, seed, force=False):
    run_dir = config.RESULTS_DIR / dataset / f"{arch}_{method}_seed{seed}"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not force:
        with open(metrics_path) as f:
            return json.load(f)

    logdir = config.TB_DIR / dataset / f"{arch}_{method}_seed{seed}"
    t0 = time.time()
    if method == "bp":
        result, model, best_state = run_bp(dataset, arch, seed, str(logdir))
        run_dir.mkdir(parents=True, exist_ok=True)
        model.load_state_dict(best_state)
    elif method == "ga":
        result, model, best_genome = run_ga(dataset, arch, seed, str(logdir))
        run_dir.mkdir(parents=True, exist_ok=True)
        from src.models import vec_to_params
        vec_to_params(model, best_genome)
    else:
        raise ValueError(method)

    result["wall_time_s"] = time.time() - t0
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)
    torch_state = model.state_dict()
    torch.save(torch_state, run_dir / "model.pt")
    return result


def build_summary(results):
    rows = []
    for r in results:
        rows.append({
            "dataset": r["dataset"], "arch": r["arch"], "method": r["method"],
            "seed": r["seed"], "test_accuracy": r["test_accuracy"],
            "test_precision": r["test_precision"], "test_recall": r["test_recall"],
            "test_f1": r["test_f1"], "wall_time_s": r["wall_time_s"],
        })
    df = pd.DataFrame(rows)
    agg = df.groupby(["dataset", "arch", "method"])["test_accuracy"].agg(
        ["mean", "std"]
    ).reset_index()
    return df, agg


def main():
    parser = argparse.ArgumentParser(description="Comparativa BP vs GA")
    parser.add_argument("--datasets", nargs="+", default=config.DATASETS)
    parser.add_argument("--archs", nargs="+", default=config.ARCHS)
    parser.add_argument("--methods", nargs="+", default=config.METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    parser.add_argument("--force", action="store_true", help="Re-ejecutar aunque exista")
    args = parser.parse_args()

    results = []
    for dataset in args.datasets:
        for arch in args.archs:
            for method in args.methods:
                for seed in args.seeds:
                    tag = f"{dataset}/{arch}_{method}_seed{seed}"
                    print(f"[{time.strftime('%H:%M:%S')}] Corriendo {tag}")
                    results.append(run_single(dataset, arch, method, seed, args.force))

    df, agg = build_summary(results)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RESULTS_DIR / "summary.csv", index=False)
    agg.to_csv(config.RESULTS_DIR / "summary_agg.csv", index=False)
    print("\n=== Resumen (accuracy test por config) ===")
    print(df.to_string(index=False))
    print("\n=== Media ± desviación ===")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
