"""Exporta las figuras de TensorBoard (confusión, UMAP, muestras) a PNG.

Uso:
  uv run python -m src.export_tb_images [--datasets mnist] [--seed 0]
"""
import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import config

RUN_DIRS = ["mlp_bp", "mlp_ga", "cnn_bp", "cnn_ga"]


def export_run(tb_dir, out_dir, only_tags=None):
    ea = EventAccumulator(str(tb_dir))
    ea.Reload()
    tags = ea.Tags().get("images", [])
    out_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    for tag in tags:
        if only_tags and not any(t in tag for t in only_tags):
            continue
        images = ea.Images(tag)
        if not images:
            continue
        last = images[-1]
        safe = tag.replace("/", "_")
        path = out_dir / f"{safe}.png"
        path.write_bytes(last.encoded_image_string)
        exported.append(path)
    return exported


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["mnist"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tags", nargs="+", default=["confusion", "umap"])
    args = parser.parse_args()

    for ds in args.datasets:
        for run in RUN_DIRS:
            tb_dir = config.TB_DIR / ds / f"{run}_seed{args.seed}"
            if not tb_dir.exists():
                continue
            out = config.PLOTS_DIR / "tb_export" / f"{ds}_{run}"
            paths = export_run(tb_dir, out, only_tags=args.tags)
            print(f"{ds}/{run}_seed{args.seed}: {len(paths)} imagenes -> {out}")
            for p in paths:
                print(f"   {p.name}")


if __name__ == "__main__":
    main()
