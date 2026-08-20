"""Helpers de logging a TensorBoard en tiempo real."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def make_writer(logdir):
    return SummaryWriter(logdir)


def log_hparams(writer, hparams, metrics=None):
    writer.add_hparams(hparams, metrics or {"placeholder": 0.0})


def log_text(writer, tag, text, step=0):
    writer.add_text(tag, text, step)


def log_grad_norm(writer, step, model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.norm().item() ** 2
    writer.add_scalar("train/grad_norm", float(np.sqrt(total)), step)


def log_grad_histograms(writer, step, model):
    for name, p in model.named_parameters():
        if p.grad is not None:
            writer.add_histogram(f"gradients/{name}", p.grad.detach().cpu(), step)


def log_weight_histograms(writer, step, model):
    for name, p in model.named_parameters():
        writer.add_histogram(f"weights/{name}", p.detach().cpu(), step)


def log_ga_weight_histograms(writer, step, population_vecs, max_samples=200_000):
    """Histograma de la distribución de pesos en toda la población (GA)."""
    stack = torch.stack(population_vecs)
    writer.add_scalar("ga/diversity", float(stack.detach().std()), step)
    flat = stack.detach().flatten()
    if flat.numel() > max_samples:
        idx = torch.randperm(flat.numel())[:max_samples]
        flat = flat[idx]
    writer.add_histogram("ga/weight_distribution", flat.cpu().numpy(), step)


def _make_prediction_figure(images, true, pred, class_names, top_k=None):
    """Figura con imágenes reales y sus etiquetas reales vs predichas."""
    n = len(images)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.atleast_1d(axes).ravel()
    for i in range(n):
        ax = axes[i]
        img = images[i]
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = img.permute(1, 2, 0)
        if img.ndim == 3 and img.shape[2] == 1:
            img = img[..., 0]
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(
            f"T:{class_names[true[i]]}\nP:{class_names[pred[i]]}", fontsize=8
        )
        ax.axis("off")
    for i in range(n, len(axes)):
        axes[i].axis("off")
    fig.tight_layout()
    return fig


def log_predictions(writer, tag, step, images, true, pred, class_names):
    fig = _make_prediction_figure(images, true, pred, class_names)
    writer.add_figure(tag, fig, step)
    plt.close(fig)


def log_confusion(writer, tag, step, cm, class_names):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6)
    fig.colorbar(im)
    fig.tight_layout()
    writer.add_figure(tag, fig, step)
    plt.close(fig)


def log_embedding(writer, tag, features, labels, metadata, step=0):
    writer.add_embedding(
        features.detach().cpu(),
        metadata=labels.cpu().tolist(),
        label_img=metadata,
        global_step=step,
        tag=tag,
    )


def log_umap(writer, tag, features, labels, class_names, step=0, n_neighbors=15):
    """Proyección UMAP 2D de las features coloreada por clase, como figura."""
    try:
        import umap
    except ImportError:
        writer.add_text(tag + "/error", "umap-learn no instalado", step)
        return
    feats = features.detach().cpu().numpy()
    y = labels.cpu().numpy()
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, random_state=0)
    emb = reducer.fit_transform(feats)

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap("tab10")
    for c in np.unique(y):
        idx = y == c
        ax.scatter(emb[idx, 0], emb[idx, 1], s=8, alpha=0.6,
                   c=[cmap(c % 10)], label=class_names[c])
    ax.set_title("UMAP de embeddings (color por clase)")
    ax.legend(fontsize=8, loc="best")
    ax.axis("off")
    fig.tight_layout()
    writer.add_figure(tag, fig, step)
    plt.close(fig)
