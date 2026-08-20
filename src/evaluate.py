"""Métricas: accuracy, precision/recall/f1 por clase, matriz de confusión."""
import numpy as np
import torch


def predict(model, x, device):
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        return logits.argmax(dim=1).cpu()


def extract_features(model, loader, device, max_samples=1500):
    """Features de la capa pre-logits, etiquetas e imágenes (submuestreadas)."""
    model.eval()
    feats, labels, imgs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            f = model.forward_features(x.to(device)).cpu()
            feats.append(f)
            labels.append(y)
            imgs.append(x)
            total = sum(i.size(0) for i in feats)
            if total >= max_samples:
                break
    feats = torch.cat(feats)[:max_samples]
    labels = torch.cat(labels)[:max_samples]
    imgs = torch.cat(imgs)[:max_samples]
    return feats, labels, imgs


def confusion_matrix(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[t, p] += 1
    return cm


def per_class_metrics(cm):
    """precision[i], recall[i], f1[i] por clase y su media macro."""
    n = cm.shape[0]
    prec = np.zeros(n)
    rec = np.zeros(n)
    f1 = np.zeros(n)
    col_sum = cm.sum(axis=0)
    row_sum = cm.sum(axis=1)
    for i in range(n):
        prec[i] = cm[i, i] / col_sum[i] if col_sum[i] > 0 else 0.0
        rec[i] = cm[i, i] / row_sum[i] if row_sum[i] > 0 else 0.0
        f1[i] = (2 * prec[i] * rec[i] / (prec[i] + rec[i])) if (prec[i] + rec[i]) > 0 else 0.0
    return {
        "per_class_precision": prec.tolist(),
        "per_class_recall": rec.tolist(),
        "per_class_f1": f1.tolist(),
        "precision": float(prec.mean()),
        "recall": float(rec.mean()),
        "f1": float(f1.mean()),
    }


def evaluate(model, loader, device):
    """Devuelve accuracy, metrics por clase y matriz de confusión sobre el loader."""
    model.eval()
    all_pred = []
    all_true = []
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            pred = out.argmax(dim=1).cpu()
            correct += (pred == y).sum().item()
            total += y.size(0)
            all_pred.append(pred)
            all_true.append(y)
    y_pred = torch.cat(all_pred)
    y_true = torch.cat(all_true)
    cm = confusion_matrix(y_true, y_pred)
    metrics = per_class_metrics(cm)
    return {
        "accuracy": correct / total,
        "cm": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        **metrics,
    }
