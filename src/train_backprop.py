"""Entrenamiento con backpropagation."""
import time

import torch
from torch.utils.data import DataLoader

from config import (
    BP_BATCH_SIZE,
    BP_EPOCHS,
    BP_LR,
    BP_METRICS_EVERY,
    BP_PATIENCE,
    BP_TRAIN_LOG_IMAGES_EVERY,
    BP_WEIGHT_DECAY,
)
from src import tensorboard as tb
from src.evaluate import evaluate
from src.models import build_model
from src.utils import get_class_names, get_device, get_splits, set_seed


def run_bp(dataset, arch, seed, logdir):
    set_seed(seed)
    device = get_device()
    train, val, test = get_splits(dataset)
    class_names = get_class_names(dataset)

    model = build_model(dataset, arch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=BP_LR, weight_decay=BP_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=BP_PATIENCE // 2, factor=0.5
    )
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(train, batch_size=BP_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val, batch_size=BP_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test, batch_size=BP_BATCH_SIZE, shuffle=False)

    writer = tb.make_writer(logdir)
    tb.log_text(writer, "architecture", str(model))
    tb.log_hparams(
        writer,
        {
            "dataset": dataset, "arch": arch, "method": "bp",
            "epochs": BP_EPOCHS, "lr": BP_LR, "batch_size": BP_BATCH_SIZE,
            "weight_decay": BP_WEIGHT_DECAY, "seed": seed,
        },
    )

    history = []
    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(BP_EPOCHS):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += x.size(0)
            tb.log_grad_norm(writer, epoch, model)
        train_loss = running_loss / total
        train_acc = correct / total

        val_res = evaluate(model, val_loader, device)
        val_acc = val_res["accuracy"]
        val_loss = _compute_loss(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        cur_lr = optimizer.param_groups[0]["lr"]

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)
        writer.add_scalar("optim/learning_rate", cur_lr, epoch)
        tb.log_grad_histograms(writer, epoch, model)
        tb.log_weight_histograms(writer, epoch, model)

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_acc": val_acc, "lr": cur_lr, "epoch_time": time.time() - t0,
        })

        if epoch % BP_TRAIN_LOG_IMAGES_EVERY == 0:
            _log_train_samples(writer, epoch, model, val, class_names)

        if epoch % BP_METRICS_EVERY == 0 or epoch == BP_EPOCHS - 1:
            writer.add_scalar("val/precision", val_res["precision"], epoch)
            writer.add_scalar("val/recall", val_res["recall"], epoch)
            writer.add_scalar("val/f1", val_res["f1"], epoch)
            tb.log_confusion(writer, "val/confusion", epoch, val_res["cm"], class_names)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= BP_PATIENCE:
            break

    model.load_state_dict(best_state)
    test_res = evaluate(model, test_loader, device)
    tb.log_confusion(writer, "test/confusion", len(history) - 1, test_res["cm"], class_names)

    _log_umap(writer, model, test, device, class_names)

    return {
        "dataset": dataset, "arch": arch, "method": "bp", "seed": seed,
        "best_val_acc": best_val_acc,
        "test_accuracy": test_res["accuracy"],
        "test_precision": test_res["precision"],
        "test_recall": test_res["recall"],
        "test_f1": test_res["f1"],
        "epochs_run": len(history),
        "history": history,
    }, model, best_state


def _compute_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += criterion(out, y).item() * x.size(0)
            total += x.size(0)
    return total_loss / total


def _log_umap(writer, model, test, device, class_names):
    from src.evaluate import extract_features
    loader = DataLoader(test, batch_size=128, shuffle=False)
    feats, labels, imgs = extract_features(model, loader, device)
    tb.log_umap(writer, "embeddings/umap", feats, labels, class_names)
    tb.log_embedding(writer, "embeddings/projector", feats, labels, imgs)
    del feats, labels, imgs


def _log_train_samples(writer, step, model, val, class_names):
    model.eval()
    n = 8
    imgs, true = [], []
    for i in range(n):
        x, y = val[i]
        imgs.append(x)
        true.append(y)
    xb = torch.stack(imgs).to(next(model.parameters()).device)
    with torch.no_grad():
        pred = model(xb).argmax(dim=1).cpu().tolist()
    tb.log_predictions(writer, "bp/val_samples", step, imgs, true, pred, class_names)
