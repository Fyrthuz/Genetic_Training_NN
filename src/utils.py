"""Utilidades: seeds, device, carga de datos y splits fijos compartidos."""
import random

import numpy as np
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

from config import (
    DATASET_MEAN_STD,
    SPLIT_SEED,
    TEST_FRACTION,
    VAL_FRACTION,
)

DATASET_CLASSES = {
    "mnist": [str(i) for i in range(10)],
    "cifar10": [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ],
}

_ds_cache = {}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_dataset(dataset):
    if dataset in _ds_cache:
        return _ds_cache[dataset]
    mean, std = DATASET_MEAN_STD[dataset]
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_ds = datasets.__dict__[dataset.upper()](
        root="data", train=True, download=True, transform=t
    )
    test_ds = datasets.__dict__[dataset.upper()](
        root="data", train=False, download=True, transform=t
    )
    _ds_cache[dataset] = (train_ds, test_ds)
    return train_ds, test_ds


def _fixed_permutation(n, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randperm(n, generator=g)


def get_splits(dataset):
    """Devuelve (train, val, test) subsets con split fijo 70/15/15 (semilla global).

    Todos los runs de un mismo dataset comparten exactamente los mismos índices.
    """
    train_ds, test_ds = _load_dataset(dataset)
    n = len(train_ds)
    perm = _fixed_permutation(n, SPLIT_SEED)
    n_val = int(n * VAL_FRACTION)
    n_test = int(n * TEST_FRACTION)
    val_idx = perm[:n_val]
    test_idx = perm[n_val:n_val + n_test]
    train_idx = perm[n_val + n_test:]
    return (
        Subset(train_ds, train_idx.tolist()),
        Subset(train_ds, val_idx.tolist()),
        Subset(train_ds, test_idx.tolist()),
    )


def get_class_names(dataset):
    return DATASET_CLASSES[dataset]
