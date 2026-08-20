"""Definición de arquitecturas compartidas (MLP y CNN) para MNIST y CIFAR-10."""
import math

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector

from config import CNN_CONV1, CNN_CONV2, CNN_FC, MLP_HIDDENS, NUM_CLASSES

DATASET_INPUT = {
    "mnist": {"channels": 1, "features": 28 * 28, "size": 28},
    "cifar10": {"channels": 3, "features": 32 * 32 * 3, "size": 32},
}


class MLP(nn.Module):
    """Red totalmente conectada: Flat -> Dense -> ReLU -> ... -> Dense(out)."""

    def __init__(self, in_features, hidden=MLP_HIDDENS, num_classes=NUM_CLASSES):
        super().__init__()
        layers = [nn.Flatten()]
        prev = in_features
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def forward_features(self, x):
        for layer in self.net[:-1]:
            x = layer(x)
        return x


class CNN(nn.Module):
    """Red convolucional: 2x (Conv3x3 -> ReLU -> MaxPool) + capas densas."""

    def __init__(self, in_channels, num_classes=NUM_CLASSES, input_size=28):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, CNN_CONV1, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(CNN_CONV1, CNN_CONV2, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        size = (input_size - 2) // 2
        size = (size - 2) // 2
        flat = CNN_CONV2 * size * size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, CNN_FC),
            nn.ReLU(),
            nn.Linear(CNN_FC, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def forward_features(self, x):
        h = self.features(x)
        for layer in self.classifier[:-1]:
            h = layer(h)
        return h


def build_model(dataset, arch):
    """Devuelve una red (MLP o CNN) con init por defecto de PyTorch."""
    info = DATASET_INPUT[dataset]
    if arch == "mlp":
        return MLP(info["features"])
    if arch == "cnn":
        return CNN(info["channels"], input_size=info["size"])
    raise ValueError(f"Arquitectura desconocida: {arch}")


def init_weights_he(model):
    """Re-inicializa pesos con He (aprox.) para dar un buen arranque al GA."""
    for m in model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in = m.weight.size(1) if isinstance(m, nn.Linear) else (
                    m.weight.size(1) * m.weight.size(2) * m.weight.size(3))
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)


def params_to_vec(model):
    """Devuelve el vector plano de pesos+sesgos."""
    return parameters_to_vector(model.parameters())


def vec_to_params(model, vec):
    """Escribe el vector plano en los parámetros del modelo (copy in-place)."""
    vec = vec.to(device=next(model.parameters()).device)
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(vec[offset:offset + numel].view_as(p))
        offset += numel
