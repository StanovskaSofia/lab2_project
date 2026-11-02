"""Архітектура CNN для класифікації команд."""
from __future__ import annotations

import torch
from torch import nn

from src.data import N_MELS


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class KWSNet(nn.Module):
    ARCHITECTURE = "KWSNet: 3x[Conv2d3x3+BN+ReLU+MaxPool] (16/32/64) + GAP + Dropout + Linear"

    def __init__(self, n_classes: int, n_mels: int = N_MELS):
        super().__init__()
        self.features = nn.Sequential(_block(1, 16), _block(16, 32), _block(32, 64))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, n_classes)
        )
        self.n_mels = n_mels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(path: str, model: KWSNet, classes: list[str]) -> None:
    torch.save(
        {"state_dict": model.state_dict(), "classes": classes, "n_mels": model.n_mels}, path
    )


def load_checkpoint(path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = KWSNet(len(ckpt["classes"]), ckpt["n_mels"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["classes"]
