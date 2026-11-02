"""Дані: Google Speech Commands -> log-Mel спектрограми."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import soundfile as sf
import torch
import torchaudio.transforms as T
from torch import nn
from torch.utils.data import Dataset

SAMPLE_RATE = 16_000
N_SAMPLES = SAMPLE_RATE  # 1 секунда аудіо
N_MELS = 40
DEFAULT_CLASSES = ["yes", "no", "up", "down"]


def fix_length(wav: torch.Tensor, n: int = N_SAMPLES) -> torch.Tensor:
    """Обрізає або доповнює нулями до рівно n семплів."""
    if wav.shape[-1] >= n:
        return wav[..., :n]
    return nn.functional.pad(wav, (0, n - wav.shape[-1]))


class LogMel(nn.Module):
    """waveform (B, T) -> нормалізована log-Mel спектрограма (B, 1, n_mels, frames)."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, n_mels: int = N_MELS):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=512, hop_length=160, n_mels=n_mels
        )
        self.to_db = T.AmplitudeToDB()

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.to_db(self.mel(wav)).unsqueeze(1)
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        return (x - mean) / (std + 1e-5)


def download(root: str) -> None:
    from torchaudio.datasets import SPEECHCOMMANDS

    os.makedirs(root, exist_ok=True)
    SPEECHCOMMANDS(root, download=True)


def list_files(
    root: str, classes: list[str], subset: str, max_per_class: int = 0
) -> list[tuple[str, int]]:
    """Повертає [(шлях_до_wav, індекс_класу)] для training / validation / testing."""
    from torchaudio.datasets import SPEECHCOMMANDS

    os.makedirs(root, exist_ok=True)
    ds = SPEECHCOMMANDS(root, download=True, subset=subset)
    counts = {c: 0 for c in classes}
    items: list[tuple[str, int]] = []
    for p in ds._walker:  # відсортований список шляхів
        label = Path(p).parent.name
        if label not in counts:
            continue
        if max_per_class and counts[label] >= max_per_class:
            continue
        counts[label] += 1
        items.append((str(p), classes.index(label)))
    return items


class KWSDataset(Dataset):
    def __init__(self, items: list[tuple[str, int]], train: bool = False):
        self.items = items
        self.train = train

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, y = self.items[i]
        wav, _ = sf.read(path, dtype="float32")
        wav = fix_length(torch.from_numpy(wav))
        if self.train:  # проста аугментація: випадковий зсув до ±0.1 с
            shift = int(torch.randint(-1600, 1600, (1,)))
            wav = torch.roll(wav, shift)
        return wav, y


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--data-root", default="data")
    a = ap.parse_args()
    if a.download:
        download(a.data_root)
        print("dataset ready in", a.data_root)
