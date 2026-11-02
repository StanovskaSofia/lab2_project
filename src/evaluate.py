"""Оцінка accuracy на тестовому наборі (працює і локально, і через docker exec)."""
from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from src.data import KWSDataset, LogMel, list_files
from src.model import load_checkpoint


@torch.inference_mode()
def run_eval(model, featurizer, loader, device="cpu"):
    model.eval()
    preds, targets = [], []
    for wav, y in loader:
        logits = model(featurizer(wav.to(device)))
        preds.append(logits.argmax(1).cpu())
        targets.append(y)
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    return float((preds == targets).mean() * 100), preds, targets


def per_class_accuracy(preds, targets, classes):
    return {
        c: round(float((preds[targets == i] == i).mean() * 100), 2) if (targets == i).any() else None
        for i, c in enumerate(classes)
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.getenv("MODEL_PATH", "artifacts/model.pt"))
    ap.add_argument("--data-root", default=os.getenv("DATA_ROOT", "/tmp/data"))
    ap.add_argument("--max-per-class", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=128)
    a = ap.parse_args()

    model, classes = load_checkpoint(a.model)
    items = list_files(a.data_root, classes, "testing", a.max_per_class)
    loader = DataLoader(KWSDataset(items), batch_size=a.batch_size)
    acc, preds, targets = run_eval(model, LogMel(n_mels=model.n_mels), loader)
    print(f"test_samples={len(items)}")
    print(f"accuracy={acc:.2f}")
    print("per_class:", per_class_accuracy(preds, targets, classes))
    print("size_kb=%.1f" % (os.path.getsize(a.model) / 1024))
