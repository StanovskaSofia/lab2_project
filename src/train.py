"""Навчання + оцінка -> artifacts/{model.pt, labels.json, metrics.json}."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import DEFAULT_CLASSES, N_MELS, KWSDataset, LogMel, list_files
from src.evaluate import per_class_accuracy, run_eval
from src.model import KWSNet, count_params, save_checkpoint


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-per-class", type=int, default=0, help="0 = без обмеження")
    ap.add_argument("--model-version", default="1.0.0")
    ap.add_argument("--data-root", default=os.getenv("DATA_ROOT", "data"))
    ap.add_argument("--out", default="artifacts")
    a = ap.parse_args()

    set_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    sets = {
        s: list_files(a.data_root, a.classes, s, a.max_per_class)
        for s in ("training", "validation", "testing")
    }
    print({k: len(v) for k, v in sets.items()})

    g = torch.Generator().manual_seed(a.seed)
    train_dl = DataLoader(KWSDataset(sets["training"], train=True), a.batch_size,
                          shuffle=True, generator=g)
    val_dl = DataLoader(KWSDataset(sets["validation"]), 128)
    test_dl = DataLoader(KWSDataset(sets["testing"]), 128)

    feat = LogMel(n_mels=N_MELS).to(device)
    model = KWSNet(len(a.classes), N_MELS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_val, best_state = -1.0, None
    for epoch in range(1, a.epochs + 1):
        model.train()
        total = 0.0
        for wav, y in tqdm(train_dl, desc=f"epoch {epoch}/{a.epochs}", leave=False):
            wav, y = wav.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(feat(wav)), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        sched.step()
        val_acc, _, _ = run_eval(model, feat, val_dl, device)
        print(f"epoch {epoch}: loss={total / len(train_dl.dataset):.4f} val_acc={val_acc:.2f}")
        if val_acc > best_val:
            best_val, best_state = val_acc, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.to("cpu")
    test_acc, preds, targets = run_eval(model, LogMel(n_mels=N_MELS), test_dl, "cpu")
    print(f"TEST accuracy = {test_acc:.2f}%")

    model_path = out / "model.pt"
    save_checkpoint(str(model_path), model, a.classes)
    (out / "labels.json").write_text(json.dumps(a.classes, ensure_ascii=False))
    metrics = {
        "model_version": a.model_version,
        "architecture": KWSNet.ARCHITECTURE,
        "classes": a.classes,
        "params": count_params(model),
        "accuracy": round(test_acc, 2),  # у відсотках
        "val_accuracy": round(best_val, 2),
        "test_samples": len(sets["testing"]),
        "per_class_accuracy": per_class_accuracy(preds, targets, a.classes),
        "confusion_matrix": confusion_matrix(targets, preds).tolist(),
        "epochs": a.epochs,
        "seed": a.seed,
        "model_size_kb": round(os.path.getsize(model_path) / 1024, 1),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("saved to", out)


if __name__ == "__main__":
    main()
