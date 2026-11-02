"""Копіює по одному тестовому .wav на клас у samples/ (yes.wav, no.wav, ...)."""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import DEFAULT_CLASSES, list_files  # noqa: E402

root = sys.argv[1] if len(sys.argv) > 1 else "data"
seen: set[int] = set()
for path, y in list_files(root, DEFAULT_CLASSES, "testing"):
    if y not in seen:
        seen.add(y)
        dst = Path("samples") / f"{DEFAULT_CLASSES[y]}.wav"
        shutil.copy(path, dst)
        print("saved", dst)
    if len(seen) == len(DEFAULT_CLASSES):
        break
