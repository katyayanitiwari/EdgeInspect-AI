"""Build the Kaggle dataset zip: training code + processed NEU-DET, with forward-slash paths.

Why forward slashes: Kaggle runs Linux. A zip written with Windows backslashes unpacks into single
files literally named 'EdgeInspect-AI\\dataset\\...' instead of folders.

Usage (project root):
  .\.venv\Scripts\python.exe scripts\make_kaggle_bundle.py
Output: deployment/kaggle_upload/dataset/edgeinspect-neu-det.zip
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deployment" / "kaggle_upload" / "dataset" / "edgeinspect-neu-det.zip"
DATA = ROOT / "dataset" / "processed" / "neu_det"
FILES = [ROOT / "training" / "train.py", ROOT / "training" / "neu_det.yaml"]


def main():
    files = FILES + sorted(p for sub in ("images", "labels") for p in (DATA / sub).rglob("*") if p.is_file())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, "EdgeInspect-AI/" + p.relative_to(ROOT).as_posix())

    # verify: no backslashes, and every split has matching image/label counts
    with zipfile.ZipFile(OUT) as z:
        names = z.namelist()
    assert not any("\\" in n for n in names), "backslash in zip path"
    for split in ("train", "val", "test"):
        n_img = sum(f"/images/{split}/" in n for n in names)
        n_lbl = sum(f"/labels/{split}/" in n for n in names)
        print(f"{split:<6} images {n_img:>5}  labels {n_lbl:>5}")
        assert n_img == n_lbl > 0, f"{split}: images/labels mismatch"
    print(f"{len(names)} files, {OUT.stat().st_size / 1e6:.1f} MB -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
