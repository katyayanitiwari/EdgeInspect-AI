"""Run training experiments on Kaggle GPU from the laptop (Kaggle CLI, no browser).

Needs Kaggle credentials in C:\\Users\\<you>\\.kaggle\\ : kaggle.json (legacy API key), or access_token + username
(new-style token; username in a one-line file named 'username')
and a phone-verified Kaggle account (required for GPU + internet).

Usage (project root):
  .\.venv\Scripts\python.exe scripts\kaggle_run.py upload                       # create/update dataset edgeinspect-neu-det
  .\.venv\Scripts\python.exe scripts\kaggle_run.py train EXP-001 --imgsz 640     # push + start notebook on a T4 GPU
  .\.venv\Scripts\python.exe scripts\kaggle_run.py status EXP-001
  .\.venv\Scripts\python.exe scripts\kaggle_run.py fetch EXP-001                # download + unzip to experiments/EXP-001
"""
import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPLOAD = ROOT / "deployment" / "kaggle_upload"
KAGGLE = str(Path(sys.executable).with_name("kaggle.exe" if sys.platform == "win32" else "kaggle"))
DATASET_SLUG = "edgeinspect-neu-det"


def username():
    """Kaggle username: from kaggle.json (legacy key), else ~/.kaggle/username (new access_token style)."""
    kdir = Path.home() / ".kaggle"
    if (kdir / "kaggle.json").exists():
        return json.loads((kdir / "kaggle.json").read_text())["username"]
    if (kdir / "access_token").exists() and (kdir / "username").exists():
        return (kdir / "username").read_text().strip()
    raise SystemExit(f"No Kaggle credentials in {kdir} (need kaggle.json, or access_token + username)")


def kaggle(*args):
    print(">", "kaggle", " ".join(args))
    r = subprocess.run([KAGGLE, *args], capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    print(out)
    if r.returncode != 0:
        raise SystemExit(f"kaggle command failed (exit {r.returncode})")
    return out


def kernel_slug(exp):
    return f"edgeinspect-{exp.lower()}"


def cmd_upload(_):
    user = username()
    d = UPLOAD / "dataset"
    if not (d / f"{DATASET_SLUG}.zip").exists():
        raise SystemExit("Bundle missing: run scripts/make_kaggle_bundle.py first")
    (d / "dataset-metadata.json").write_text(json.dumps({
        "title": "EdgeInspect NEU-DET", "id": f"{user}/{DATASET_SLUG}",
        "licenses": [{"name": "unknown"}]}, indent=2))
    r = subprocess.run([KAGGLE, "datasets", "status", f"{user}/{DATASET_SLUG}"], capture_output=True, text=True)
    if r.returncode == 0 and "ready" in r.stdout.lower():
        kaggle("datasets", "version", "-p", str(d), "-m", "update bundle", "-r", "skip")
    else:
        kaggle("datasets", "create", "-p", str(d), "-r", "skip")  # private by default


def cmd_train(a):
    user = username()
    kdir = UPLOAD / "kernel" / a.exp
    kdir.mkdir(parents=True, exist_ok=True)
    nb = json.loads((ROOT / "training" / "kaggle_train.ipynb").read_text(encoding="utf-8"))
    nb["cells"][1]["source"] = [f"EXP_ID = '{a.exp}'\n", f"IMGSZ = {a.imgsz}\n",
                                f"EPOCHS = {a.epochs}\n", f"BATCH = {a.batch}\n", f"MODEL = '{a.model}'\n"]
    train_cell = "".join(nb["cells"][4]["source"])
    if "--model" not in train_cell:
        train_cell = train_cell.replace("--name {EXP_ID}", "--name {EXP_ID} --model {MODEL}")
    nb["cells"][4]["source"] = train_cell.splitlines(keepends=True)
    (kdir / "kaggle_train.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (kdir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{kernel_slug(a.exp)}", "title": kernel_slug(a.exp),
        "code_file": "kaggle_train.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [f"{user}/{DATASET_SLUG}"]}, indent=2))
    kaggle("kernels", "push", "-p", str(kdir), "--accelerator", "NvidiaTeslaT4")


def cmd_status(a):
    kaggle("kernels", "status", f"{username()}/{kernel_slug(a.exp)}")


def cmd_fetch(a):
    user = username()
    tmp = UPLOAD / "output" / a.exp
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    kaggle("kernels", "output", f"{user}/{kernel_slug(a.exp)}", "-p", str(tmp))
    z = tmp / f"{a.exp}.zip"
    if not z.exists():
        raise SystemExit(f"{z.name} not in output; check the log files in {tmp}")
    dest = ROOT / "experiments" / a.exp
    if dest.exists():
        raise SystemExit(f"{dest} already exists; not overwriting")
    with zipfile.ZipFile(z) as f:
        # the notebook zips experiments/<EXP> with <EXP>/ as the top folder
        assert all(n.startswith(f"{a.exp}/") for n in f.namelist()), "unexpected zip layout"
        f.extractall(ROOT / "experiments")
    print(f"Unzipped to {dest.relative_to(ROOT)}: {sorted(p.name for p in dest.iterdir())}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("upload")
    t = sub.add_parser("train")
    t.add_argument("exp")
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--model", default="yolov8n.pt")
    for name in ("status", "fetch"):
        sub.add_parser(name).add_argument("exp")
    a = ap.parse_args()
    {"upload": cmd_upload, "train": cmd_train, "status": cmd_status, "fetch": cmd_fetch}[a.cmd](a)


if __name__ == "__main__":
    main()
