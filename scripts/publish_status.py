"""Rebuild the status PDF + website from measured files and publish both.

  1. reports/make_showcase.py --pdf --site
  2. copy site/index.html to the site repo (C:\\Users\\Katyayani\\edgeinspect-ai-site) and git push
     -> https://katyayanitiwari.github.io/edgeinspect-ai-site/  (GitHub Pages, public)
  3. copy the PDF to the Desktop code folder

Usage (project root):
  .\.venv\Scripts\python.exe scripts\publish_status.py
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE_REPO = Path.home() / "edgeinspect-ai-site"
DESKTOP_COPY = Path.home() / "OneDrive" / "Desktop" / "EdgeInspect-AI-code"
PDF = ROOT / "reports" / "EdgeInspect-AI_Project_Status.pdf"


def git(*args):
    return subprocess.run(["git", "-C", str(SITE_REPO), *args], capture_output=True, text=True)


def main():
    subprocess.run([sys.executable, str(ROOT / "reports" / "make_showcase.py"), "--pdf", "--site"], check=True)

    shutil.copy2(ROOT / "site" / "index.html", SITE_REPO / "index.html")
    git("add", "index.html")
    if git("diff", "--cached", "--quiet").returncode == 0:
        print("Website unchanged, nothing to push.")
    else:
        git("commit", "-m", "Status update " + time.strftime("%Y-%m-%d %H:%M"))
        r = git("push")
        print(r.stdout + r.stderr)
        if r.returncode != 0:
            raise SystemExit("git push failed (see message above)")
        print("Pushed. Live in about a minute: https://katyayanitiwari.github.io/edgeinspect-ai-site/")

    if DESKTOP_COPY.exists():
        shutil.copy2(PDF, DESKTOP_COPY / PDF.name)
        print(f"PDF copied to {DESKTOP_COPY / PDF.name}")


if __name__ == "__main__":
    main()
