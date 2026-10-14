"""Download the 40 CWRU .mat files this study uses (about 90 MB).

Source: the public GitHub mirror XiongMeijing/CWRU-1, which stores the original
Case Western Reserve University Bearing Data Center files. Works on Windows,
macOS and Linux with only the Python standard library.

    python scripts/download_data.py
"""
from __future__ import annotations

import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bearing.data import CLASSES, LOADS  # noqa: E402

BASE = "https://raw.githubusercontent.com/XiongMeijing/CWRU-1/master/Data/"


def targets():
    for c in CLASSES:
        for l in LOADS:
            rel = f"Normal/Normal_{l}.mat" if c == "Normal" else f"12k_DE/{c}_{l}.mat"
            yield rel


def main(dest: Path = ROOT / "data" / "cwru"):
    todo = [r for r in targets() if not (dest / r).exists()]
    print(f"{40 - len(todo)} files present, downloading {len(todo)} to {dest}")
    for i, rel in enumerate(todo, 1):
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        url = BASE + urllib.parse.quote(rel)
        tmp = out.with_suffix(".part")
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(out)
            print(f"[{i}/{len(todo)}] {rel}")
        except Exception as e:  # keep going, report at the end
            print(f"[{i}/{len(todo)}] FAILED {rel}: {e}")
    missing = [r for r in targets() if not (dest / r).exists()]
    if missing:
        print("Missing files:", missing)
        print("Alternative: git clone https://github.com/XiongMeijing/CWRU-1 and copy its Data folder to data/cwru")
        sys.exit(1)
    print("All 40 files ready.")


if __name__ == "__main__":
    main()
