#!/usr/bin/env python3
from __future__ import annotations

import base64
import io
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BODY = ROOT / "notebooks/overnight_a100_body.py"
OUT = ROOT / "notebooks/overnight_a100_fresh_cell.py"
INCLUDE = [
    "pyproject.toml",
    "README.md",
    "requirements-colab.txt",
    "src",
]


def build_tarball() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel in INCLUDE:
            path = ROOT / rel
            tf.add(path, arcname=f"frozen_prior_amt/{rel}")
    return buf.getvalue()


def main() -> None:
    blob = base64.b64encode(build_tarball()).decode("ascii")
    body = BODY.read_text(encoding="utf-8")
    OUT.write_text(
        f'''# Paste this whole cell into a fresh A100 Colab notebook.
# It creates /content/frozen_prior_amt from the local source snapshot, installs
# dependencies, then runs the fixed overnight seeded-refinement experiment.

import base64 as _fpamt_base64
import io as _fpamt_io
import os as _fpamt_os
import shutil as _fpamt_shutil
import subprocess as _fpamt_subprocess
import sys as _fpamt_sys
import tarfile as _fpamt_tarfile
from pathlib import Path as _FpamtPath

_FPAMT_BLOB = """{blob}"""
_fpamt_root = _FpamtPath("/content/frozen_prior_amt")
if _fpamt_root.exists():
    _fpamt_shutil.rmtree(_fpamt_root)
with _fpamt_tarfile.open(fileobj=_fpamt_io.BytesIO(_fpamt_base64.b64decode(_FPAMT_BLOB)), mode="r:gz") as _tf:
    _tf.extractall("/content", filter="data")
_fpamt_os.chdir(_fpamt_root)
_fpamt_subprocess.check_call([_fpamt_sys.executable, "-m", "pip", "install", "-q", "-r", str(_fpamt_root / "requirements-colab.txt")])
_fpamt_subprocess.check_call([_fpamt_sys.executable, "-m", "pip", "install", "-q", "-e", str(_fpamt_root)])
print("Fresh project ready at", _fpamt_root)

{body}
''',
        encoding="utf-8",
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
