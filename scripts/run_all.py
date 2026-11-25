#!/usr/bin/env python3
"""Standalone runner so the repository works even without `pip install -e .`."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chance_is_a_line.cli import main

if __name__ == "__main__":
    argv = sys.argv[1:] or ["reproduce"]
    main(argv)
