"""Central constants, paths, and reproducibility defaults."""

from __future__ import annotations

from pathlib import Path

# Repository root (resolved relative to src/chance_is_a_line/config.py)
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Statistical defaults
DEFAULT_ICC: float = 0.10
DEFAULT_MARGIN: float = 0.10
DEFAULT_CONF: float = 0.95
MIN_M_EFF: int = 8

# Reference simulation parameters (Section 2 & Appendix A)
REF_A_H: float = 0.60
REF_A_A: float = 0.75
DEFAULT_N_SIM: int = 200_000
DEFAULT_SEED: int = 0
