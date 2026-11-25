"""Chance Is a Line: Reproducibility package for human-AI reliance auditing."""

__version__ = "1.0.0"

from .config import DEFAULT_ICC, DEFAULT_MARGIN, MIN_M_EFF
from .icc import icc_from_sd, simulate_participants
from .inference import classify, contrast_interval, j_interval
from .reconstruction import j_bounds, j_from_rair_rsr, j_from_reliance, rair_rsr_from_reliance
from .simulation import (
    always_accept,
    always_retain,
    analytic_over_reliance,
    constant,
    discriminating,
    metrics,
    simulate,
)

__all__ = [
    "DEFAULT_ICC",
    "DEFAULT_MARGIN",
    "MIN_M_EFF",
    "always_accept",
    "always_retain",
    "analytic_over_reliance",
    "classify",
    "constant",
    "contrast_interval",
    "discriminating",
    "icc_from_sd",
    "j_bounds",
    "j_from_rair_rsr",
    "j_from_reliance",
    "j_interval",
    "metrics",
    "rair_rsr_from_reliance",
    "simulate",
    "simulate_participants",
]
