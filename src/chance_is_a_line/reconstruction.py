"""Exact multiclass reconstruction and bounds for Youden's J = RAIR + RSR - 1.

Preserves the distinction between binary (k=2) and multiclass (k>2) tasks where
two incorrect parties disagree with probability (k-2)/(k-1), entering n_dis
without belonging to either reliance cell (n0 or n1).
"""

from __future__ import annotations

import numpy as np

TIERS: dict[int, str] = {
    1: "per-condition RAIR and RSR reported          -> J exact, dJ exact",
    2: "aggregate RAIR and RSR only                  -> J exact, dJ unavailable",
    3: "over- and under-reliance + accuracies        -> J reconstructed",
    4: "one of over/under + accuracies               -> J bounded",
    5: "agreement/switch fraction only               -> bias only, J unidentified",
}


def split_from_accuracies(a_H: float, a_A: float, k: int = 2) -> tuple[float, float, float]:
    """Expected disagreement split (n1, n0, n_dis) as fractions of all trials."""
    n1 = a_H * (1.0 - a_A)
    n0 = (1.0 - a_H) * a_A
    both_wrong_disagree = (1.0 - a_H) * (1.0 - a_A) * ((k - 2) / (k - 1)) if k > 2 else 0.0
    n_dis = n1 + n0 + both_wrong_disagree
    return n1, n0, n_dis


def j_from_rair_rsr(rair: float, rsr: float) -> float:
    return float(rair) + float(rsr) - 1.0


def rair_rsr_from_reliance(
    over: float, under: float, a_H: float, a_A: float, k: int = 2
) -> tuple[float, float]:
    """Invert disagreement-normalized over-/under-reliance to recover (RAIR, RSR)."""
    n1, n0, n_dis = split_from_accuracies(a_H, a_A, k)
    if n1 <= 0 or n0 <= 0:
        return (float("nan"), float("nan"))
    rsr = 1.0 - over * n_dis / n1
    rair = 1.0 - under * n_dis / n0
    return rair, rsr


def j_from_reliance(over: float, under: float, a_H: float, a_A: float, k: int = 2) -> float:
    """Tier 3: reconstruct Youden's J from over-/under-reliance and baseline accuracies."""
    rair, rsr = rair_rsr_from_reliance(over, under, a_H, a_A, k)
    if np.isnan(rair) or np.isnan(rsr):
        return float("nan")
    return rair + rsr - 1.0


def j_bounds(
    over: float | None = None,
    under: float | None = None,
    a_H: float | None = None,
    a_A: float | None = None,
    k: int = 2,
) -> tuple[float, float]:
    """Tier 4: exact feasibility bounds on J when only one of over/under is reported."""
    if a_H is None or a_A is None:
        return (float("nan"), float("nan"))
    n1, n0, n_dis = split_from_accuracies(a_H, a_A, k)
    if n1 <= 0 or n0 <= 0:
        return (float("nan"), float("nan"))
    if over is None and under is None:
        return (-1.0, 1.0)
    if over is None:
        rair = 1.0 - under * n_dis / n0
        return (rair - 1.0, rair)
    if under is None:
        rsr = 1.0 - over * n_dis / n1
        return (rsr - 1.0, rsr)
    val = j_from_reliance(over, under, a_H, a_A, k)
    return (val, val)


def null_over_reliance(a_H: float, a_A: float, p_switch: float, k: int = 2) -> float:
    n1, _, n_dis = split_from_accuracies(a_H, a_A, k)
    return p_switch * n1 / n_dis if n_dis else float("nan")
