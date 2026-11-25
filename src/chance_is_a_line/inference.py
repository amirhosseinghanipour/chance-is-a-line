"""Cell-specific cluster-adjusted confidence intervals and equivalence testing.

Under beta-binomial clustering inside reliance cell c in {0, 1}, each
participant contributes an expected m_c trials to cell c:
    m_0 = m * (1 - a_H) * a_A     (RAIR cell)
    m_1 = m * a_H * (1 - a_A)     (RSR cell)
The exact cell-level design effect is DEFF_c = max(1, 1 + (m_c - 1) * rho).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from .config import DEFAULT_CONF, DEFAULT_ICC, DEFAULT_MARGIN


def _is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, (float, np.floating)) and np.isnan(v))


def cell_trials_per_participant(a_H: float, a_A: float, m: float) -> tuple[float, float] | None:
    """Expected effective trials per participant in (RSR cell m1, RAIR cell m0)."""
    if any(_is_missing(v) for v in (a_H, a_A, m)):
        return None
    m1 = float(m) * float(a_H) * (1.0 - float(a_A))
    m0 = float(m) * (1.0 - float(a_H)) * float(a_A)
    return max(m1, 0.0), max(m0, 0.0)


def cell_counts(
    a_H: float,
    a_A: float,
    n_participants: float,
    n_trials_per_participant: float,
    k: int = 2,
) -> tuple[float, float] | None:
    """Total expected observations across all participants in (RSR n1, RAIR n0)."""
    if any(_is_missing(v) for v in (a_H, a_A, n_participants, n_trials_per_participant)):
        return None
    total = float(n_participants) * float(n_trials_per_participant)
    n1 = total * float(a_H) * (1.0 - float(a_A))
    n0 = total * (1.0 - float(a_H)) * float(a_A)
    return max(n1, 0.0), max(n0, 0.0)


def deff(m_cluster: float | None, icc: float = DEFAULT_ICC) -> float:
    """Design effect 1 + (m_cluster - 1) * rho, bounded below by 1.0."""
    if _is_missing(m_cluster):
        return 1.0
    return max(1.0, 1.0 + (float(m_cluster) - 1.0) * float(icc))


def var_proportion(
    p: float, n_total_cell: float, m_cluster: float | None = None, icc: float = DEFAULT_ICC
) -> float:
    if _is_missing(n_total_cell) or n_total_cell <= 0 or _is_missing(p):
        return float("nan")
    return float(p) * (1.0 - float(p)) / float(n_total_cell) * deff(m_cluster, icc)


def _arm_variances(
    rair: float,
    rsr: float,
    a_H: float,
    a_A: float,
    n_participants: float,
    m: float,
    k: int = 2,
    icc: float = DEFAULT_ICC,
    deff_mode: str = "cell",
) -> dict[str, float] | None:
    cc = cell_counts(a_H, a_A, n_participants, m, k)
    cm = cell_trials_per_participant(a_H, a_A, m)
    if cc is None or cm is None:
        return None
    n1, n0 = cc
    m1, m0 = cm
    if deff_mode == "cell":
        mc0, mc1 = m0, m1
    elif deff_mode == "total":
        mc0, mc1 = m, m
    else:
        raise ValueError(f"unknown deff_mode {deff_mode!r}")
    return {
        "v_rair": var_proportion(rair, n0, mc0, icc),
        "v_rsr": var_proportion(rsr, n1, mc1, icc),
        "n1": n1,
        "n0": n0,
        "m1": m1,
        "m0": m0,
        "deff_rair": deff(mc0, icc),
        "deff_rsr": deff(mc1, icc),
    }


def j_interval(
    rair: float,
    rsr: float,
    a_H: float,
    a_A: float,
    n_participants: float,
    m: float,
    k: int = 2,
    icc: float = DEFAULT_ICC,
    conf: float = DEFAULT_CONF,
    deff_mode: str = "cell",
) -> tuple[float, float, float]:
    """Return (J, J_lower, J_upper) cluster-adjusted confidence interval."""
    av = _arm_variances(rair, rsr, a_H, a_A, n_participants, m, k=k, icc=icc, deff_mode=deff_mode)
    if av is None:
        return (float("nan"), float("nan"), float("nan"))
    v = av["v_rair"] + av["v_rsr"]
    if np.isnan(v):
        return (float("nan"), float("nan"), float("nan"))
    se = float(np.sqrt(v))
    z = float(stats.norm.ppf(0.5 + conf / 2.0))
    j = float(rair) + float(rsr) - 1.0
    return j, j - z * se, j + z * se


def contrast_interval(
    rair_a: float,
    rsr_a: float,
    rair_b: float,
    rsr_b: float,
    a_H: float,
    a_A: float,
    n_participants: float,
    m: float,
    k: int = 2,
    icc: float = DEFAULT_ICC,
    conf: float = DEFAULT_CONF,
    paired: bool = False,
    rho_within: float = 0.0,
    deff_mode: str = "cell",
    a_H_b: float | None = None,
    a_A_b: float | None = None,
    n_participants_b: float | None = None,
    m_b: float | None = None,
) -> dict[str, Any] | None:
    """Cluster-adjusted CI for dJ = J_b - J_a and dBias = Bias_b - Bias_a."""
    a_H_2 = a_H if _is_missing(a_H_b) else float(a_H_b)  # type: ignore[arg-type]
    a_A_2 = a_A if _is_missing(a_A_b) else float(a_A_b)  # type: ignore[arg-type]
    n_2 = n_participants if _is_missing(n_participants_b) else float(n_participants_b)  # type: ignore[arg-type]
    m_2 = m if _is_missing(m_b) else float(m_b)  # type: ignore[arg-type]

    av_a = _arm_variances(rair_a, rsr_a, a_H, a_A, n_participants, m, k=k, icc=icc, deff_mode=deff_mode)
    av_b = _arm_variances(rair_b, rsr_b, a_H_2, a_A_2, n_2, m_2, k=k, icc=icc, deff_mode=deff_mode)
    if av_a is None or av_b is None:
        return None

    z = float(stats.norm.ppf(0.5 + conf / 2.0))
    va_r, vb_r = av_a["v_rair"], av_b["v_rair"]
    va_s, vb_s = av_a["v_rsr"], av_b["v_rsr"]

    if paired and rho_within:
        v_rair = va_r + vb_r - 2.0 * rho_within * float(np.sqrt(va_r * vb_r))
        v_rsr = va_s + vb_s - 2.0 * rho_within * float(np.sqrt(va_s * vb_s))
    else:
        v_rair, v_rsr = va_r + vb_r, va_s + vb_s

    dj = (rair_b + rsr_b - 1.0) - (rair_a + rsr_a - 1.0)
    se_dj = float(np.sqrt(v_rair + v_rsr))

    db = 0.5 * ((rair_b + 1.0 - rsr_b) - (rair_a + 1.0 - rsr_a))
    se_db = 0.5 * se_dj

    return {
        "dJ": dj,
        "dJ_lo": dj - z * se_dj,
        "dJ_hi": dj + z * se_dj,
        "dBias": db,
        "dBias_lo": db - z * se_db,
        "dBias_hi": db + z * se_db,
        "n1_a": av_a["n1"],
        "n0_a": av_a["n0"],
        "n1_b": av_b["n1"],
        "n0_b": av_b["n0"],
        "m1_a": av_a["m1"],
        "m0_a": av_a["m0"],
        "m1_b": av_b["m1"],
        "m0_b": av_b["m0"],
        "deff_mode": deff_mode,
        "paired_flag": paired,
        "rho_within": rho_within,
        "icc_used": icc,
    }


def classify(ci: dict[str, Any] | None, margin: float = DEFAULT_MARGIN) -> str:
    """Classify a contrast against a pre-specified equivalence margin."""
    if ci is None or np.isnan(ci["dJ_lo"]):
        return "unidentified"
    dj_equiv = (ci["dJ_lo"] >= -margin) and (ci["dJ_hi"] <= margin)
    dj_sig = (ci["dJ_lo"] > 0) or (ci["dJ_hi"] < 0)
    db_equiv = (ci["dBias_lo"] >= -margin / 2.0) and (ci["dBias_hi"] <= margin / 2.0)
    db_sig = (ci["dBias_lo"] > 0) or (ci["dBias_hi"] < 0)

    if dj_equiv and db_sig:
        return "bias"
    if dj_sig and db_equiv:
        return "skill"
    if dj_sig and db_sig:
        return "mixed"
    if dj_equiv and db_equiv:
        return "null"
    return "indeterminate"


def classify_point(dj: float, db: float, tol: float = 0.05) -> str:
    if np.isnan(dj):
        return "unidentified"
    dj_small, db_small = abs(dj) <= tol, abs(db) <= tol / 2.0
    if dj_small and not db_small:
        return "bias"
    if not dj_small and db_small:
        return "skill"
    if not dj_small and not db_small:
        return "mixed"
    return "null"
