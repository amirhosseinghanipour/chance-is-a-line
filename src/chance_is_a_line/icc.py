"""Method-of-moments ICC estimator and beta-binomial participant simulator."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import DEFAULT_ICC, MIN_M_EFF


def icc_from_sd(p: float, sd: float, m_eff: float) -> tuple[float, float]:
    """Method-of-moments ICC from participant mean and SD.

    Returns (icc_clipped, icc_raw).
    """
    for v in (p, sd, m_eff):
        if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
            return float("nan"), float("nan")
    if not (0.0 < p < 1.0) or sd < 0.0 or m_eff <= 1.0:
        return float("nan"), float("nan")
    binom_var = p * (1.0 - p)
    if binom_var <= 0.0:
        return float("nan"), float("nan")
    raw = ((sd**2) / binom_var - 1.0 / m_eff) / (1.0 - 1.0 / m_eff)
    return float(np.clip(raw, 0.0, 1.0)), float(raw)


def effective_m(row: pd.Series | dict[str, Any], cell: str) -> float:
    m = row.get("n_trials_per_participant")
    a_h = row.get("a_H_initial")
    a_a = row.get("a_A_actual")
    if a_a is None or (isinstance(a_a, (float, np.floating)) and np.isnan(a_a)):
        a_a = row.get("a_A_stated")
    for v in (m, a_h, a_a):
        if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
            return float("nan")
    share = (1.0 - float(a_h)) * float(a_a) if cell == "rair" else float(a_h) * (1.0 - float(a_a))
    return float(m) * share


def icc_for_row(row: pd.Series, min_m_eff: int = MIN_M_EFF) -> float:
    ests = []
    for mean_col, sd_col, cell in [
        ("rair", "rair_sd", "rair"),
        ("rsr", "rsr_sd", "rsr"),
        ("over_reliance", "over_sd", "rsr"),
        ("under_reliance", "under_sd", "rair"),
    ]:
        p, sd = row.get(mean_col), row.get(sd_col)
        if p is None or sd is None:
            continue
        if (isinstance(p, (float, np.floating)) and np.isnan(p)) or (
            isinstance(sd, (float, np.floating)) and np.isnan(sd)
        ):
            continue
        m_eff = effective_m(row, cell)
        if np.isnan(m_eff) or m_eff < min_m_eff:
            continue
        icc, _ = icc_from_sd(float(p), float(sd), m_eff)
        if not np.isnan(icc):
            ests.append(icc)
    return float(np.mean(ests)) if ests else float("nan")


def icc_diagnosis(df: pd.DataFrame, min_m_eff: int = MIN_M_EFF) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        reasons, best = [], float("nan")
        for mean_col, sd_col, cell in [
            ("rair", "rair_sd", "rair"),
            ("rsr", "rsr_sd", "rsr"),
            ("over_reliance", "over_sd", "rsr"),
            ("under_reliance", "under_sd", "rair"),
        ]:
            p_, sd = row.get(mean_col), row.get(sd_col)
            if (
                p_ is None
                or sd is None
                or (isinstance(p_, (float, np.floating)) and np.isnan(p_))
                or (isinstance(sd, (float, np.floating)) and np.isnan(sd))
            ):
                reasons.append(f"{mean_col}: no mean+SD pair")
                continue
            m_eff = effective_m(row, cell)
            if np.isnan(m_eff):
                reasons.append(f"{mean_col}: a_H missing -> no cell size")
                continue
            if m_eff < min_m_eff:
                reasons.append(f"{mean_col}: m_eff={m_eff:.1f} < {min_m_eff}")
                continue
            est, _ = icc_from_sd(float(p_), float(sd), m_eff)
            if not np.isnan(est):
                best = est
                reasons.append(f"{mean_col}: OK (m_eff={m_eff:.1f})")
        rows.append(
            {
                "study_id": row.get("study_id"),
                "paper_id": row.get("paper_id"),
                "condition_label": row.get("condition_label"),
                "icc_est": best,
                "blocked_by": "; ".join(reasons),
            }
        )
    return pd.DataFrame(rows)


def pool_icc(df: pd.DataFrame, default: float = DEFAULT_ICC) -> tuple[float, int, str]:
    if "icc_est" not in df.columns:
        return default, 0, "assumed"
    ok = df[df.icc_est.notna() & df.n_participants.notna()]
    if not len(ok):
        return default, 0, "assumed"
    w = ok.n_participants.astype(float)
    return float((ok.icc_est * w).sum() / w.sum()), len(ok), "estimated"


def attach_icc(df: pd.DataFrame, default: float = DEFAULT_ICC) -> tuple[pd.DataFrame, float, int, str]:
    df = df.copy()
    df["icc_est"] = df.apply(icc_for_row, axis=1)
    pooled, n_ok, source = pool_icc(df, default)
    df["icc_pooled"] = pooled
    df["icc_used"] = df.icc_est.fillna(pooled)
    df["icc_source"] = np.where(
        df.icc_est.notna(),
        "estimated",
        "pooled" if source == "estimated" else "assumed",
    )
    return df, pooled, n_ok, source


def simulate_participants(
    n_participants: int,
    m_trials: int,
    sens: float,
    spec: float,
    icc: float,
    a_H: float = 0.60,
    a_A: float = 0.75,
    k: int = 2,
    seed: int = 0,
) -> pd.DataFrame:
    """Vectorized trial-level beta-binomial simulation with participant heterogeneity."""
    rng = np.random.default_rng(seed)

    def _beta_params(mean: float, rho: float) -> tuple[float, float] | None:
        if rho <= 0:
            return None
        s = 1.0 / rho - 1.0
        return mean * s, (1.0 - mean) * s

    bp_hit = _beta_params(sens, icc)
    bp_fa = _beta_params(1.0 - spec, icc)
    p_hit = rng.beta(*bp_hit, n_participants) if bp_hit else np.full(n_participants, sens)
    p_fa = rng.beta(*bp_fa, n_participants) if bp_fa else np.full(n_participants, 1.0 - spec)

    shape = (n_participants, m_trials)
    h_ok = rng.random(shape) < a_H
    a_ok = rng.random(shape) < a_A
    same_wrong = rng.random(shape) < (1.0 / (k - 1))
    agree = (h_ok & a_ok) | (~h_ok & ~a_ok & same_wrong)
    disagree = ~agree
    p_sw = np.where(a_ok, p_hit[:, None], p_fa[:, None])
    switched = disagree & (rng.random(shape) < p_sw)

    cell0 = disagree & (~h_ok) & a_ok
    cell1 = disagree & h_ok & (~a_ok)

    n0 = cell0.sum(axis=1)
    n1 = cell1.sum(axis=1)
    n_dis = disagree.sum(axis=1)
    n_hit = (cell0 & switched).sum(axis=1)
    n_cr = (cell1 & (~switched)).sum(axis=1)
    n_sw_n1 = (cell1 & switched).sum(axis=1)
    n_keep_n0 = (cell0 & (~switched)).sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        rair = np.where(n0 > 0, n_hit / n0, np.nan)
        rsr = np.where(n1 > 0, n_cr / n1, np.nan)
        over = np.where(n_dis > 0, n_sw_n1 / n_dis, np.nan)
        under = np.where(n_dis > 0, n_keep_n0 / n_dis, np.nan)

    return pd.DataFrame(
        {
            "pid": np.arange(n_participants),
            "a_H": h_ok.mean(axis=1),
            "a_A": a_ok.mean(axis=1),
            "rair": rair,
            "rsr": rsr,
            "over": over,
            "under": under,
            "n0": n0,
            "n1": n1,
            "n_dis": n_dis,
            "n_hit": n_hit,
            "n_cr": n_cr,
            "n_sw_n1": n_sw_n1,
            "n_keep_n0": n_keep_n0,
        }
    )
