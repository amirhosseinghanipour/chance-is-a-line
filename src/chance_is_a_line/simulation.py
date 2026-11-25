"""Judge-Advisor System (JAS) simulator and reliance metrics.

Computes reliance metrics under null and discriminating decision policies as a
function of AI accuracy (a_A), pre-advice human accuracy (a_H), task
cardinality (k), and response bias.

Tracked over-reliance denominator conventions:
    - "dis"            : switches in n1 cell / n_disagreement (confounded with a_A)
    - "all"            : switches in n1 cell / n_trials       (confounded with a_A)
    - "ai_wrong"       : switches in n1 cell / n_(AI wrong)   (= p_switch * a_H; invariant)
    - "ai_wrong_accept": Pr(accept | AI wrong)                (= 1 - RSR under null; invariant)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import DEFAULT_N_SIM, DEFAULT_SEED, REF_A_H

PolicyFn = Callable[[bool, bool], float]


def always_accept(h: bool, a: bool) -> float:
    return 1.0


def always_retain(h: bool, a: bool) -> float:
    return 0.0


def constant(p: float) -> PolicyFn:
    """Return a non-discriminating policy that switches with constant probability p."""

    def _policy(h: bool, a: bool) -> float:
        return p

    _policy.__name__ = f"constant_p{p:.2f}"
    return _policy


def oracle(h: bool, a: bool) -> float:
    return 1.0 if a else 0.0


def anti_oracle(h: bool, a: bool) -> float:
    return 0.0 if a else 1.0


def discriminating(sens: float, spec: float) -> PolicyFn:
    """Switch with probability `sens` when AI is right and `1 - spec` when AI is wrong."""

    def _policy(h: bool, a: bool) -> float:
        return sens if a else (1.0 - spec)

    _policy.__name__ = f"disc_se{sens:.2f}_sp{spec:.2f}"
    return _policy


def _p_switch_vec(policy: PolicyFn, h_ok: np.ndarray, a_ok: np.ndarray) -> np.ndarray:
    lut = np.empty((2, 2), dtype=float)
    for hi in (0, 1):
        for ai in (0, 1):
            lut[hi, ai] = float(policy(bool(hi), bool(ai)))
    return lut[h_ok.astype(np.int8), a_ok.astype(np.int8)]


def simulate(
    policy: PolicyFn,
    a_H: float = 0.60,
    a_A: float = 0.75,
    n: int = DEFAULT_N_SIM,
    k: int = 2,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Simulate a k-alternative Judge-Advisor task with an explicit RNG seed."""
    rng = np.random.default_rng(seed)
    h_ok = rng.random(n) < a_H
    a_ok = rng.random(n) < a_A
    same_wrong = rng.random(n) < (1.0 / (k - 1))
    agree = (h_ok & a_ok) | (~h_ok & ~a_ok & same_wrong)
    disagree = ~agree

    p_switch = _p_switch_vec(policy, h_ok, a_ok)
    switched = disagree & (rng.random(n) < p_switch)
    final_ok = np.where(switched, a_ok, h_ok)
    accepted_ai = agree | switched
    return pd.DataFrame(
        {
            "h_ok": h_ok,
            "a_ok": a_ok,
            "agree": agree,
            "switched": switched,
            "accepted_ai": accepted_ai,
            "final_ok": final_ok,
        }
    )


def analytic_over_reliance(
    a_H: float,
    a_A: float,
    p_switch: float = 1.0,
    k: int = 2,
    denom: str = "dis",
) -> float:
    """Exact (noise-free) over-reliance under a non-discriminating constant-p policy."""
    n1 = a_H * (1.0 - a_A)
    n0 = (1.0 - a_H) * a_A
    both_wrong_dis = (1.0 - a_H) * (1.0 - a_A) * ((k - 2) / (k - 1)) if k > 2 else 0.0
    n_dis = n1 + n0 + both_wrong_dis
    numer = p_switch * n1
    if denom == "dis":
        return numer / n_dis if n_dis else float("nan")
    if denom == "all":
        return numer
    if denom == "ai_wrong":
        return numer / (1.0 - a_A) if a_A < 1.0 else float("nan")
    if denom == "ai_wrong_accept":
        return float(p_switch) if a_A < 1.0 else float("nan")
    raise ValueError(f"unknown denom {denom!r}")


def metrics(df: pd.DataFrame) -> dict[str, float]:
    """Compute reliance metrics used across the human-AI decision-making literature."""
    dis = df[~df.agree]
    n_dis = len(dis)
    if n_dis == 0:
        return {}

    pos_ai = int(((~dis.h_ok) & dis.a_ok & dis.final_ok).sum())
    neg_self = int(((~dis.h_ok) & dis.a_ok & ~dis.final_ok).sum())
    pos_self = int((dis.h_ok & (~dis.a_ok) & dis.final_ok).sum())
    neg_ai = int((dis.h_ok & (~dis.a_ok) & ~dis.final_ok).sum())

    rair = pos_ai / (pos_ai + neg_self) if (pos_ai + neg_self) else float("nan")
    rsr = pos_self / (pos_self + neg_ai) if (pos_self + neg_ai) else float("nan")

    over = neg_ai / n_dis
    under = neg_self / n_dis

    n_ai_wrong = int((~df.a_ok).sum())
    n_accept_ai_wrong = (
        int(((~df.a_ok) & df.accepted_ai).sum())
        if "accepted_ai" in df.columns
        else float("nan")
    )

    return {
        "accuracy": float(df.final_ok.mean()),
        "acc_initial": float(df.h_ok.mean()),
        "acc_wid": float(dis.final_ok.mean()),
        "agreement": float(df.agree.mean()),
        "switch": float(df.switched.mean()),
        "switch_wid": float(dis.switched.mean()),
        "over_reliance": float(over),
        "under_reliance": float(under),
        "over_reliance_all": float(neg_ai / len(df)),
        "over_reliance_aiw": float(neg_ai / n_ai_wrong) if n_ai_wrong else float("nan"),
        "over_reliance_aiw_accept": (
            float(n_accept_ai_wrong / n_ai_wrong) if n_ai_wrong else float("nan")
        ),
        "RAIR": float(rair),
        "RSR": float(rsr),
        "J": float(rair + rsr - 1.0),
    }


def run(policy: PolicyFn, **kw: Any) -> dict[str, float]:
    """Convenience helper: simulate a policy and compute all reliance metrics."""
    return metrics(simulate(policy, **kw))


def verify_analytic_claims(
    out_path: Path | None = None,
    a_H: float = REF_A_H,
    n_sim: int = DEFAULT_N_SIM,
    seed: int = DEFAULT_SEED,
    verbose: bool = True,
) -> dict[str, Any]:
    """Reproduce all four Section 2 analytic claims (exact + Monte Carlo)."""
    out: dict[str, Any] = {"a_H": a_H, "seed": seed, "n_sim": n_sim}

    # Claim 1: null diagonal RAIR + RSR = 1
    null_rows = []
    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        for a in (0.5, 0.65, 0.8, 0.95):
            m = run(constant(p), a_H=a_H, a_A=a, n=n_sim, seed=seed)
            null_rows.append({"p": p, "a_A": a, "RAIR": m["RAIR"], "RSR": m["RSR"], "sum": m["RAIR"] + m["RSR"]})
    null_df = pd.DataFrame(null_rows)
    c1 = float(null_df["sum"].sub(1.0).abs().max())
    out["claim1_max_abs_dev_from_null_line"] = round(c1, 4)

    # Claim 2: over-reliance confound across denominators
    grid = np.arange(0.50, 0.96, 0.05)
    exact_dis = [analytic_over_reliance(a_H, float(a), p_switch=1.0, denom="dis") for a in grid]
    exact_all = [analytic_over_reliance(a_H, float(a), p_switch=1.0, denom="all") for a in grid]
    exact_aiw_sw = [analytic_over_reliance(a_H, float(a), p_switch=1.0, denom="ai_wrong") for a in grid]
    exact_aiw_acc = [analytic_over_reliance(a_H, float(a), p_switch=1.0, denom="ai_wrong_accept") for a in grid]
    sim_dis = [run(always_accept, a_H=a_H, a_A=float(round(a, 2)), n=n_sim, seed=seed)["over_reliance"] for a in grid]

    out["claim2_exact_dis_range"] = [round(max(exact_dis), 3), round(min(exact_dis), 3)]
    out["claim2_sim_dis_range"] = [round(max(sim_dis), 3), round(min(sim_dis), 3)]
    out["claim2_exact_all_range"] = [round(max(exact_all), 3), round(min(exact_all), 3)]
    out["claim2_ai_wrong_switches_range"] = [round(min(exact_aiw_sw), 3), round(max(exact_aiw_sw), 3)]
    out["claim2_ai_wrong_accept_range"] = [round(min(exact_aiw_acc), 3), round(max(exact_aiw_acc), 3)]

    # Claim 3: non-identifiability of over = 0.14
    target = 0.14
    c3_rows = []
    aa = np.linspace(0.50, 0.99, 5000)
    for name, p_sw in [("always accept", 1.0), ("coin flip", 0.5), ("mostly ignore", 0.25)]:
        vals = np.array([analytic_over_reliance(a_H, float(a), p_switch=p_sw) for a in aa])
        if vals.min() <= target <= vals.max():
            a_star = float(np.interp(target, vals[::-1], aa[::-1]))
            m = run(constant(p_sw), a_H=a_H, a_A=round(a_star, 3), n=n_sim, seed=seed)
            c3_rows.append(
                {
                    "policy": name,
                    "a_A": round(a_star, 3),
                    "over": round(target, 3),
                    "acc": round(m["accuracy"], 3),
                }
            )
    out["claim3"] = c3_rows

    # Claim 4: criterion shifts at J = 0 and Gaussian SDT d' = 1.5
    m_a = run(constant(0.60), a_H=a_H, a_A=0.75, n=300_000, seed=seed)
    m_b = run(constant(0.30), a_H=a_H, a_A=0.75, n=300_000, seed=seed)
    out["claim4"] = {
        "RAIR": [round(m_a["RAIR"], 3), round(m_b["RAIR"], 3)],
        "RSR": [round(m_a["RSR"], 3), round(m_b["RSR"], 3)],
        "J": [round(m_a["J"], 3), round(m_b["J"], 3)],
    }

    dprime = 1.5
    sdt_tab = []
    for c in (-0.5, 0.0, 0.25, 0.75, 1.25, 1.5, 2.0):
        hit = float(norm.cdf(dprime - c))
        fa = float(norm.cdf(-c))
        sdt_tab.append(
            {
                "criterion": c,
                "RAIR": round(hit, 3),
                "RSR": round(1.0 - fa, 3),
                "J": round(hit - fa, 3),
                "d_prime": dprime,
            }
        )
    sdt_df = pd.DataFrame(sdt_tab)
    out["claim4b_J_varies_with_criterion_at_fixed_dprime"] = sdt_tab
    out["claim4b_J_range_at_fixed_dprime"] = [round(float(sdt_df.J.min()), 3), round(float(sdt_df.J.max()), 3)]
    out["claim4b_J_max_at_c"] = round(dprime / 2.0, 3)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    if verbose:
        print(f"CLAIM 1  null line: exact RAIR+RSR = 1; max simulated deviation = {c1:.4f}")
        print(
            f"CLAIM 2  fixed accept-all (a_H={a_H}), a_A in [0.50, 0.95]: "
            f"dis={out['claim2_exact_dis_range'][0]:.3f}->{out['claim2_exact_dis_range'][1]:.3f}, "
            f"all={out['claim2_exact_all_range'][0]:.3f}->{out['claim2_exact_all_range'][1]:.3f}, "
            f"ai_wrong={out['claim2_ai_wrong_switches_range'][0]:.3f}, "
            f"1-RSR={out['claim2_ai_wrong_accept_range'][0]:.3f}"
        )
        for r in c3_rows:
            print(f"CLAIM 3  over={target:.2f} | {r['policy']:14s} a_A={r['a_A']:.2f} acc={r['acc']:.3f}")
        print(
            f"CLAIM 4  J=0 shift: RAIR {m_a['RAIR']:.3f}->{m_b['RAIR']:.3f}, "
            f"RSR {m_a['RSR']:.3f}->{m_b['RSR']:.3f}, J {m_a['J']:+.3f}->{m_b['J']:+.3f} | "
            f"SDT(d'=1.5): J in [{out['claim4b_J_range_at_fixed_dprime'][0]:.3f}, "
            f"{out['claim4b_J_range_at_fixed_dprime'][1]:.3f}]"
        )
    return out
