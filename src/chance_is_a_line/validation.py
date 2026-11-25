"""Monte Carlo validation (Eqs. 3, 5, CI coverage) & open trial-log auditing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .config import EXTERNAL_DATA_DIR, MIN_M_EFF
from .icc import icc_from_sd, simulate_participants
from .inference import j_interval
from .reconstruction import j_from_reliance, rair_rsr_from_reliance
from .simulation import discriminating, run


def validate_reconstruction(n_sim: int = 200_000, seed: int = 0) -> pd.DataFrame:
    """Validate Eq. (3) across 81 factorial cells spanning k in {2, 3, 4}."""
    rows = []
    idx = 0
    for k in (2, 3, 4):
        for a_h in (0.45, 0.60, 0.75):
            for a_a in (0.50, 0.67, 0.80):
                for sens, spec in [(0.70, 0.65), (0.35, 0.70), (0.25, 0.25)]:
                    m = run(discriminating(sens, spec), a_H=a_h, a_A=a_a, n=n_sim, k=k, seed=seed + idx)
                    idx += 1
                    j_true = m["J"]
                    rair_rec, rsr_rec = rair_rsr_from_reliance(
                        m["over_reliance"], m["under_reliance"], m["acc_initial"], a_a, k=k
                    )
                    j_rec = rair_rec + rsr_rec - 1.0
                    j_naive_k2 = j_from_reliance(
                        m["over_reliance"], m["under_reliance"], m["acc_initial"], a_a, k=2
                    )
                    rows.append(
                        {
                            "k": k, "a_H": a_h, "a_A": a_a, "sens": sens, "spec": spec,
                            "J_true": j_true, "J_rec": j_rec,
                            "abs_err_J": abs(j_rec - j_true),
                            "abs_err_RAIR": abs(rair_rec - m["RAIR"]),
                            "abs_err_RSR": abs(rsr_rec - m["RSR"]),
                            "abs_err_naive_k2": abs(j_naive_k2 - j_true),
                        }
                    )
    return pd.DataFrame(rows)


def validate_icc_grid(
    rhos: tuple[float, ...] = (0.00, 0.05, 0.10, 0.20, 0.30),
    m_eff_targets: tuple[int, ...] = (3, 4, 6, 8, 12, 20, 30),
    n_participants: int = 400,
    n_reps: int = 60,
    seed: int = 42,
) -> pd.DataFrame:
    """Validate Eq. (5) method-of-moments ICC across true rho and m_eff."""
    a_h, a_a = 0.60, 0.75
    cell_share_rair = (1.0 - a_h) * a_a
    rows = []
    for rho in rhos:
        for m_target in m_eff_targets:
            m_trials = max(int(round(m_target / cell_share_rair)), 4)
            ests_clip, ests_raw, m_effs = [], [], []
            for rep in range(n_reps):
                s = seed + int(rho * 1000) * 100 + m_target * 1000 + rep
                d = simulate_participants(
                    n_participants=n_participants, m_trials=m_trials,
                    sens=0.60, spec=0.60, icc=rho, a_H=a_h, a_A=a_a, k=2, seed=s,
                )
                v = d["rair"].dropna()
                m_obs = float(d.loc[v.index, "n0"].mean())
                est_c, est_r = icc_from_sd(float(v.mean()), float(v.std(ddof=1)), m_obs)
                if not np.isnan(est_c):
                    ests_clip.append(est_c)
                    ests_raw.append(est_r)
                    m_effs.append(m_obs)
            arr_c, arr_r = np.array(ests_clip), np.array(ests_raw)
            rows.append(
                {
                    "rho_true": rho,
                    "m_eff_target": m_target,
                    "m_eff_actual": float(np.mean(m_effs)),
                    "m_trials_total": m_trials,
                    "mean_est_clipped": float(np.mean(arr_c)),
                    "bias_clipped": float(np.mean(arr_c) - rho),
                    "rmse_clipped": float(np.sqrt(np.mean((arr_c - rho) ** 2))),
                    "mean_est_raw": float(np.mean(arr_r)),
                    "bias_raw": float(np.mean(arr_r) - rho),
                    "pct_truncated_zero": float(np.mean(arr_r < 0.0) * 100.0),
                    "clears_threshold": m_target >= MIN_M_EFF,
                }
            )
    return pd.DataFrame(rows)


def validate_ci_coverage(
    rhos: tuple[float, ...] = (0.00, 0.05, 0.10, 0.20, 0.30),
    n_participants: int = 120,
    m_trials: int = 60,
    sens: float = 0.65,
    spec: float = 0.65,
    a_H: float = 0.60,
    a_A: float = 0.70,
    n_reps: int = 500,
    seed: int = 2026,
) -> pd.DataFrame:
    """Monte Carlo empirical 95% CI coverage for Youden's J."""
    j_true = sens + spec - 1.0
    rows = []
    for r_idx, rho in enumerate(rhos):
        cov_naive = cov_cell = cov_plugin = cov_total = 0
        hw_naive, hw_cell, hw_plugin, hw_total = [], [], [], []
        for rep in range(n_reps):
            d = simulate_participants(
                n_participants=n_participants, m_trials=m_trials,
                sens=sens, spec=spec, icc=rho, a_H=a_H, a_A=a_A, k=2,
                seed=seed + r_idx * 10_000 + rep,
            )
            rair_hat = float(d.n_hit.sum()) / int(d.n0.sum())
            rsr_hat = float(d.n_cr.sum()) / int(d.n1.sum())
            a_h_hat, a_a_hat = float(d.a_H.mean()), float(d.a_A.mean())

            v0, v1 = d["rair"].dropna(), d["rsr"].dropna()
            r0, _ = icc_from_sd(float(v0.mean()), float(v0.std(ddof=1)), float(d.loc[v0.index, "n0"].mean()))
            r1, _ = icc_from_sd(float(v1.mean()), float(v1.std(ddof=1)), float(d.loc[v1.index, "n1"].mean()))
            ests = [x for x in (r0, r1) if not np.isnan(x)]
            rho_hat = float(np.mean(ests)) if ests else 0.0

            _, lo_n, hi_n = j_interval(rair_hat, rsr_hat, a_h_hat, a_a_hat, n_participants, m_trials, icc=0.0, deff_mode="cell")
            _, lo_c, hi_c = j_interval(rair_hat, rsr_hat, a_h_hat, a_a_hat, n_participants, m_trials, icc=rho, deff_mode="cell")
            _, lo_p, hi_p = j_interval(rair_hat, rsr_hat, a_h_hat, a_a_hat, n_participants, m_trials, icc=rho_hat, deff_mode="cell")
            _, lo_t, hi_t = j_interval(rair_hat, rsr_hat, a_h_hat, a_a_hat, n_participants, m_trials, icc=rho, deff_mode="total")

            cov_naive += int(lo_n <= j_true <= hi_n)
            cov_cell += int(lo_c <= j_true <= hi_c)
            cov_plugin += int(lo_p <= j_true <= hi_p)
            cov_total += int(lo_t <= j_true <= hi_t)

            hw_naive.append((hi_n - lo_n) / 2.0)
            hw_cell.append((hi_c - lo_c) / 2.0)
            hw_plugin.append((hi_p - lo_p) / 2.0)
            hw_total.append((hi_t - lo_t) / 2.0)

        rows.append(
            {
                "rho_true": rho,
                "n_participants": n_participants,
                "m_trials": m_trials,
                "n_reps": n_reps,
                "cov_naive": round(cov_naive / n_reps, 4),
                "cov_cell_true": round(cov_cell / n_reps, 4),
                "cov_cell_plugin": round(cov_plugin / n_reps, 4),
                "cov_total_true": round(cov_total / n_reps, 4),
                "mean_hw_naive": round(float(np.mean(hw_naive)), 4),
                "mean_hw_cell_true": round(float(np.mean(hw_cell)), 4),
                "mean_hw_cell_plugin": round(float(np.mean(hw_plugin)), 4),
                "mean_hw_total_true": round(float(np.mean(hw_total)), 4),
            }
        )
    return pd.DataFrame(rows)


def audit_trial_log(trials_df: pd.DataFrame, k: int = 2) -> dict[str, Any]:
    """Audit any disaggregated Judge-Advisor trial-level DataFrame."""
    df = trials_df.copy()
    h = df["human_correct"].astype(bool)
    a = df["ai_correct"].astype(bool)
    f = df["final_correct"].astype(bool)
    dis = df["disagree"].astype(bool) if "disagree" in df.columns else (h ^ a)

    cell0 = dis & (~h) & a
    cell1 = dis & h & (~a)
    n0, n1, n_dis = int(cell0.sum()), int(cell1.sum()), int(dis.sum())
    rair_pooled = float(f[cell0].mean()) if n0 else float("nan")
    rsr_pooled = float(f[cell1].mean()) if n1 else float("nan")
    j_pooled = rair_pooled + rsr_pooled - 1.0

    over = float((~f[cell1]).sum()) / n_dis if n_dis else float("nan")
    under = float((~f[cell0]).sum()) / n_dis if n_dis else float("nan")
    a_h, a_a = float(h.mean()), float(a.mean())
    j_rec = j_from_reliance(over, under, a_h, a_a, k=k)

    part_rows = []
    for pid, g in df.groupby("participant_id"):
        gh = g["human_correct"].astype(bool)
        ga = g["ai_correct"].astype(bool)
        gf = g["final_correct"].astype(bool)
        gd = g["disagree"].astype(bool) if "disagree" in g.columns else (gh ^ ga)
        c0 = gd & (~gh) & ga
        c1 = gd & gh & (~ga)
        part_rows.append(
            {
                "pid": pid,
                "rair": float(gf[c0].mean()) if c0.sum() else np.nan,
                "rsr": float(gf[c1].mean()) if c1.sum() else np.nan,
                "n0": int(c0.sum()),
                "n1": int(c1.sum()),
            }
        )
    p_df = pd.DataFrame(part_rows)
    rair_pm = float(p_df["rair"].dropna().mean())
    rsr_pm = float(p_df["rsr"].dropna().mean())
    icc_rair, icc_rair_raw = icc_from_sd(
        rair_pm, float(p_df["rair"].dropna().std(ddof=1)), float(p_df.loc[p_df["rair"].notna(), "n0"].mean())
    )
    icc_rsr, icc_rsr_raw = icc_from_sd(
        rsr_pm, float(p_df["rsr"].dropna().std(ddof=1)), float(p_df.loc[p_df["rsr"].notna(), "n1"].mean())
    )
    return {
        "n_participants": int(df["participant_id"].nunique()),
        "n_trials": int(len(df)),
        "a_H": a_h,
        "a_A": a_a,
        "RAIR_pooled": rair_pooled,
        "RSR_pooled": rsr_pooled,
        "J_pooled": j_pooled,
        "RAIR_participant_mean": rair_pm,
        "RSR_participant_mean": rsr_pm,
        "J_participant_mean": rair_pm + rsr_pm - 1.0,
        "J_reconstructed": j_rec,
        "recon_abs_err": abs(j_rec - j_pooled),
        "m_eff_rair": float(p_df.loc[p_df["rair"].notna(), "n0"].mean()),
        "m_eff_rsr": float(p_df.loc[p_df["rsr"].notna(), "n1"].mean()),
        "icc_rair_est": icc_rair,
        "icc_rair_raw": icc_rair_raw,
        "icc_rsr_est": icc_rsr,
        "icc_rsr_raw": icc_rsr_raw,
    }


def audit_he2023_open_data(
    cache_dir: Path = EXTERNAL_DATA_DIR,
    allow_download: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame] | None:
    """Fetch & audit real public trial logs from He et al. (CHI 2023).

    Data are downloaded directly from the authors' public GitHub repository into
    a local cache directory and are never redistributed in this repository.
    """
    base = "https://raw.githubusercontent.com/RichardHGL/CHI2023_DKE/main/anonymous_data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    for fname in ("all_valid_data.csv", "selected_samples.csv"):
        local = cache_dir / fname
        if not local.exists():
            if not allow_download:
                return None
            try:
                with urlopen(f"{base}/{fname}", timeout=15) as resp:
                    local.write_bytes(resp.read())
            except Exception as exc:
                print(f"[Offline / fetch skipped] Could not download {fname}: {exc}")
                return None
        files[fname] = local

    users = pd.read_csv(files["all_valid_data.csv"])
    tasks = pd.read_csv(files["selected_samples.csv"])

    def _pick_col(cols: list[str], candidates: list[str]) -> str | None:
        lower = {c.lower(): c for c in cols}
        for cand in candidates:
            if cand in lower:
                return lower[cand]
        for c in cols:
            if any(cand in c.lower() for cand in candidates):
                return c
        return None

    gt_col = _pick_col(list(tasks.columns), ["answer", "ground_truth", "correct_answer", "label", "gt"])
    ai_col = _pick_col(list(tasks.columns), ["ai_prediction", "prediction", "ai_advice", "ai_answer", "advice", "pred"])
    if gt_col is None or ai_col is None:
        return None

    gt_vals = tasks[gt_col].astype(str).str.strip().tolist()
    ai_vals = tasks[ai_col].astype(str).str.strip().tolist()

    trial_rows = []
    for idx, row in users.iterrows():
        pid = row.get("username", idx)
        for q in range(len(gt_vals)):
            qc, ac = f"question{q}", f"advice{q}"
            if qc not in users.columns or ac not in users.columns:
                continue
            h_ans = str(row[qc]).strip()
            f_ans = str(row[ac]).strip()
            if h_ans in ("", "nan", "None") or f_ans in ("", "nan", "None"):
                continue
            gt, ai = gt_vals[q], ai_vals[q]
            trial_rows.append(
                {
                    "participant_id": pid,
                    "tutorial": row.get("tutorial", np.nan),
                    "xai": row.get("XAI", np.nan),
                    "item_id": q,
                    "human_correct": h_ans == gt,
                    "ai_correct": ai == gt,
                    "final_correct": f_ans == gt,
                    "disagree": h_ans != ai,
                }
            )
    tdf = pd.DataFrame(trial_rows)
    if not len(tdf):
        return None
    summary = audit_trial_log(tdf, k=4)
    summary["source"] = "He et al. (CHI 2023) github.com/RichardHGL/CHI2023_DKE"
    return summary, tdf


def run_all_validations(
    out_dir: Path,
    fetch_he2023: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run all Monte Carlo validations and optional He et al. (2023) open-data audit."""
    out_dir.mkdir(parents=True, exist_ok=True)

    rec_df = validate_reconstruction()
    rec_df.to_csv(out_dir / "reconstruction_validation.csv", index=False)

    icc_df = validate_icc_grid()
    icc_df.to_csv(out_dir / "icc_validation.csv", index=False)

    cov_df = validate_ci_coverage()
    cov_df.to_csv(out_dir / "ci_coverage_validation.csv", index=False)

    # Synthetic trial-level log audit
    rng = np.random.default_rng(123)
    n_p, m_t, rho_true, a_h, a_a = 200, 50, 0.10, 0.60, 0.70
    s = 1.0 / rho_true - 1.0
    p_hit = rng.beta(0.65 * s, 0.35 * s, n_p)
    p_fa = rng.beta(0.35 * s, 0.65 * s, n_p)
    trial_rows = []
    for i in range(n_p):
        h = rng.random(m_t) < a_h
        a = rng.random(m_t) < a_a
        dis = h ^ a
        sw = dis & (rng.random(m_t) < np.where(a, p_hit[i], p_fa[i]))
        f = np.where(sw, a, h)
        for t in range(m_t):
            trial_rows.append((i, bool(h[t]), bool(a[t]), bool(f[t]), bool(dis[t])))
    synth_df = pd.DataFrame(
        trial_rows,
        columns=["participant_id", "human_correct", "ai_correct", "final_correct", "disagree"],
    )
    synth_audit = audit_trial_log(synth_df, k=2)

    he_summary = None
    if fetch_he2023:
        he_res = audit_he2023_open_data(allow_download=True)
        if he_res is not None:
            he_summary, _ = he_res
            (out_dir / "he2023_trial_audit.json").write_text(json.dumps(he_summary, indent=2, sort_keys=True) + "\n")

    summary = {
        "recon_max_abs_err_J": round(float(rec_df.abs_err_J.max()), 4),
        "recon_max_naive_k2_err_k3": round(float(rec_df[rec_df.k == 3].abs_err_naive_k2.max()), 4),
        "recon_max_naive_k2_err_k4": round(float(rec_df[rec_df.k == 4].abs_err_naive_k2.max()), 4),
        "icc_bias_m4_rho10": round(float(icc_df[(icc_df.rho_true == 0.10) & (icc_df.m_eff_target == 4)].bias_clipped.iloc[0]), 4),
        "icc_bias_m8_rho10": round(float(icc_df[(icc_df.rho_true == 0.10) & (icc_df.m_eff_target == 8)].bias_clipped.iloc[0]), 4),
        "icc_bias_m20_rho10": round(float(icc_df[(icc_df.rho_true == 0.10) & (icc_df.m_eff_target == 20)].bias_clipped.iloc[0]), 4),
        "cov_naive_rho10": float(cov_df[cov_df.rho_true == 0.10].cov_naive.iloc[0]),
        "cov_cell_true_rho10": float(cov_df[cov_df.rho_true == 0.10].cov_cell_true.iloc[0]),
        "cov_cell_plugin_rho10": float(cov_df[cov_df.rho_true == 0.10].cov_cell_plugin.iloc[0]),
        "cov_naive_rho20": float(cov_df[cov_df.rho_true == 0.20].cov_naive.iloc[0]),
        "cov_cell_true_rho20": float(cov_df[cov_df.rho_true == 0.20].cov_cell_true.iloc[0]),
        "cov_cell_plugin_rho20": float(cov_df[cov_df.rho_true == 0.20].cov_cell_plugin.iloc[0]),
        "synthetic_trial_audit": synth_audit,
        "he2023_open_trial_audit": he_summary,
    }
    (out_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if verbose:
        print(
            f"Validation complete: Eq.3 max |err|={summary['recon_max_abs_err_J']:.4f} "
            f"(naive k=4 err={summary['recon_max_naive_k2_err_k4']:.4f}) | "
            f"Eq.5 bias(m=8, rho=0.10)={summary['icc_bias_m8_rho10']:+.4f} | "
            f"95% CI cov(rho=0.20): naive={summary['cov_naive_rho20']*100:.1f}%, "
            f"cell={summary['cov_cell_true_rho20']*100:.1f}%, "
            f"plug-in={summary['cov_cell_plugin_rho20']*100:.1f}%"
        )
    return summary
