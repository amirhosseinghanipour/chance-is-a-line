"""Corpus reanalysis, sensitivity sweeps, and exact cell-DEFF power tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DEFAULT_ICC, DEFAULT_MARGIN, REF_A_A, REF_A_H
from .icc import attach_icc, icc_diagnosis
from .inference import classify, classify_point, contrast_interval
from .reconstruction import (
    j_bounds,
    j_from_rair_rsr,
    j_from_reliance,
    rair_rsr_from_reliance,
    split_from_accuracies,
)

NUMERIC_COLS = [
    "a_H_initial", "a_A_stated", "a_A_actual", "rair", "rsr",
    "over_reliance", "under_reliance", "agreement_fraction",
    "switch_fraction", "accuracy_final", "accuracy_wid", "k_options",
    "reporting_tier", "is_reference_condition", "n_participants",
    "n_trials_per_participant", "rair_sd", "rsr_sd", "over_sd",
    "under_sd", "rho_within",
]


def load_extraction(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, comment="#")
    for c in NUMERIC_COLS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "paper_id" not in df.columns:
        raise ValueError("extraction.csv requires a paper_id column.")
    return df


def _get(row: Any, name: str, default: Any = np.nan) -> Any:
    v = row.get(name, default) if hasattr(row, "get") else getattr(row, name, default)
    return default if v is None else v


def _a_a(row: Any) -> float:
    v = _get(row, "a_A_actual")
    return float(_get(row, "a_A_stated")) if pd.isna(v) else float(v)


def _k(row: Any) -> int:
    v = _get(row, "k_options")
    return int(v) if not pd.isna(v) else 2


def resolve_j(row: pd.Series) -> tuple[float, float, float, str]:
    a_a, a_h = _a_a(row), _get(row, "a_H_initial")
    o, u = _get(row, "over_reliance"), _get(row, "under_reliance")
    if not pd.isna(_get(row, "rair")) and not pd.isna(_get(row, "rsr")):
        j = j_from_rair_rsr(float(row["rair"]), float(row["rsr"]))
        return j, j, j, "exact"
    have = not (pd.isna(a_a) or pd.isna(a_h))
    if have and not pd.isna(o) and not pd.isna(u):
        j = j_from_reliance(float(o), float(u), float(a_h), a_a, _k(row))
        return j, j, j, "reconstructed"
    if have and (not pd.isna(o) or not pd.isna(u)):
        lo, hi = j_bounds(
            over=None if pd.isna(o) else float(o),
            under=None if pd.isna(u) else float(u),
            a_H=float(a_h),
            a_A=a_a,
            k=_k(row),
        )
        return float("nan"), lo, hi, "bounded"
    return float("nan"), float("nan"), float("nan"), "unidentified"


def backfill_rair_rsr(df: pd.DataFrame) -> pd.DataFrame:
    df["rair_source"] = np.where(df.rair.notna(), "reported", "")
    need = df.rair.isna() & df.rsr.isna() & df.over_reliance.notna() & df.under_reliance.notna()
    for i in df[need].index:
        r = df.loc[i]
        a_a, a_h = _a_a(r), r.a_H_initial
        if pd.isna(a_a) or pd.isna(a_h):
            continue
        ra, rs = rair_rsr_from_reliance(float(r.over_reliance), float(r.under_reliance), float(a_h), a_a, _k(r))
        df.loc[i, "rair"] = ra
        df.loc[i, "rsr"] = rs
        df.loc[i, "rair_source"] = "reconstructed"
    return df


def null_floor(row: pd.Series) -> float:
    a_a, a_h = _a_a(row), _get(row, "a_H_initial")
    if pd.isna(a_a) or pd.isna(a_h):
        return float("nan")
    n1, _, n_dis = split_from_accuracies(float(a_h), a_a, _k(row))
    p = _get(row, "switch_fraction")
    p = 0.5 if pd.isna(p) else float(p)
    return p * n1 / n_dis if n_dis else float("nan")


def compute_contrasts(
    df: pd.DataFrame,
    icc: float = DEFAULT_ICC,
    margin: float = DEFAULT_MARGIN,
    use_estimated_icc: bool = True,
    rho_within: float = 0.0,
    deff_mode: str = "cell",
) -> pd.DataFrame:
    out = []
    for sid, g in df.groupby("study_id"):
        ref_rows = g[g.is_reference_condition == 1]
        if not len(ref_rows):
            continue
        ref = ref_rows.iloc[0]
        for r in g[g.is_reference_condition != 1].itertuples():
            is_subgroup = str(_get(r, "row_type", "")) == "participant_subgroup"
            rec: dict[str, Any] = {
                "study_id": sid,
                "paper_id": getattr(ref, "paper_id", ""),
                "citation": getattr(ref, "citation", ""),
                "contrast": f"{ref.condition_label} -> {r.condition_label}",
                "tier": r.reporting_tier,
                "contrast_kind": "subgroup_descriptive" if is_subgroup else "randomised_manipulation",
            }
            if pd.isna(ref.rair) or pd.isna(ref.rsr) or pd.isna(r.rair) or pd.isna(r.rsr):
                rec.update(
                    label="unidentified", method="none", dJ=np.nan,
                    dJ_lo=np.nan, dJ_hi=np.nan, dBias=np.nan, half_width=np.nan,
                )
                out.append(rec)
                continue
            row_icc = icc
            if use_estimated_icc:
                cand = _get(r, "icc_used")
                if not pd.isna(cand):
                    row_icc = float(cand)
            paired = str(_get(r, "design", "")).lower().startswith("within")
            rw = _get(r, "rho_within")
            rw = rho_within if pd.isna(rw) else float(rw)
            ci = contrast_interval(
                float(ref.rair), float(ref.rsr), float(r.rair), float(r.rsr),
                ref.a_H_initial, _a_a(ref), ref.n_participants, ref.n_trials_per_participant,
                k=_k(r), icc=row_icc, paired=paired, rho_within=rw, deff_mode=deff_mode,
                a_H_b=_get(r, "a_H_initial"), a_A_b=_a_a(r),
                n_participants_b=r.n_participants, m_b=r.n_trials_per_participant,
            )
            if ci is None or np.isnan(ci["dJ_lo"]):
                dj = (float(r.rair) + float(r.rsr) - 1.0) - (float(ref.rair) + float(ref.rsr) - 1.0)
                db = 0.5 * ((float(r.rair) + 1.0 - float(r.rsr)) - (float(ref.rair) + 1.0 - float(ref.rsr)))
                rec.update(
                    label=classify_point(dj, db), method="point_fallback",
                    dJ=dj, dJ_lo=np.nan, dJ_hi=np.nan, dBias=db, half_width=np.nan,
                )
            else:
                rec.update(
                    label=classify(ci, margin=margin), method="interval",
                    dJ=ci["dJ"], dJ_lo=ci["dJ_lo"], dJ_hi=ci["dJ_hi"],
                    dBias=ci["dBias"], half_width=(ci["dJ_hi"] - ci["dJ_lo"]) / 2.0,
                    icc_used=ci["icc_used"], paired=ci["paired_flag"], deff_mode=deff_mode,
                )
            out.append(rec)
    return pd.DataFrame(out)


def half_width_crossing(
    margin: float,
    m: int,
    icc: float,
    a_H: float = REF_A_H,
    a_A: float = REF_A_A,
    deff_mode: str = "cell",
    lo: float = 5.0,
    hi: float = 20_000.0,
) -> float:
    def _hw(n: float) -> float:
        ci = contrast_interval(0.5, 0.5, 0.5, 0.5, a_H, a_A, n, m, icc=icc, deff_mode=deff_mode)
        assert ci is not None
        return float((ci["dJ_hi"] - ci["dJ_lo"]) / 2.0)

    if _hw(lo) < margin:
        return lo
    if _hw(hi) > margin:
        return float("nan")
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if _hw(mid) > margin:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def build_power_table(
    icc: float = DEFAULT_ICC,
    a_H: float = REF_A_H,
    a_A: float = REF_A_A,
    deff_mode: str = "cell",
) -> pd.DataFrame:
    rows = []
    for n_part in (50, 100, 150, 200, 300, 500):
        for m in (10, 20, 40):
            ci = contrast_interval(0.5, 0.5, 0.5, 0.5, a_H, a_A, n_part, m, icc=icc, deff_mode=deff_mode)
            assert ci is not None
            rows.append(
                {
                    "participants": n_part,
                    "trials": m,
                    "half_width": round(float((ci["dJ_hi"] - ci["dJ_lo"]) / 2.0), 3),
                }
            )
    return pd.DataFrame(rows)


def analyze_corpus(
    csv_path: Path,
    out_dir: Path,
    icc: float = DEFAULT_ICC,
    margin: float = DEFAULT_MARGIN,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run full corpus reanalysis and write all CSV/JSON/TXT outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_extraction(csv_path)
    df[["J", "J_lo", "J_hi", "J_method"]] = df.apply(resolve_j, axis=1, result_type="expand")
    df = backfill_rair_rsr(df)
    df[["J", "J_lo", "J_hi", "J_method"]] = df.apply(resolve_j, axis=1, result_type="expand")

    icc_diag = icc_diagnosis(df)
    df, _, _, icc_source = attach_icc(df, default=icc)
    df["null_over_reliance"] = df.apply(null_floor, axis=1)
    df["over_minus_null"] = df.over_reliance - df.null_over_reliance

    c_cell = compute_contrasts(df, icc, margin, deff_mode="cell")
    c_total = compute_contrasts(df, icc, margin, deff_mode="total")
    iv = c_cell[c_cell.method == "interval"] if len(c_cell) else c_cell
    iv_total = c_total[c_total.method == "interval"] if len(c_total) else c_total

    ident = df[df.J.notna()]
    by_paper = df.groupby("paper_id").n_participants.sum()
    denom = df.over_reliance_denominator.value_counts().to_dict() if "over_reliance_denominator" in df.columns else {}

    claims: dict[str, Any] = {
        "over_reliance_denominators": denom,
        "n_conditions_confounded_denominator": int(sum(v for k, v in denom.items() if k in ("disagreement", "all"))),
        "n_conditions_invariant_denominator": int(denom.get("ai_wrong", 0)),
        "n_conditions": int(len(df)),
        "n_papers": int(df.paper_id.nunique()),
        "n_condition_groups": int(df.study_id.nunique()),
        "n_experimental_conditions": int((df.row_type == "experimental_condition").sum()),
        "n_participant_subgroups": int((df.row_type == "participant_subgroup").sum()),
        "n_J_identified": int(len(ident)),
        "n_J_unidentified": int(df.J.isna().sum()),
        "n_J_negative": int((ident.J < -0.05).sum()),
        "n_J_near_zero": int((ident.J.abs() <= 0.05).sum()),
        "n_J_positive": int((ident.J > 0.05).sum()),
        "J_by_condition": {f"{r.paper_id}/{r.condition_label}": round(float(r.J), 4) for _, r in ident.iterrows()},
        "reported_a_H": int(df.a_H_initial.notna().sum()),
        "reported_over_reliance": int(df.over_reliance.notna().sum()),
        "reported_rair": int(df.rair.notna().sum()),
        "reported_sd": int(df.rair_sd.notna().sum()),
        "n_contrasts": int(len(c_cell)),
        "n_contrasts_interval": int(len(iv)),
        "n_icc_estimated": int(icc_diag.icc_est.notna().sum()),
        "icc_used_source": "assumed_default" if icc_diag.icc_est.notna().sum() == 0 else icc_source,
        "icc_default": float(icc),
        "margin": float(margin),
        "deff_mode_primary": "cell",
        "median_half_width": round(float(iv.half_width.median()), 4) if len(iv) else None,
        "median_half_width_total_deff": round(float(iv_total.half_width.median()), 4) if len(iv_total) else None,
        "schemmer_dJ_ci": [round(float(iv.dJ_lo.iloc[0]), 4), round(float(iv.dJ_hi.iloc[0]), 4)] if len(iv) else None,
        "N_per_condition_min": int(df.n_participants.min()),
        "N_per_condition_mean": round(float(df.n_participants.mean()), 2),
        "N_per_condition_median": float(df.n_participants.median()),
        "N_per_condition_max": int(df.n_participants.max()),
        "N_total_per_paper": {str(k): int(v) for k, v in by_paper.items()},
        "N_total_per_paper_mean": round(float(by_paper.mean()), 2),
        "crossing_N": {str(m): round(half_width_crossing(margin, m, icc, deff_mode="cell"), 1) for m in (10, 20, 40)},
        "crossing_N_total_deff": {
            str(m): round(half_width_crossing(margin, m, icc, deff_mode="total"), 1) for m in (10, 20, 40)
        },
    }

    df.to_csv(out_dir / "conditions_with_J.csv", index=False)
    c_cell.to_csv(out_dir / "contrasts.csv", index=False)
    c_total.to_csv(out_dir / "contrasts_total_deff.csv", index=False)
    icc_diag.to_csv(out_dir / "icc_diagnosis.csv", index=False)
    build_power_table(icc=icc).to_csv(out_dir / "power_table.csv", index=False)

    sens_rows = []
    for i in (0.0, 0.05, 0.10, 0.20):
        for mg in (0.05, 0.10, 0.15):
            c_s = compute_contrasts(df, i, mg, use_estimated_icc=False, deff_mode="cell")
            ivs = c_s[c_s.method == "interval"]
            cl = ivs[ivs.label.isin(["bias", "skill", "mixed", "null"])]
            sens_rows.append(
                {
                    "icc": i,
                    "icc_mode": "assumed",
                    "rho_within": 0.0,
                    "margin": mg,
                    "n_interval": len(ivs),
                    "n_classifiable": len(cl),
                    "frac_classifiable": len(cl) / len(ivs) if len(ivs) else np.nan,
                    "frac_bias": (cl.label == "bias").mean() if len(cl) else np.nan,
                    "half_width": float(ivs.half_width.iloc[0]) if len(ivs) else np.nan,
                }
            )
    pd.DataFrame(sens_rows).to_csv(out_dir / "sensitivity.csv", index=False)
    (out_dir / "claims.json").write_text(json.dumps(claims, indent=2, sort_keys=True) + "\n")

    report_lines = [
        f"ICC = {icc} | margin = +/-{margin} | DEFF mode = cell-specific m_eff",
        f"conditions: {claims['n_conditions']} | papers: {claims['n_papers']} | groups: {claims['n_condition_groups']}",
        f"denominators: confounded={claims['n_conditions_confounded_denominator']}/{claims['n_conditions']}, "
        f"invariant={claims['n_conditions_invariant_denominator']}/{claims['n_conditions']}",
        f"J identified: {claims['n_J_identified']}/{claims['n_conditions']} "
        f"(negative={claims['n_J_negative']}, near_zero={claims['n_J_near_zero']}, positive={claims['n_J_positive']})",
        f"empirical ICC estimates: {claims['n_icc_estimated']}/{claims['n_conditions']}",
        f"Schemmer dJ 95% CI: {claims['schemmer_dJ_ci']} (half-width={claims['median_half_width']:.3f}; "
        f"total-DEFF={claims['median_half_width_total_deff']:.3f})",
        f"N_cross (cell-DEFF): m=10 -> {claims['crossing_N']['10']:.0f}, "
        f"m=20 -> {claims['crossing_N']['20']:.0f}, m=40 -> {claims['crossing_N']['40']:.0f}",
    ]
    report_txt = "\n".join(report_lines) + "\n"
    (out_dir / "report.txt").write_text(report_txt)
    if verbose:
        print(report_txt)
    return claims
