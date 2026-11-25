"""Extraction verification guardrails and validated condition insertion helper."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VALUE_EVIDENCE: dict[str, str] = {
    "rair": "evidence_rair",
    "rsr": "evidence_rsr",
    "over_reliance": "evidence_over",
    "under_reliance": "evidence_under",
    "a_A_actual": "evidence_a_A",
    "a_H_initial": "evidence_a_H",
    "n_participants": "evidence_n",
    "n_trials_per_participant": "evidence_trials",
}

WORD_NUM: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12,
    "sixteen": 16, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60,
}

REQUIRED_FIELDS = [
    "study_id", "paper_id", "condition_label", "is_reference_condition",
    "row_type", "interpretation_note", "over_reliance_denominator",
]
VALID_DENOMINATORS = ("disagreement", "all", "ai_wrong")


def numbers_in(text: Any) -> set[float]:
    if not isinstance(text, str):
        return set()
    out: set[float] = set()
    for match in re.findall(r"\d+\.?\d*", text.replace(",", "")):
        try:
            v = float(match)
        except ValueError:
            continue
        out.add(v)
        if v > 1:
            out.add(round(v / 100.0, 6))
    return out


def value_supported(value: Any, snippet: Any, tol: float = 0.005) -> tuple[bool, str]:
    if pd.isna(value):
        return True, "not reported"
    if isinstance(snippet, str) and snippet.strip().upper().startswith("DERIVED:"):
        return True, "derived"
    cands = numbers_in(snippet)
    if isinstance(snippet, str):
        for w, v in WORD_NUM.items():
            if re.search(rf"\b{w}\b", snippet, re.I):
                cands.add(float(v))
    if not cands:
        return False, "NO EVIDENCE SNIPPET"
    if any(abs(float(value) - c) <= tol for c in cands):
        return True, "ok"
    return False, f"value {value} not found in snippet"


def check_extraction_df(df: pd.DataFrame) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    issues: list[tuple[str, str, str]] = []
    derived: list[tuple[str, str, str]] = []
    for r in df.itertuples():
        tag = f"{r.study_id}/{r.condition_label}"
        for col, ev in VALUE_EVIDENCE.items():
            val = getattr(r, col, np.nan)
            snip = getattr(r, ev, np.nan) if ev in df.columns else np.nan
            ok, msg = value_supported(val, snip)
            if not ok:
                issues.append((tag, col, msg))
            elif msg == "derived":
                derived.append((tag, col, str(snip)[:90]))
        for col in [
            "rair", "rsr", "over_reliance", "under_reliance",
            "a_H_initial", "a_A_actual", "accuracy_final",
        ]:
            v = getattr(r, col, np.nan)
            if not pd.isna(v) and not (0.0 <= float(v) <= 1.0):
                issues.append((tag, col, f"out of [0,1]: {v}"))
        if not pd.isna(getattr(r, "rair", np.nan)) and not pd.isna(getattr(r, "rsr", np.nan)):
            j = float(r.rair) + float(r.rsr) - 1.0
            if not (-1.0 <= j <= 1.0):
                issues.append((tag, "J", f"impossible: {j}"))
        note = getattr(r, "interpretation_note", "")
        if not isinstance(note, str) or not note.strip():
            issues.append((tag, "interpretation_note", "MISSING - state what this row is"))
    return issues, derived


def check_denominator_declared(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    if "over_reliance_denominator" not in df.columns:
        return [("CORPUS", "over_reliance_denominator", "column missing")]
    valid = set(VALID_DENOMINATORS)
    out: list[tuple[str, str, str]] = []
    for r in df.itertuples():
        v = str(getattr(r, "over_reliance_denominator", "")).strip()
        if v not in valid:
            out.append((f"{r.study_id}/{r.condition_label}", "over_reliance_denominator", f"{v!r} invalid"))
    return out


def soft_warnings(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    warns: list[tuple[str, str, str]] = []
    for r in df.itertuples():
        tag = f"{r.study_id}/{r.condition_label}"
        ev_n = str(getattr(r, "evidence_n", "") or "")
        n = getattr(r, "n_participants", np.nan)
        if (
            not pd.isna(n)
            and re.search(r"\btotal\b|\brecruited\b", ev_n, re.I)
            and not re.search(r"\bhalf\b|\beach\b|\bper arm\b", ev_n, re.I)
        ):
            warns.append((tag, "n_participants", f"N={int(n)} cited from recruitment total; confirm no split"))
        ev_t = str(getattr(r, "evidence_trials", "") or "")
        lbl = str(getattr(r, "condition_label", "")).lower()
        if (
            "DERIVED" in ev_t
            and not re.search(r"\bhalf\b|\bcut\b", ev_t, re.I)
            and any(k in lbl for k in ("hard", "long", "short", "reduced"))
        ):
            warns.append((tag, "n_trials_per_participant", "derived from general statement, but label suggests shortened protocol"))
    return warns


def verify_extraction(csv_path: Path, verbose: bool = True) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    df = pd.read_csv(csv_path, comment="#")
    for c in list(VALUE_EVIDENCE) + ["accuracy_final"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    issues, derived = check_extraction_df(df)
    issues += check_denominator_declared(df)
    warns = soft_warnings(df)
    if verbose:
        print(f"conditions: {len(df)}   studies: {df.study_id.nunique()}   papers: {df.paper_id.nunique()}")
        print(f"{len(derived)} DERIVED values | {len(issues)} issues | {len(warns)} warnings")
        for tag, col, msg in issues:
            print(f"  ISSUE:   {tag:40s} {col:22s} {msg}")
        for tag, col, msg in warns:
            print(f"  WARNING: {tag:40s} {col:22s} {msg}")
    return issues, warns


def add_condition(csv_path: Path, **kw: Any) -> pd.DataFrame:
    """Append one condition to extraction.csv with strict pre-write validation."""
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_FIELDS if kw.get(c) is None]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if kw["row_type"] not in ("experimental_condition", "participant_subgroup"):
        raise ValueError("row_type must be 'experimental_condition' or 'participant_subgroup'")
    if kw["over_reliance_denominator"] not in VALID_DENOMINATORS:
        raise ValueError(f"over_reliance_denominator must be one of {list(VALID_DENOMINATORS)}")

    for val, ev in VALUE_EVIDENCE.items():
        if kw.get(val) is not None and not kw.get(ev):
            raise ValueError(f"'{val}' given without '{ev}' - quote source or write 'DERIVED: ...'")

    for c in ["rair", "rsr", "over_reliance", "under_reliance", "a_H_initial", "a_A_actual"]:
        v = kw.get(c)
        if v is not None and not (0 <= float(v) <= 1):
            raise ValueError(f"{c}={v} outside [0, 1] - use proportions, not percentages")

    for k in kw:
        if k not in df.columns:
            df[k] = np.nan
    row = {c: kw.get(c, np.nan) for c in df.columns}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return df
