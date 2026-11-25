"""Publication figures for Behavior Research Methods (Springer Nature)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import DEFAULT_ICC, DEFAULT_MARGIN, REF_A_H
from .inference import contrast_interval
from .simulation import analytic_over_reliance

C_BIAS, C_SKILL, C_MIX, C_NULL = "#C0522E", "#3D5A73", "#7E57A2", "#8A8A8A"

MARKERS = {
    "schemmer2023": "o",
    "vonfelten2025": "s",
    "he2023": "^",
    "vasconcelos2023": "D",
}
SHORT_LABELS = {
    "schemmer2023": "Schemmer et al. (2023)",
    "vonfelten2025": "von Felten et al. (2025)",
    "he2023": "He et al. (2023)",
    "vasconcelos2023": "Vasconcelos et al. (2023)",
}


def apply_style() -> None:
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font="DejaVu Serif",
        rc={
            "font.size": 9,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 7.2,
            "grid.linewidth": 0.5,
            "axes.edgecolor": "#333",
            "axes.linewidth": 0.8,
            "figure.dpi": 300,
        },
    )


def plot_reliance_plane(cond: pd.DataFrame | None, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    x = np.linspace(0, 1, 200)
    ax.fill_between(x, 1 - x, 1, color="#2E7D32", alpha=0.05, lw=0)
    ax.fill_between(x, 0, 1 - x, color="#B71C1C", alpha=0.05, lw=0)
    ax.plot(x, 1 - x, "k--", lw=1.2, zorder=3)
    ax.text(0.76, 0.30, "no discrimination\n(J = 0)", rotation=-45, ha="center", va="center", fontsize=7.0, color="#444")

    star_xy = (1.0 / 3.0, 1.0 / 3.0)
    ax.plot([star_xy[0]], [star_xy[1]], marker="*", color="#111", ms=10.5, zorder=7)
    ax.annotate(
        "prior baseline\n(0.33, 0.33)",
        xy=star_xy,
        xytext=(0.56, 0.13),
        ha="left",
        va="center",
        fontsize=6.4,
        color="#222",
        arrowprops=dict(arrowstyle="->", color="#555", lw=0.7, shrinkA=2, shrinkB=5, connectionstyle="arc3,rad=-0.12"),
        zorder=8,
    )

    if cond is not None and len(cond):
        d = cond.dropna(subset=["rair", "rsr"]).copy()
        for pid, g in d.groupby("paper_id"):
            mk = MARKERS.get(str(pid), "o")
            for rt, gg in g.groupby("row_type"):
                filled = rt == "experimental_condition"
                ax.scatter(
                    gg.rair, gg.rsr, s=34, marker=mk, zorder=5,
                    facecolors="#222" if filled else "none",
                    edgecolors="#222", linewidths=1.05, alpha=0.9,
                )
        handles = [
            plt.Line2D([], [], ls="", marker=MARKERS.get(str(p), "o"), mfc="#222", mec="#222", ms=5, label=SHORT_LABELS.get(str(p), str(p)))
            for p in sorted(d.paper_id.unique())
        ]
        handles.append(plt.Line2D([], [], ls="", marker="o", mfc="none", mec="#222", ms=5, label="Post-hoc subgroup (hollow)"))
        ax.legend(
            handles=handles, frameon=True, facecolor="white", framealpha=0.88,
            edgecolor="none", fontsize=6.0, loc="lower left",
            handletextpad=0.35, borderpad=0.25, labelspacing=0.28,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("RAIR  (hit rate)")
    ax.set_ylabel("RSR  (correct-rejection rate)")
    plt.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig1_plane.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_over_reliance_floor(out_dir: Path, a_H: float = REF_A_H, mark: float = 0.14) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    a_as = np.linspace(0.50, 0.95, 200)
    curves = [
        ("always accept", 1.00, "#B71C1C"),
        ("coin flip", 0.50, C_BIAS),
        ("mostly ignore", 0.25, "#E0A458"),
    ]
    for name, p_sw, col in curves:
        ys = [analytic_over_reliance(a_H, float(a), p_switch=p_sw) for a in a_as]
        ax.plot(a_as, ys, lw=1.8, label=name, color=col)

    ax.axhline(mark, color="#444", lw=0.9, ls=":", zorder=1)
    ax.text(0.505, mark + 0.014, f"one reported value ({mark:.2f})", fontsize=6.5, color="#444")
    for _, p_sw, col in curves:
        ys = np.array([analytic_over_reliance(a_H, float(a), p_switch=p_sw) for a in a_as])
        if ys.min() <= mark <= ys.max():
            a_star = float(np.interp(mark, ys[::-1], a_as[::-1]))
            ax.plot([a_star], [mark], marker="o", ms=5, color=col, mec="white", mew=0.8, zorder=6)
            ax.annotate(f"{a_star:.2f}", xy=(a_star, mark), xytext=(0, -11), textcoords="offset points", ha="center", fontsize=6.3, color=col)

    ax.set_xlabel("AI accuracy ($a_A$)")
    ax.set_ylabel("Reported over-reliance\n(disagreement denominator)")
    ax.text(0.95, 0.58, f"pre-advice human\naccuracy $a_H$ = {a_H:.2f}", ha="right", va="top", fontsize=6.5, color="#555", transform=ax.transAxes)
    ax.set_ylim(0, 0.65)
    ax.legend(frameon=False, fontsize=7.2, loc="upper right")
    plt.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig2_floor.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_power_curves(
    out_dir: Path,
    margin: float = DEFAULT_MARGIN,
    icc: float = DEFAULT_ICC,
    corpus_max_n: float | None = None,
    corpus_med_n: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    ns = np.linspace(30, 500, 150)
    for m, ls in [(10, ":"), (20, "-"), (40, "--")]:
        hw = []
        for n in ns:
            ci = contrast_interval(0.5, 0.5, 0.5, 0.5, 0.60, 0.75, float(n), m, icc=icc, deff_mode="cell")
            assert ci is not None
            hw.append((ci["dJ_hi"] - ci["dJ_lo"]) / 2.0)
        ax.plot(ns, hw, ls, lw=1.75, color=C_SKILL, label=f"{m} trials / participant")
    ax.axhline(margin, color=C_BIAS, lw=1.35)
    ax.text(492, margin + 0.007, f"equivalence margin ({margin:.2f})", ha="right", fontsize=7.2, color=C_BIAS)
    if corpus_med_n:
        ax.axvline(corpus_med_n, color="#999", lw=0.95, ls=":")
        ax.text(corpus_med_n + 4, 0.262, f"median\n(N={int(corpus_med_n)})", fontsize=6.2, color="#666")
    if corpus_max_n:
        ax.axvline(corpus_max_n, color="#444", lw=1.1, ls="-.")
        ax.text(corpus_max_n + 7, 0.220, f"corpus max\n(N = {int(corpus_max_n)})", fontsize=6.5, color="#333")
    ax.set_xlabel("Participants per condition ($N$)")
    ax.set_ylabel(r"Half-width of 95% CI for $\Delta J$")
    ax.legend(frameon=False, fontsize=7.4, loc="upper right")
    ax.set_ylim(0, 0.30)
    ax.set_xlim(25, 505)
    plt.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig3_power.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_validation_panels(res_dir: Path, out_dir: Path) -> None:
    f_icc = res_dir / "icc_validation.csv"
    f_cov = res_dir / "ci_coverage_validation.csv"
    if not (f_icc.exists() and f_cov.exists()):
        return
    df_icc = pd.read_csv(f_icc)
    df_cov = pd.read_csv(f_cov)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.85))
    palette = {0.00: "#8A8A8A", 0.05: "#6BAED6", 0.10: "#3D5A73", 0.20: "#C0522E", 0.30: "#7E57A2"}
    for rho, g in df_icc.groupby("rho_true"):
        ax1.plot(g.m_eff_target, g.bias_clipped, marker="o", ms=3.8, lw=1.4, color=palette.get(float(rho), "#333"), label=f"$\\rho$ = {float(rho):.2f}")
    ax1.axhline(0.0, color="#222", lw=0.8, ls=":")
    ax1.axvline(8, color="#B71C1C", lw=1.1, ls="--")
    ax1.text(8.6, 0.068, "MIN_M_EFF = 8", fontsize=6.8, color="#B71C1C")
    ax1.set_xlabel(r"Effective cell trials / participant ($m_{\mathrm{eff}}$)")
    ax1.set_ylabel(r"Mean bias ($\hat{\rho}_{\mathrm{clipped}} - \rho$)")
    ax1.set_title("A. Eq. (5) ICC Estimation Bias", fontsize=8.8)
    ax1.legend(frameon=False, fontsize=6.4, ncol=2, loc="upper right")

    ax2.plot(df_cov.rho_true, df_cov.cov_naive * 100, marker="s", ms=4.5, lw=1.6, color="#B71C1C", ls="--", label=r"Naive unclustered ($\rho=0$)")
    ax2.plot(df_cov.rho_true, df_cov.cov_cell_true * 100, marker="o", ms=4.5, lw=1.6, color="#3D5A73", label=r"Cell-DEFF (true $\rho$)")
    ax2.plot(df_cov.rho_true, df_cov.cov_cell_plugin * 100, marker="^", ms=4.5, lw=1.6, color="#2E7D32", ls="-.", label=r"Cell-DEFF (plug-in $\hat{\rho}$)")
    ax2.axhline(95.0, color="#333", lw=0.9, ls=":")
    ax2.text(0.295, 91.5, "nominal 95%", ha="right", fontsize=6.6, color="#333")
    ax2.set_xlabel(r"True intraclass correlation ($\rho$)")
    ax2.set_ylabel("Empirical 95% CI coverage (%)")
    ax2.set_title("B. Empirical 95% CI Coverage for $J$", fontsize=8.8)
    ax2.set_ylim(55, 101)
    ax2.legend(frameon=False, fontsize=6.6, loc="lower left")

    plt.tight_layout(pad=0.5)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig4_validation.{ext}", bbox_inches="tight")
    plt.close(fig)


def generate_all_figures(res_dir: Path, out_dir: Path) -> None:
    apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    cond_path = res_dir / "conditions_with_J.csv"
    cond = pd.read_csv(cond_path) if cond_path.exists() else None
    max_n = float(cond.n_participants.max()) if cond is not None else None
    med_n = float(cond.n_participants.median()) if cond is not None else None
    plot_reliance_plane(cond, out_dir)
    plot_over_reliance_floor(out_dir)
    plot_power_curves(out_dir, corpus_max_n=max_n, corpus_med_n=med_n)
    plot_validation_panels(res_dir, out_dir)
