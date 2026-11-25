"""Command-line interface for end-to-end paper reproduction."""

from __future__ import annotations

import argparse
from pathlib import Path

from .audit import verify_extraction
from .config import DATA_DIR, DEFAULT_ICC, DEFAULT_MARGIN, FIGURES_DIR, RESULTS_DIR
from .corpus import analyze_corpus
from .export import export_brm_artifacts
from .figures import generate_all_figures
from .simulation import verify_analytic_claims
from .validation import run_all_validations


def reproduce_all(
    extraction_csv: Path = DATA_DIR / "extraction.csv",
    prisma_csv: Path = DATA_DIR / "prisma_screening.csv",
    results_dir: Path = RESULTS_DIR,
    figures_dir: Path = FIGURES_DIR,
    icc: float = DEFAULT_ICC,
    margin: float = DEFAULT_MARGIN,
    fetch_he2023: bool = True,
) -> None:
    """Execute the full deterministic reproduction pipeline."""
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Verifying Section 2 analytic claims ...")
    verify_analytic_claims(out_path=results_dir / "claims_analytic.json")

    print("\n[2/6] Running extraction audit guardrails ...")
    issues, warns = verify_extraction(extraction_csv)
    if issues or warns:
        raise RuntimeError(
            f"Extraction audit failed: {len(issues)} issues, {len(warns)} warnings"
        )

    print("\n[3/6] Running corpus reanalysis & exact cell-DEFF power audit ...")
    analyze_corpus(extraction_csv, results_dir, icc=icc, margin=margin)

    print(
        "\n[4/6] Running Monte Carlo validation (Eqs. 3, 5, 95% CI coverage) & open-data audit ..."
    )
    run_all_validations(results_dir, fetch_he2023=fetch_he2023)

    print("\n[5/6] Generating publication figures ...")
    generate_all_figures(results_dir, figures_dir)

    print("\n[6/6] Exporting APA 7th LaTeX tables, macros, and full audit files ...")
    paths = export_brm_artifacts(results_dir, prisma_csv)
    for p in paths:
        print(f"  - {p}")
    print("\nPipeline complete.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="chance-is-a-line",
        description="Reproducibility pipeline for 'Chance Is a Line: Why RAIR and RSR Must Be Read Together'.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_all = sub.add_parser("reproduce", help="Run the entire pipeline end-to-end")
    p_all.add_argument("--extraction", type=Path, default=DATA_DIR / "extraction.csv")
    p_all.add_argument("--prisma", type=Path, default=DATA_DIR / "prisma_screening.csv")
    p_all.add_argument("--results", type=Path, default=RESULTS_DIR)
    p_all.add_argument("--figures", type=Path, default=FIGURES_DIR)
    p_all.add_argument("--icc", type=float, default=DEFAULT_ICC)
    p_all.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    p_all.add_argument(
        "--offline",
        action="store_true",
        help="Skip fetching He et al. (2023) trial logs from GitHub",
    )

    p_claims = sub.add_parser("claims", help="Verify Section 2 analytic claims")
    p_claims.add_argument(
        "--out", type=Path, default=RESULTS_DIR / "claims_analytic.json"
    )

    p_audit = sub.add_parser(
        "audit", help="Verify extraction.csv against verbatim quotes"
    )
    p_audit.add_argument("--extraction", type=Path, default=DATA_DIR / "extraction.csv")

    p_corpus = sub.add_parser("corpus", help="Run corpus reanalysis")
    p_corpus.add_argument(
        "--extraction", type=Path, default=DATA_DIR / "extraction.csv"
    )
    p_corpus.add_argument("--results", type=Path, default=RESULTS_DIR)
    p_corpus.add_argument("--icc", type=float, default=DEFAULT_ICC)
    p_corpus.add_argument("--margin", type=float, default=DEFAULT_MARGIN)

    p_val = sub.add_parser(
        "validate", help="Run Monte Carlo validation & He et al. (2023) open-data audit"
    )
    p_val.add_argument("--results", type=Path, default=RESULTS_DIR)
    p_val.add_argument("--offline", action="store_true")

    p_fig = sub.add_parser("figures", help="Generate publication figures")
    p_fig.add_argument("--results", type=Path, default=RESULTS_DIR)
    p_fig.add_argument("--figures", type=Path, default=FIGURES_DIR)

    p_exp = sub.add_parser("export", help="Export APA 7th LaTeX tables and macros")
    p_exp.add_argument("--results", type=Path, default=RESULTS_DIR)
    p_exp.add_argument("--prisma", type=Path, default=DATA_DIR / "prisma_screening.csv")

    args = parser.parse_args(argv)

    if args.cmd == "reproduce":
        reproduce_all(
            extraction_csv=args.extraction,
            prisma_csv=args.prisma,
            results_dir=args.results,
            figures_dir=args.figures,
            icc=args.icc,
            margin=args.margin,
            fetch_he2023=not args.offline,
        )
    elif args.cmd == "claims":
        verify_analytic_claims(out_path=args.out)
    elif args.cmd == "audit":
        issues, _ = verify_extraction(args.extraction)
        if issues:
            raise SystemExit(1)
    elif args.cmd == "corpus":
        analyze_corpus(args.extraction, args.results, icc=args.icc, margin=args.margin)
    elif args.cmd == "validate":
        run_all_validations(args.results, fetch_he2023=not args.offline)
    elif args.cmd == "figures":
        generate_all_figures(args.results, args.figures)
    elif args.cmd == "export":
        export_brm_artifacts(args.results, args.prisma)


if __name__ == "__main__":
    main()
