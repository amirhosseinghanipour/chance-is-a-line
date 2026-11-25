# Chance Is a Line: Why RAIR and RSR Must Be Read Together

Official reproducibility repository for *"Chance Is a Line: Why RAIR and RSR Must Be Read Together"* (prepared for *Behavior Research Methods*).

All analytic verifications, corpus extraction audits, Monte Carlo simulations (multiclass Eq. 3 reconstruction, Eq. 5 method-of-moments ICC bias, and 95% CI coverage), publication figures, and APA 7th LaTeX tables/macros reproduce deterministically from a single command on CPU in under 60 seconds.

## Repository Layout

```text
chance-is-a-line/
├── data/
│   ├── extraction.csv             # Verbatim-audited corpus (13 conditions, 4 studies)
│   └── prisma_screening.csv       # Verified PRISMA counts from Raees & Papangelis (2026)
├── src/chance_is_a_line/
│   ├── config.py                  # Central constants, paths, and RNG seeds
│   ├── simulation.py              # Judge-Advisor System simulator & Section 2 claims
│   ├── reconstruction.py          # Multiclass RAIR/RSR/J reconstruction & bounds (Eq. 3)
│   ├── inference.py               # Cell-specific DEFF CIs & equivalence tests (Eq. 4)
│   ├── icc.py                     # Method-of-moments ICC & beta-binomial simulator (Eq. 5)
│   ├── audit.py                   # Verbatim extraction verifier & add_condition() helper
│   ├── corpus.py                  # Corpus reanalysis, sensitivity sweeps, & power table
│   ├── validation.py              # Monte Carlo benchmarks & He et al. (2023) trial-log audit
│   ├── figures.py                 # Publication figures (fig1–fig4, PDF + 300 DPI PNG)
│   ├── export.py                  # APA 7th LaTeX tables, macros, & full audit CSV/MD
│   └── cli.py                     # CLI entry point (`chance-is-a-line`)
├── scripts/
│   └── run_all.py                 # Standalone runner (works without installing package)
├── tests/                         # Unit + end-to-end regression test suite (pytest)
├── Makefile
├── pyproject.toml
├── CITATION.cff
└── LICENSE
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Reproduce all numbers, Monte Carlo benchmarks, figures, and LaTeX tables
chance-is-a-line reproduce

# Run unit + end-to-end regression tests
pytest -v
```

Without installing the package, run directly via:

```bash
python scripts/run_all.py reproduce
```

To run completely offline (skipping the GitHub download of He et al., 2023):

```bash
python scripts/run_all.py reproduce --offline
```

## Generated Outputs

Running `reproduce` writes all generated artifacts under `outputs/` (ignored by Git):

* `outputs/results/claims_analytic.json` — Section 2 analytic & Monte Carlo claims
* `outputs/results/claims.json`, `conditions_with_J.csv`, `contrasts.csv`, `sensitivity.csv`, `power_table.csv` — corpus audit & exact cell-DEFF power tables
* `outputs/results/reconstruction_validation.csv`, `icc_validation.csv`, `ci_coverage_validation.csv`, `validation_summary.json` — Monte Carlo validation benchmarks
* `outputs/results/he2023_trial_audit.json` — live audit of He et al. (CHI 2023) disaggregated trial logs (when online)
* `outputs/results/manuscript_macros.tex`, `table1_recomputed_J.tex`, `tableC1_ci_coverage.tex`, `tableB1_extraction_audit.tex`, `extraction_audit_full.csv`, `extraction_audit_full.md` — publication-ready APA 7th exports
* `outputs/figures/fig1_plane.pdf`, `fig2_floor.pdf`, `fig3_power.pdf`, `fig4_validation.pdf` — vector PDFs (+ 300 DPI PNGs)

## Third-Party Data Notice

The open trial-level audit of He et al. (CHI 2023) fetches `anonymous_data/all_valid_data.csv` and `anonymous_data/selected_samples.csv` directly from the authors' official public repository (`https://github.com/RichardHGL/CHI2023_DKE`) into `data/external/` (git-ignored). In accordance with the original authors' data-use terms, their raw files are **not** redistributed in this repository.

## Auditing Your Own Trial-Level Data

Pass any trial-level CSV containing `participant_id`, `human_correct`, `ai_correct`, `final_correct` (and optionally `disagree`) to `audit_trial_log`:

```python
import pandas as pd
from chance_is_a_line.validation import audit_trial_log

trials = pd.read_csv("your_trials.csv")
summary = audit_trial_log(trials, k=2)
print(summary)
```
