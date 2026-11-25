"""End-to-end extraction audit and artifact regression tests."""

from __future__ import annotations

import json
import unittest

import pandas as pd

from chance_is_a_line.audit import check_denominator_declared, soft_warnings, verify_extraction
from chance_is_a_line.config import DATA_DIR, FIGURES_DIR, RESULTS_DIR


class TestPipelineIntegrity(unittest.TestCase):
    def test_extraction_audit_zero_issues_and_warnings(self) -> None:
        issues, warns = verify_extraction(DATA_DIR / "extraction.csv", verbose=False)
        self.assertEqual(issues, [])
        self.assertEqual(warns, [])

    def test_guardrails_catch_known_traps(self) -> None:
        bad = pd.DataFrame(
            [
                {
                    "study_id": "vasconcelos2023",
                    "condition_label": "hard_prediction_only",
                    "n_participants": 170,
                    "n_trials_per_participant": 6,
                    "evidence_n": "We recruited a total of N = 340 participants",
                    "evidence_trials": "DERIVED: 30 test questions",
                }
            ]
        )
        self.assertEqual(len(soft_warnings(bad)), 2)
        self.assertEqual(len(check_denominator_declared(bad)), 1)

    def test_generated_outputs_match_paper_claims(self) -> None:
        claims_path = RESULTS_DIR / "claims.json"
        if not claims_path.exists():
            self.skipTest("Run `python scripts/run_all.py reproduce` first to verify generated artifacts.")
        claims = json.loads(claims_path.read_text())
        self.assertEqual(claims["n_conditions"], 13)
        self.assertEqual(claims["n_papers"], 4)
        self.assertEqual(claims["n_conditions_confounded_denominator"], 10)
        self.assertEqual(claims["n_conditions_invariant_denominator"], 3)
        self.assertEqual(claims["N_per_condition_max"], 100)
        self.assertEqual(claims["n_J_identified"], 10)
        self.assertEqual(claims["n_J_unidentified"], 3)
        self.assertEqual(claims["n_J_negative"], 5)
        self.assertEqual(claims["n_icc_estimated"], 0)
        self.assertEqual(claims["deff_mode_primary"], "cell")
        self.assertLess(claims["median_half_width"], claims["median_half_width_total_deff"])

        val = json.loads((RESULTS_DIR / "validation_summary.json").read_text())
        self.assertLess(val["recon_max_abs_err_J"], 0.01)
        self.assertGreater(val["recon_max_naive_k2_err_k4"], 0.30)
        self.assertLess(abs(val["icc_bias_m8_rho10"]), 0.025)
        self.assertLess(val["cov_naive_rho20"], 0.80)
        self.assertTrue(0.91 <= val["cov_cell_true_rho20"] <= 0.98)
        self.assertTrue(0.90 <= val["cov_cell_plugin_rho20"] <= 0.99)

        for fname in (
            "table1_recomputed_J.tex",
            "tableB1_extraction_audit.tex",
            "tableC1_ci_coverage.tex",
            "manuscript_macros.tex",
        ):
            txt = (RESULTS_DIR / fname).read_text()
            self.assertGreater(len(txt), 100)
            self.assertNotIn(r"\cite{", txt)

        for fig in ("fig1_plane.pdf", "fig2_floor.pdf", "fig3_power.pdf", "fig4_validation.pdf"):
            self.assertTrue((FIGURES_DIR / fig).exists())


if __name__ == "__main__":
    unittest.main()
