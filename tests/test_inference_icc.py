"""Unit tests for Eq. (5) ICC estimation and exact cell-specific DEFF intervals."""

from __future__ import annotations

import unittest

from chance_is_a_line.icc import icc_from_sd, simulate_participants
from chance_is_a_line.inference import classify, contrast_interval, j_interval


class TestInferenceAndICC(unittest.TestCase):
    def test_icc_recovery_large_m_eff(self) -> None:
        d = simulate_participants(n_participants=500, m_trials=60, sens=0.60, spec=0.60, icc=0.15, seed=9)
        v = d["rair"].dropna()
        m_eff = float(d.loc[v.index, "n0"].mean())
        est, _ = icc_from_sd(float(v.mean()), float(v.std(ddof=1)), m_eff)
        self.assertGreaterEqual(m_eff, 8.0)
        self.assertLess(abs(est - 0.15), 0.04)

    def test_cell_deff_tighter_than_total_deff(self) -> None:
        _, lo_c, hi_c = j_interval(0.30, 0.70, 0.55, 0.50, 100, 16, icc=0.10, deff_mode="cell")
        _, lo_t, hi_t = j_interval(0.30, 0.70, 0.55, 0.50, 100, 16, icc=0.10, deff_mode="total")
        self.assertLess(hi_c - lo_c, hi_t - lo_t)

    def test_schemmer_arm_specific_contrast_indeterminate(self) -> None:
        ci = contrast_interval(
            0.2959, 0.7187, 0.3887, 0.6945,
            a_H=0.555, a_A=0.50, n_participants=100, m=16,
            a_H_b=0.5385, a_A_b=0.50, n_participants_b=99, m_b=16,
            icc=0.10, deff_mode="cell",
        )
        assert ci is not None
        hw = (ci["dJ_hi"] - ci["dJ_lo"]) / 2.0
        self.assertAlmostEqual(hw, 0.1043, places=3)
        self.assertEqual(classify(ci, margin=0.10), "indeterminate")


if __name__ == "__main__":
    unittest.main()
