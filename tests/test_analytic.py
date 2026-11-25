"""Unit tests for Section 2 analytic identities and multiclass reconstruction."""

from __future__ import annotations

import unittest

from chance_is_a_line.reconstruction import j_bounds, j_from_reliance, rair_rsr_from_reliance
from chance_is_a_line.simulation import analytic_over_reliance, constant, discriminating, run


class TestAnalyticClaims(unittest.TestCase):
    def test_null_line_identity(self) -> None:
        for p in (0.0, 0.25, 0.5, 0.75, 1.0):
            for a_a in (0.50, 0.80):
                m = run(constant(p), a_H=0.60, a_A=a_a, n=100_000, seed=0)
                self.assertLess(abs((m["RAIR"] + m["RSR"]) - 1.0), 0.01)
                self.assertLess(abs(m["J"]), 0.01)

    def test_over_reliance_denominators(self) -> None:
        lo_acc = analytic_over_reliance(0.60, 0.50, p_switch=1.0, denom="dis")
        hi_acc = analytic_over_reliance(0.60, 0.95, p_switch=1.0, denom="dis")
        self.assertEqual(round(lo_acc, 3), 0.600)
        self.assertEqual(round(hi_acc, 3), 0.073)

        for a_a in (0.50, 0.70, 0.95):
            self.assertAlmostEqual(analytic_over_reliance(0.60, a_a, p_switch=1.0, denom="ai_wrong"), 0.60, places=6)
            self.assertAlmostEqual(analytic_over_reliance(0.60, a_a, p_switch=1.0, denom="ai_wrong_accept"), 1.00, places=6)

    def test_multiclass_reconstruction_accuracy(self) -> None:
        for k in (2, 3, 4):
            m = run(discriminating(0.70, 0.65), a_H=0.60, a_A=0.67, n=150_000, k=k, seed=7)
            rair_r, rsr_r = rair_rsr_from_reliance(
                m["over_reliance"], m["under_reliance"], m["acc_initial"], 0.67, k=k
            )
            j_r = j_from_reliance(m["over_reliance"], m["under_reliance"], m["acc_initial"], 0.67, k=k)
            self.assertLess(abs(rair_r - m["RAIR"]), 0.01)
            self.assertLess(abs(rsr_r - m["RSR"]), 0.01)
            self.assertLess(abs(j_r - m["J"]), 0.01)

    def test_j_bounds_contain_true_j(self) -> None:
        m = run(discriminating(0.65, 0.70), a_H=0.60, a_A=0.75, n=100_000, k=3, seed=1)
        lo, hi = j_bounds(over=m["over_reliance"], a_H=m["acc_initial"], a_A=0.75, k=3)
        self.assertLessEqual(lo - 0.01, m["J"])
        self.assertLessEqual(m["J"], hi + 0.01)


if __name__ == "__main__":
    unittest.main()
