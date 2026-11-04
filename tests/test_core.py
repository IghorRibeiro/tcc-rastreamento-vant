"""Fast checks of the prediction, matching and statistics code (no data needed)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "etapa3_predicao"))
sys.path.insert(0, str(ROOT / "stats"))

import inference  # noqa: E402
import verify_tcc  # noqa: E402
from trajpred import GroundTruth, aggregate_sequence, evaluate_sequence, pair_errors  # noqa: E402
from trajpred.kalman import corrected_states, forward, single_target_errors  # noqa: E402


def linear_centers(n=80, vx=2.0, vy=-1.0):
    return {f: (100.0 + vx * f, 200.0 + vy * f) for f in range(1, n + 1)}


class KalmanTest(unittest.TestCase):
    def test_constant_velocity_is_extrapolated_with_small_error(self):
        centers = linear_centers()
        errors = single_target_errors(centers, centers, damping=1.0, horizons=(5, 30))
        self.assertLess(np.mean(errors[30]), 5.0)
        self.assertLess(np.mean(errors[5]), np.mean(errors[30]))

    def test_damping_shortens_the_extrapolation(self):
        _, state = list(corrected_states(linear_centers(), 1.0))[-1]
        persistent = forward(state, 30, 1.0)
        damped = forward(state, 30, 0.95)
        self.assertLess(abs(damped[0, 0] - state[0, 0]), abs(persistent[0, 0] - state[0, 0]))

    def test_first_observation_only_initialises(self):
        self.assertEqual(list(corrected_states({7: (1.0, 2.0)})), [])


class MatchingTest(unittest.TestCase):
    def setUp(self):
        rows = []
        for frame in range(1, 61):
            rows.append((frame, 1, 10 + 2 * frame, 20, 12, 10))
            rows.append((frame, 2, 300 - 2 * frame, 150, 12, 10))
        self.gt = pd.DataFrame(rows, columns=["frame", "id", "x", "y", "w", "h"])

    def test_perfect_tracker_has_no_switches_and_zero_error(self):
        metrics, corr = evaluate_sequence(self.gt, self.gt.copy(), "perfect")
        self.assertAlmostEqual(metrics["mota"], 1.0)
        self.assertAlmostEqual(metrics["idf1"], 1.0)
        self.assertEqual(metrics["num_switches"], 0)
        self.assertTrue(metrics["idf1_matches"])
        pairs, counts = pair_errors(self.gt, GroundTruth(self.gt), corr, (5, 10), (1.0,))
        table = aggregate_sequence(pairs, counts)
        self.assertLess(table[table["mode"] == "clear"]["mean_px"].max(), 1.0)
        self.assertEqual(int(table["unassociated"].sum()), 0)

    def test_identity_swap_is_counted(self):
        swapped = self.gt.copy()
        late = swapped["frame"] > 30
        swapped.loc[late, "id"] = swapped.loc[late, "id"].map({1: 2, 2: 1})
        metrics, _ = evaluate_sequence(self.gt, swapped, "swapped")
        self.assertEqual(metrics["num_switches"], 2)
        self.assertLess(metrics["idf1"], 1.0)


class StatisticsTest(unittest.TestCase):
    def test_holm_matches_the_textbook_example(self):
        adjusted = inference.holm([0.01, 0.04, 0.03, 0.005])
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06, 0.02])

    def test_wilcoxon_detects_a_consistent_shift(self):
        diffs = np.arange(1, 21) * 0.1
        self.assertLess(inference.wilcoxon_rb(diffs)["p"], 1e-3)

    def test_slope_from_sufficient_statistics(self):
        x = np.arange(1, 30, dtype=float)
        y = 5.0 + 3.0 * x
        intercept, slope, r2 = inference.ols_from_sums(inference.sufficient_stats(x, y))
        self.assertAlmostEqual(slope, 3.0)
        self.assertAlmostEqual(intercept, 5.0)
        self.assertAlmostEqual(r2, 1.0)


class TccNumbersTest(unittest.TestCase):
    def test_summary_reproduces_the_numbers_printed_in_the_tcc(self):
        summary = pd.read_csv(ROOT / "results" / "summary.csv")
        index = {(r.table, r.row, r.column): r.value for r in summary.itertuples()}
        missing = [key[:3] for key in ((t, r, c, *_) for t, r, c, *_ in verify_tcc.E) if key[:3] not in index]
        self.assertEqual(missing, [])
        for table, row, column, expected, decimals, _flags in verify_tcc.E:
            self.assertAlmostEqual(index[(table, row, column)], expected, delta=0.5 * 10 ** (-decimals) + 1e-9,
                                   msg=f"{table}/{row}/{column}")


if __name__ == "__main__":
    unittest.main()
