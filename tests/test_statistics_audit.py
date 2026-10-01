"""Regression checks for calibration, uncertainty, and blend timing bugs."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]


def load(filename):
    spec = importlib.util.spec_from_file_location(filename[:-3], ROOT / "code" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = load("05_analysis.py")
supplement = load("12_supplement.py")


class CalibrationAuditTests(unittest.TestCase):
    def test_identical_prices_cannot_have_isotonic_discrimination(self):
        frame = pd.DataFrame({"p": [0.5] * 6, "y": [0, 0, 0, 1, 1, 1]})
        expected = {"brier": 0.25, "mcb": 0.0, "dsc": 0.0, "unc": 0.25}
        self.assertEqual(analysis.corp_decomposition(frame), expected)
        self.assertEqual(analysis.corp_decomposition(frame.iloc[::-1]), expected)

    def test_tied_corp_matches_independent_isotonic_implementation(self):
        rng = np.random.default_rng(713)
        frame = pd.DataFrame({"p": rng.choice([0.1, 0.3, 0.5, 0.8], 200),
                              "y": rng.integers(0, 2, 200)})
        fitted = IsotonicRegression().fit_transform(frame["p"], frame["y"])
        residual = np.mean((fitted - frame["y"]) ** 2)
        actual = analysis.corp_decomposition(frame)
        self.assertAlmostEqual(actual["mcb"], np.mean((frame["p"] - frame["y"]) ** 2) - residual)
        for seed in range(5):
            shuffled = analysis.corp_decomposition(frame.sample(frac=1, random_state=seed))
            for key in actual:
                self.assertAlmostEqual(shuffled[key], actual[key])
        self.assertGreaterEqual(actual["mcb"], -1e-15)
        self.assertGreaterEqual(actual["dsc"], -1e-15)

    def test_weighted_pav_uses_observation_weights(self):
        fitted = analysis.pav(np.array([0.8, 0.2, 0.9]), np.array([1, 3, 2]))
        np.testing.assert_allclose(fitted, [0.35, 0.35, 0.9])

    def test_miscalibration_bootstrap_includes_price_variation(self):
        # Frequencies have no variation, but the sample mean pricing error does.
        frame = pd.DataFrame({"p": [0.1, 0.4, 0.6, 0.9], "y": [0] * 4,
                              "event_id": list("abcd")})
        bins = analysis.calib_bins(frame, nbins=1, boot=500)
        self.assertEqual(bins.loc[0, "ci_lo"], 0)
        self.assertEqual(bins.loc[0, "ci_hi"], 0)
        self.assertLess(bins.loc[0, "error_ci_lo"], bins.loc[0, "error_ci_hi"])
        # A separate call or intervening analysis cannot alter seeded output.
        pd.testing.assert_frame_equal(bins, analysis.calib_bins(frame, nbins=1, boot=500))

    def test_empty_bins_and_single_event_have_undefined_intervals(self):
        frame = pd.DataFrame({"p": [0.2, 0.3], "y": [0, 1], "event_id": ["a", "a"]})
        result = analysis.calib_bins(frame, nbins=20, boot=10)
        self.assertTrue(result["ci_lo"].isna().all())
        self.assertTrue(result["error_ci_hi"].isna().all())
        self.assertEqual(result["n"].sum(), 2)
        mean, se, n, events = analysis.cluster_mean_se(frame, "p")
        self.assertEqual((mean, n, events), (0.25, 2, 1))
        self.assertTrue(np.isnan(se))

    def test_quantile_groups_keep_equal_values_together(self):
        constant = analysis.quantile_groups(pd.Series([1.0] * 10))
        self.assertEqual(set(constant), {"mid"})
        values = pd.Series([0.0] * 8 + [1.0] * 2 + [np.nan])
        groups = analysis.quantile_groups(values)
        self.assertEqual(groups[values == 0].nunique(), 1)
        self.assertEqual(groups[values == 1].nunique(), 1)
        self.assertTrue(pd.isna(groups.iloc[-1]))

    def test_regressions_return_explicit_status_for_degenerate_subset(self):
        frame = pd.DataFrame({"p": [0.5, 0.5], "y": [0, 1], "event_id": ["a", "b"]})
        actual = analysis.mz_regressions(frame)
        self.assertEqual(actual["status"], "not_estimable")
        self.assertTrue(np.isnan(actual["logodds"]["b"]))

    def test_perfect_separation_is_reported_as_unestimable(self):
        frame = pd.DataFrame({"p": [0.1, 0.2, 0.8, 0.9], "y": [0, 0, 1, 1],
                              "event_id": list("abcd")})
        actual = analysis.mz_regressions(frame)
        self.assertEqual(actual["status"], "not_estimable")

    def test_binned_terms_are_distinguished_from_raw_brier_score(self):
        frame = pd.DataFrame({"p": [0.21, 0.24, 0.71, 0.74], "y": [0, 1, 0, 1],
                              "event_id": list("abcd")})
        actual = analysis.scores(frame)
        self.assertAlmostEqual(actual["brier_binned"], actual["reliability"]
                               - actual["resolution"] + actual["uncertainty"])
        self.assertAlmostEqual(actual["brier"], actual["brier_binned"] + actual["binning_residual"])
        self.assertNotAlmostEqual(actual["binning_residual"], 0)

    def test_endpoint_favorite_is_counted_and_zero_price_is_not_a_trade(self):
        frame = pd.DataFrame({"p": [0.0, 1.0], "y": [0, 1], "event_id": ["a", "b"],
                              "t_res": pd.to_datetime(["2025-01-01", "2025-01-02"], utc=True)})
        buckets = analysis.flb_buckets(frame)
        self.assertEqual(sum(row["n"] for row in buckets.values()), 2)
        self.assertEqual(analysis.backtest(frame, 0)["decile_buy_yes_gross"], [None] * 10)


class BlendTimingAuditTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            "snap_month": ["2024-01", "2024-02", "2024-03", "2024-03", "2024-04"],
            "event_id": ["old", "unresolved", "old", "new", "late"],
            "t_res": ["2024-02-01", "2024-05-01", "2024-02-01", "2024-05-01", "2024-05-01"],
            "p": [0.2, 0.4, 0.7, 0.3, 0.6], "y": [0, 1, 1, 0, 1], "h": [7] * 5,
            "pred_iso": [0.1, 0.5, 0.7, 0.4, 0.6], "pred_logit": [0.1, 0.5, 0.7, 0.4, 0.6],
            "pred_gbm_price": [0.1, 0.5, 0.7, 0.4, 0.6], "pred_gbm_full": [0.1, 0.5, 0.7, 0.4, 0.6],
        })

    def test_tuning_outcomes_are_available_and_evaluation_events_are_disjoint(self):
        frame = self.frame()
        first, second, info = supplement.blend_split(frame)
        self.assertEqual(frame.index[first].tolist(), [0])
        self.assertEqual(frame.index[second].tolist(), [3, 4])
        self.assertEqual(info["cutoff_month"], "2024-03")
        self.assertEqual(info["n_excluded_unresolved_tuning"], 1)
        self.assertEqual(info["n_excluded_event_overlap"], 1)

    def test_unavailable_future_labels_cannot_change_selected_blend_weight(self):
        frame = self.frame()
        before = supplement.analyze_predictions(frame)
        frame.loc[1, "y"] = 0
        after = supplement.analyze_predictions(frame)
        for name in ["logit_honest", "gbm_full_honest"]:
            self.assertEqual(before["blend"][name]["lambda_star"], after["blend"][name]["lambda_star"])
            self.assertEqual(before["blend"][name]["delta_brier"], after["blend"][name]["delta_brier"])

    def test_one_month_split_fails_clearly(self):
        frame = self.frame().assign(snap_month="2024-01")
        with self.assertRaisesRegex(ValueError, "at least two test months"):
            supplement.blend_split(frame)


class PresentationAuditTests(unittest.TestCase):
    def test_tables_render_missing_t_values_baseline_and_strategies(self):
        archived = json.loads((ROOT / "results/analysis.json").read_text())
        archived["horizon"].pop("7", None)
        for item in archived["horizon"].values():
            item["scores"]["bss"] = None
            for bucket in item["flb"].values():
                bucket["t"] = None
        for costs in archived["backtest"].values():
            for side in costs.values():
                for value in side.values():
                    if isinstance(value, dict):
                        value["t"] = None
        archived["backtest_sched_h30"] = {"gross": {}, "net_1c": {}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "code").mkdir()
            (root / "results").mkdir()
            shutil.copyfile(ROOT / "code/07_tables.py", root / "code/07_tables.py")
            (root / "results/analysis.json").write_text(json.dumps(archived))
            result = subprocess.run([sys.executable, str(root / "code/07_tables.py")], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("—", (root / "results/tables.md").read_text())


if __name__ == "__main__":
    unittest.main()
