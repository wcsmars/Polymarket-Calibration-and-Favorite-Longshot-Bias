"""Offline checks for research invariants and archived aggregate consistency.

Run from the project root: python -m unittest discover -s tests -v
These small synthetic checks do not rerun or independently validate the study.
"""
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_script(filename):
    spec = importlib.util.spec_from_file_location(filename[:-3], ROOT / "code" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = load_script("05_analysis.py")
features = load_script("08_features.py")
model = load_script("09_model.py")
fetcher = load_script("01_fetch_markets.py")


class SnapshotFeaturesTests(unittest.TestCase):
    def test_future_observations_cannot_change_snapshot_features(self):
        day = features.DAY
        times = np.arange(12, dtype=float) * day
        prices = np.linspace(0.15, 0.75, len(times))
        snapshot = 10 * day
        expected = features.path_features(times[:11], prices[:11], snapshot)
        altered_future = prices.copy()
        altered_future[11:] = 0.001
        self.assertEqual(features.path_features(times, prices, snapshot), expected)
        self.assertEqual(features.path_features(times, altered_future, snapshot), expected)
        self.assertEqual(expected["n_obs_pre"], 11)
        self.assertAlmostEqual(expected["p"], prices[10])

    def test_staleness_boundary_and_insufficient_history(self):
        times = np.array([0.0, float(features.DAY)])
        prices = np.array([0.3, 0.4])
        boundary = times[-1] + features.STALE_TOL
        self.assertIsNotNone(features.path_features(times, prices, boundary))
        self.assertIsNone(features.path_features(times, prices, boundary + 1))
        self.assertIsNone(features.path_features(times, prices, times[0]))
        self.assertIsNone(features.path_features(times, prices, -1))


class EventSplitTests(unittest.TestCase):
    def setUp(self):
        # Deliberately interleave old and recent rows from the same event.
        self.times = np.array([1, 10, 2, 3, 4, 9, 5, 6])
        self.events = np.array(["new", "new", "old", "old", "mid", "mid", "old", "old"])

    def test_validation_keeps_whole_recent_events(self):
        fit, valid = model.es_split(self.times, self.events, frac=0.375)
        self.assertTrue(np.all(fit | valid))
        self.assertFalse(np.any(fit & valid))
        self.assertTrue(set(self.events[fit]).isdisjoint(self.events[valid]))
        self.assertEqual(set(self.events[valid]), {"new", "mid"})
        self.assertGreaterEqual(valid.sum(), int(len(self.events) * 0.375))
        self.assertTrue(fit.any())

    def test_event_assignment_does_not_depend_on_row_order(self):
        _, original = model.es_split(self.times, self.events, frac=0.375)
        order = np.array([4, 0, 7, 2, 6, 1, 5, 3])
        _, permuted = model.es_split(self.times[order], self.events[order], frac=0.375)
        self.assertEqual(set(self.events[original]), set(self.events[order][permuted]))


class ScoringTests(unittest.TestCase):
    def test_scores_match_hand_calculated_binary_example(self):
        frame = pd.DataFrame({
            "p": [0.2, 0.4, 0.7, 0.9], "y": [0, 1, 0, 1],
            "event_id": ["a", "a", "b", "c"],
        })
        actual = analysis.scores(frame)
        self.assertAlmostEqual(actual["brier"], 0.225)
        self.assertAlmostEqual(actual["brier_ref"], 0.25)
        self.assertAlmostEqual(actual["bss"], 0.1)
        self.assertAlmostEqual(actual["log_score"], -math.log(0.8 * 0.4 * 0.3 * 0.9) / 4)
        self.assertEqual(actual["n_events"], 3)

    def test_perfect_forecasts_and_constant_outcome_edge_cases(self):
        frame = pd.DataFrame({"p": [0.0, 1.0], "y": [0, 1], "event_id": ["a", "b"]})
        perfect = analysis.scores(frame)
        self.assertEqual(perfect["brier"], 0)
        self.assertEqual(perfect["bss"], 1)
        self.assertTrue(math.isfinite(perfect["log_score"]))
        constant = analysis.scores(frame.assign(y=0))
        self.assertIsNone(constant["bss"])
        self.assertEqual(constant["uncertainty"], 0)

    def test_corp_identity_for_distinct_forecast_probabilities(self):
        frame = pd.DataFrame({"p": [0.1, 0.3, 0.6, 0.9], "y": [0, 1, 0, 1]})
        actual = analysis.corp_decomposition(frame)
        # The two middle outcomes pool to 0.5 under isotonic recalibration.
        self.assertAlmostEqual(actual["brier"], 0.2175)
        self.assertAlmostEqual(actual["mcb"], 0.0925)
        self.assertAlmostEqual(actual["dsc"], 0.125)
        self.assertAlmostEqual(actual["brier"], actual["mcb"] - actual["dsc"] + actual["unc"])

    def test_cluster_uncertainty_does_not_treat_duplicate_rows_as_new_events(self):
        frame = pd.DataFrame({"event_id": ["a", "a", "b", "b"], "error": [0, 0, 1, 1]})
        mean, se, n, groups = analysis.cluster_mean_se(frame, "error")
        repeated = analysis.cluster_mean_se(pd.concat([frame, frame], ignore_index=True), "error")
        self.assertEqual(mean, 0.5)
        self.assertAlmostEqual(se, math.sqrt(2) / 4)
        self.assertEqual(repeated[:2], (mean, se))
        self.assertEqual(repeated[2:], (2 * n, groups))


class ArchivedAggregateTests(unittest.TestCase):
    def test_calibration_bins_reconcile_to_horizon_summaries(self):
        saved = json.loads((ROOT / "results" / "analysis.json").read_text())
        for horizon, record in saved["horizon"].items():
            with self.subTest(horizon=horizon):
                bins = pd.read_csv(ROOT / "results" / f"calibration_bins_h{horizon}.csv")
                scores = record["scores"]
                self.assertEqual(int(bins["n"].sum()), scores["n"])
                base_rate = (bins["n"] * bins["y_freq"]).sum() / scores["n"]
                self.assertAlmostEqual(base_rate, scores["base_rate"], places=12)
                reliability = (bins["n"] * (bins["p_mean"] - bins["y_freq"]) ** 2).sum() / scores["n"]
                self.assertAlmostEqual(reliability, scores["reliability"], places=12)
                corp = record["corp"]
                self.assertAlmostEqual(scores["brier"], corp["mcb"] - corp["dsc"] + corp["unc"])

    def test_model_horizons_and_score_differences_reconcile(self):
        saved = json.loads((ROOT / "results" / "models" / "model_metrics.json").read_text())
        for anchor, record in saved.items():
            with self.subTest(anchor=anchor):
                horizons = list(record["by_horizon"].values())
                self.assertEqual(sum(item["n"] for item in horizons), record["n"])
                for column, pooled in [
                    ("brier_price", record["brier_price"]),
                    ("brier_gbm_full", record["models"]["gbm_full"]["brier"]),
                ]:
                    weighted = sum(item["n"] * item[column] for item in horizons) / record["n"]
                    self.assertAlmostEqual(weighted, pooled, places=12)
                for name, comparison in record["models"].items():
                    with self.subTest(model=name):
                        delta = record["brier_price"] - comparison["brier"]
                        self.assertAlmostEqual(delta, comparison["delta_brier_vs_price"], places=12)
                        self.assertAlmostEqual(delta, comparison["dm_event"]["mean"], places=12)
                        self.assertEqual(comparison["dm_monthly"]["n_months"], record["n_months"])

    def test_supplement_blend_endpoints_match_saved_model_scores(self):
        location = ROOT / "results" / "models"
        metrics = json.loads((location / "model_metrics.json").read_text())
        supplement = json.loads((location / "supplement.json").read_text())
        for anchor, record in supplement.items():
            for name in ["logit", "gbm_full"]:
                with self.subTest(anchor=anchor, model=name):
                    blend = record["blend"][name]
                    self.assertAlmostEqual(blend["0"], metrics[anchor]["brier_price"], places=12)
                    self.assertAlmostEqual(blend["1.0"], metrics[anchor]["models"][name]["brier"], places=12)


class CollectionTests(unittest.TestCase):
    def test_repeated_request_failures_stop_at_retry_limit(self):
        failure = subprocess.CalledProcessError(22, ["curl"])
        with patch("subprocess.run", side_effect=failure) as run, \
                patch.object(fetcher.time, "sleep") as sleep, \
                patch.object(fetcher.random, "random", return_value=0), \
                patch("builtins.print"):
            with self.assertRaises(RuntimeError):
                fetcher.get({"closed": "true"}, retries=2)
        self.assertEqual(run.call_count, 2)
        sleep.assert_called_once()

    def test_crowded_windows_split_repeatedly_without_losing_markets(self):
        start = datetime(2025, 4, 1, tzinfo=timezone.utc)
        ends = [start + timedelta(days=30) * (i + 0.5) / 1000 for i in range(1000)]
        markets = [{"id": str(i), "endDate": end.isoformat()} for i, end in enumerate(ends)]

        def fake_get(params):
            low = fetcher._utc(params["end_date_min"])
            high = fetcher._utc(params["end_date_max"])
            hits = [m for m, end in zip(markets, ends) if low <= end < high]
            return hits[params["offset"]:params["offset"] + params["limit"]]

        out = io.StringIO()
        # A 200-row cap forces three levels of splitting on 1,000 markets.
        with patch.object(fetcher, "get", side_effect=fake_get), \
                patch.object(fetcher, "MAX_OFFSET", 200), \
                patch("builtins.print"):
            n_new = fetcher.fetch_window("2025-04-01", "2025-05-01", set(), out)
        saved = [json.loads(line)["id"] for line in out.getvalue().splitlines()]
        self.assertEqual(n_new, len(markets))
        self.assertEqual(sorted(saved, key=int), [m["id"] for m in markets])


if __name__ == "__main__":
    unittest.main()
