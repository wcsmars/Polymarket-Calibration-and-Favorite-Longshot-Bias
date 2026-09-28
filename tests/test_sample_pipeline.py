"""Smoke test: run the early pipeline stages on the bundled synthetic sample.

sample/run_sample.py copies code/ and the sample inputs into a temporary project
root and runs 03, 04, 05, 07 and 08 there. These checks cover the
sample-construction counts, the no-look-ahead timing rules of panel prices and
feature snapshots, and the reconciliation of the analysis outputs. They also
confirm that the archived results/, figures/ and any local data/ in this
folder are left untouched. The synthetic numbers are not research results.

Run from the project root: python -m unittest discover -s tests -v
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DAY = 86400
STALE_TOL = 1.5 * DAY
HORIZONS = [1, 3, 7, 14, 30, 60, 90]


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load(ROOT / "sample" / "run_sample.py")
generator = load(ROOT / "sample" / "make_sample.py")


def snapshot():
    """Size and modification time of every archived output and local input."""
    state = {}
    for folder in ("results", "figures", "data"):
        base = ROOT / folder
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and not path.name.startswith("."):
                stat = path.stat()
                state[path.relative_to(ROOT).as_posix()] = (stat.st_size, stat.st_mtime_ns)
    return state


def clip(p):
    return float(np.clip(p, 0.001, 0.999))


class SamplePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = snapshot()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.work = Path(cls.tmp.name) / "project"
        runner.run(cls.work)
        cls.after = snapshot()
        processed = cls.work / "data" / "processed"
        cls.sample = pd.read_csv(processed / "sample_markets.csv",
                                 dtype={"id": str, "event_id": str, "yes_token": str})
        for column in ["t_res", "t_end", "t_created"]:
            cls.sample[column] = pd.to_datetime(cls.sample[column], utc=True, format="mixed")
        cls.panel = pd.read_csv(processed / "panel.csv", dtype={"id": str, "event_id": str})
        cls.market_level = pd.read_csv(processed / "market_level.csv", dtype={"id": str})
        cls.features = pd.read_parquet(processed / "ml_dataset.parquet")
        cls.histories = {}
        with open(ROOT / "sample" / "price_histories.jsonl") as f:
            for line in f:
                record = json.loads(line)
                cls.histories[record["id"]] = record

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def history(self, market_id):
        return np.array(self.histories[market_id]["history"], dtype=float)

    def test_archived_outputs_and_local_inputs_are_untouched(self):
        self.assertIn("results/analysis.json", self.before)
        self.assertEqual(self.after, self.before)

    def test_each_inclusion_filter_removes_its_planted_rows(self):
        counts = json.loads((self.work / "results" / "sample_construction.json").read_text())
        self.assertEqual(counts, {
            "all_resolved_markets": 293, "binary_yes_no": 283, "clean_resolution": 275,
            "clob_market": 269, "has_resolution_time": 266, "end_date_cutoff": 254,
            "volume_filter": 239, "lifetime_filter": 231,
        })
        self.assertEqual(len(self.sample), counts["lifetime_filter"])
        self.assertTrue(self.sample["volumeNum"].ge(1000).all())
        self.assertTrue(self.sample["t_end"].le(pd.Timestamp("2025-07-31 23:59:59+00:00")).all())
        self.assertTrue(self.sample["y"].isin([0, 1]).all())

    def test_price_histories_cover_exactly_the_sampled_markets(self):
        # 02 fetches one history per sampled market, including failed requests.
        self.assertEqual(set(self.histories), set(self.sample["id"]))
        unusable = {i for i, record in self.histories.items() if record["n"] < 2}
        self.assertEqual(len(unusable), 7)
        self.assertEqual(set(self.market_level["id"]), set(self.histories) - unusable)

    def test_panel_prices_are_the_latest_fresh_quote_before_each_horizon(self):
        closes = self.sample.set_index("id")["t_res"]
        expected, stale = {}, 0
        for market_id in self.market_level["id"]:
            history = self.history(market_id)
            for h in HORIZONS:
                target = closes[market_id].timestamp() - h * DAY
                seen = history[history[:, 0] <= target]
                if len(seen) and target - seen[-1, 0] <= STALE_TOL:
                    expected[(market_id, h)] = clip(seen[-1, 1])
                elif len(seen):
                    stale += 1
        actual = {(row.id, row.h): row.p for row in self.panel.itertuples()}
        self.assertEqual(actual.keys(), expected.keys())
        for key, price in expected.items():
            self.assertAlmostEqual(actual[key], price, places=12, msg=key)
        # Gaps in the sample histories exercise the staleness limit.
        self.assertGreater(stale, 0)

    def test_analysis_reconciles_for_horizons_with_enough_observations(self):
        saved = json.loads((self.work / "results" / "analysis.json").read_text())
        counts = self.panel.groupby("h").size()
        analysed = {int(h) for h in saved["horizon"]}
        self.assertEqual(analysed, {h for h, n in counts.items() if n >= 200})
        self.assertEqual(analysed, {1, 3, 7, 14})
        for h in HORIZONS:
            path = self.work / "results" / f"calibration_bins_h{h}.csv"
            with self.subTest(horizon=h):
                self.assertEqual(path.exists(), h in analysed)
                if h not in analysed:
                    continue
                record = saved["horizon"][str(h)]
                scores = record["scores"]
                bins = pd.read_csv(path)
                self.assertEqual(scores["n"], counts[h])
                self.assertEqual(int(bins["n"].sum()), scores["n"])
                base_rate = (bins["n"] * bins["y_freq"]).sum() / scores["n"]
                self.assertAlmostEqual(base_rate, scores["base_rate"], places=12)
                corp = record["corp"]
                self.assertAlmostEqual(scores["brier"], corp["mcb"] - corp["dsc"] + corp["unc"])
                slope = record["regressions"]["logodds"]
                self.assertTrue(np.isfinite(slope["b"]) and slope["se_b"] > 0)
        self.assertEqual(set(saved["backtest"]), {"h7", "h30"})
        tables = (self.work / "results" / "tables.md").read_text(encoding="utf-8")
        self.assertIn(f"| 7 | {counts[7]} |", tables)

    def test_feature_snapshots_use_only_quotes_at_or_before_the_snapshot(self):
        markets = self.sample.set_index("id")
        self.assertEqual(set(self.features["anchor"]), {"sched", "res"})
        for row in self.features.itertuples():
            anchor = markets.at[row.id, "t_end" if row.anchor == "sched" else "t_res"]
            self.assertAlmostEqual(row.snap_ts, anchor.timestamp() - row.h * DAY)
            history = self.history(row.id)
            self.assertEqual(row.n_obs_pre, int((history[:, 0] <= row.snap_ts).sum()))
            last = history[row.n_obs_pre - 1]
            self.assertLessEqual(row.snap_ts - last[0], STALE_TOL)
            self.assertAlmostEqual(row.p, clip(last[1]))


class SampleFilesTests(unittest.TestCase):
    def test_committed_sample_matches_its_generator(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator.write(tmp)
            for name in runner.INPUTS:
                with self.subTest(file=name):
                    fresh = (Path(tmp) / name).read_bytes()
                    saved = (ROOT / "sample" / name).read_bytes().replace(b"\r\n", b"\n")
                    self.assertTrue(fresh == saved,
                                    f"sample/{name} differs; rerun python sample/make_sample.py")

    def test_sample_stays_small(self):
        size = sum((ROOT / "sample" / name).stat().st_size for name in runner.INPUTS)
        self.assertLess(size, 500_000)

    def test_runner_refuses_the_project_root(self):
        with self.assertRaises(ValueError):
            runner.run(ROOT)


if __name__ == "__main__":
    unittest.main()
