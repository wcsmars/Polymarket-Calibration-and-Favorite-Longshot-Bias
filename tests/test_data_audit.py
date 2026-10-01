"""Offline regressions for collection integrity and forecast-time construction."""
import importlib.util
from http.client import IncompleteRead
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name[:-3], ROOT / "code" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


metadata = load("01_fetch_markets.py")
prices = load("02_fetch_prices.py")
sample = load("03_build_sample.py")
panel = load("04_build_panel.py")
features = load("08_features.py")
data_io = load("data_io.py")
DAY = 86400


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class RawIntegrityTests(unittest.TestCase):
    def test_normalization_sorts_and_coalesces_identical_quotes(self):
        result = data_io.normalize_history([[30, .7], [10, .2], [30, .7], [20, .5]])
        np.testing.assert_array_equal(result, [[10, .2], [20, .5], [30, .7]])
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            data_io.normalize_history([[10, .2], [10, .3]])

    def test_invalid_probabilities_are_rejected_on_collection_and_flagged_in_saved_inputs(self):
        for point in [[10, 1.0025], [10, np.nan], [10, -0.1], [-1, .5], [1.5, .5]]:
            with self.subTest(point=point), self.assertRaises(ValueError):
                data_io.normalize_history([point])
        with self.assertWarnsRegex(RuntimeWarning, "Discarded 1"):
            result = data_io.normalize_history([[10, .2], [20, 1.0025]], strict=False)
        np.testing.assert_array_equal(result, [[10, .2]])

    def test_interrupted_tail_is_backed_up_without_losing_complete_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            fragment = b'{"id": "second", "history": ['
            path.write_bytes(b'{"id": "first"}\n' + fragment)
            with self.assertWarnsRegex(RuntimeWarning, "Recovered interrupted"):
                data_io.prepare_append(path)
            self.assertEqual(list(data_io.read_jsonl(path)), [{"id": "first"}])
            self.assertEqual(path.with_name(path.name + ".partial").read_bytes(), fragment)
            path.write_bytes(b'{"id": "first"}')
            data_io.prepare_append(path)
            self.assertEqual(path.read_bytes(), b'{"id": "first"}\n')

    def test_complete_corrupt_records_are_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text('{"id": "first"}\ninvalid\n')
            data_io.prepare_append(path)
            with self.assertRaisesRegex(ValueError, ":2:"):
                list(data_io.read_jsonl(path))

    def test_failed_retry_does_not_erase_success_and_counts_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.jsonl"
            write_jsonl(path, [{"id": "a", "n": -1, "history": []},
                              {"id": "a", "n": 1, "history": [[10, .3]]},
                              {"id": "a", "n": -1, "history": []}])
            np.testing.assert_array_equal(data_io.load_histories(path)["a"], [[10, .3]])
            write_jsonl(path, [{"id": "a", "n": 2, "history": [[10, .3]]}])
            with self.assertRaisesRegex(ValueError, "count mismatch"):
                data_io.load_histories(path)


class CollectionRegressionTests(unittest.TestCase):
    def test_price_error_payload_is_failure_not_an_empty_history(self):
        with patch.object(prices.urllib.request, "urlopen", return_value=io.BytesIO(b'{"error":"busy"}')):
            self.assertIsNone(prices.fetch_history("123", retries=1))
        with patch.object(prices.urllib.request, "urlopen", return_value=io.BytesIO(b'{"history":[]}')):
            self.assertEqual(prices.fetch_history("123", retries=1), [])

    def test_malformed_quote_is_retried_without_worker_failure(self):
        responses = [io.BytesIO(b'{"history":[{"t":10,"p":2}]}'),
                     io.BytesIO(b'{"history":[{"t":20,"p":0.123456},{"t":10,"p":0.5}]}')]
        with patch.object(prices.urllib.request, "urlopen", side_effect=responses), \
                patch.object(prices.time, "sleep"):
            self.assertEqual(prices.fetch_history("123", retries=2),
                             [{"t": 10, "p": .5}, {"t": 20, "p": .123456}])

    def test_interrupted_http_body_is_retried(self):
        with patch.object(prices.urllib.request, "urlopen", side_effect=[
                IncompleteRead(b"{", 100), io.BytesIO(b'{"history":[{"t":10,"p":0.2}]}')]) as request, \
                patch.object(prices.time, "sleep"):
            self.assertEqual(prices.fetch_history("123", retries=2), [{"t": 10, "p": .2}])
        self.assertEqual(request.call_count, 2)

    def test_price_resume_retries_failed_ids_and_keeps_successes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path, out = root / "sample.csv", root / "histories.jsonl"
            pd.DataFrame({"id": ["a", "b"], "yes_token": ["11", "22"]}).to_csv(sample_path, index=False)
            write_jsonl(out, [{"id": "a", "n": 0, "history": []},
                             {"id": "b", "n": -1, "history": []}])
            with patch.object(prices, "SAMPLE", str(sample_path)), patch.object(prices, "OUT", str(out)), \
                    patch.object(prices, "fetch_history", return_value=[{"t": 10, "p": .2}]) as fetch, \
                    patch("builtins.print"):
                prices.main()
                prices.main()
            fetch.assert_called_once_with("22")
            self.assertEqual(set(data_io.load_histories(out)), {"a", "b"})

    def test_price_fetch_command_reports_persistent_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path, out = root / "sample.csv", root / "histories.jsonl"
            pd.DataFrame({"id": ["a"], "yes_token": ["11"]}).to_csv(sample_path, index=False)
            with patch.object(prices, "SAMPLE", str(sample_path)), patch.object(prices, "OUT", str(out)), \
                    patch.object(prices, "fetch_history", return_value=None), patch("builtins.print"):
                with self.assertRaisesRegex(RuntimeError, "rerun to retry"):
                    prices.main()
            self.assertEqual(list(data_io.read_jsonl(out)), [{"id": "a", "n": -1, "history": []}])

    def test_metadata_offset_rejection_splits_earlier_than_configured_cap(self):
        seen_calls = []
        def get(params):
            seen_calls.append(params)
            low, high = metadata._utc(params["end_date_min"]), metadata._utc(params["end_date_max"])
            if (high - low).days > 1:
                if params["offset"]:
                    raise metadata.PaginationLimitError("HTTP 422")
                return [{"id": "a"}, {"id": "b"}]
            return [{"id": "a"}] if low.day == 1 else [{"id": "b"}]
        output = io.StringIO()
        with patch.object(metadata, "get", side_effect=get), patch.object(metadata, "PAGE", 2), \
                patch("builtins.print"):
            count = metadata.fetch_window("2025-01-01", "2025-01-03", set(), output)
        self.assertEqual(count, 2)
        self.assertEqual(len(output.getvalue().splitlines()), 2)
        self.assertTrue(all(call["end_date_min"].endswith("Z") for call in seen_calls))
        self.assertTrue(all(call["order"] == "id" for call in seen_calls))

    def test_metadata_http_status_and_shape_are_validated(self):
        response = SimpleNamespace(stdout=b'{"error":"offset too large"}\n422')
        with patch("subprocess.run", return_value=response):
            with self.assertRaises(metadata.PaginationLimitError):
                metadata.get({"offset": 100})
        for payload in [b'{}\n200', b'[{"question":"missing id"}]\n200', b'error\n503']:
            with self.subTest(payload=payload), patch("subprocess.run", return_value=SimpleNamespace(stdout=payload)):
                with self.assertRaises(RuntimeError):
                    metadata.get({}, retries=1)

    def test_missing_metadata_file_invalidates_old_completed_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out, checkpoint = root / "metadata.jsonl", root / "checkpoint.json"
            checkpoint.write_text('["2025-01-01|2025-02-01"]')
            with patch.object(metadata, "OUT", str(out)), patch.object(metadata, "CHECKPOINT", str(checkpoint)), \
                    patch.object(metadata, "month_windows", return_value=[("2025-01-01", "2025-02-01")]), \
                    patch.object(metadata, "get", return_value=[{"id": 123}]) as get, patch("builtins.print"):
                metadata.main()
            get.assert_called_once()
            self.assertEqual(list(data_io.read_jsonl(out))[0]["id"], "123")
            self.assertEqual(json.loads(checkpoint.read_text()), ["2025-01-01|2025-02-01"])


class ConstructionRegressionTests(unittest.TestCase):
    def test_native_lists_reversed_outcomes_and_boolean_strings(self):
        base = {"id": "1", "closed": True, "outcomes": ["No", "Yes"],
                "outcomePrices": [0, 1], "clobTokenIds": ["no", "yes"],
                "enableOrderBook": "true", "closedTime": "2025-02-01T00:00:00Z",
                "createdAt": "2025-01-01T00:00:00Z", "endDate": "2025-02-01T00:00:00Z",
                "volumeNum": 1000, "eventNegRisk": ["false"], "negRisk": "false"}
        records = [base, {**base, "id": "2", "closed": False},
                   {**base, "id": "3", "enableOrderBook": "false"},
                   {**base, "id": "4", "volumeNum": "inf"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, out = root / "raw.jsonl", root / "processed" / "sample.csv"
            write_jsonl(raw, records)
            with patch.object(sample, "RAW", str(raw)), patch.object(sample, "OUT", str(out)), \
                    patch.object(sample, "ROOT", root), patch("builtins.print"):
                sample.main()
            result = pd.read_csv(out, dtype={"id": str})
            self.assertEqual(result.id.tolist(), ["1"])
            self.assertEqual(result.yes_token.tolist(), ["yes"])
            self.assertEqual(result.y.tolist(), [1])
            self.assertFalse(result.event_neg_risk.any())
            self.assertFalse(result.negRisk.any())

    def test_proposed_disputed_and_other_nonfinal_statuses_are_excluded(self):
        base = {"id": "resolved", "closed": True, "outcomes": ["Yes", "No"],
                "outcomePrices": [1, 0], "clobTokenIds": ["yes", "no"],
                "enableOrderBook": True, "closedTime": "2025-02-01T00:00:00Z",
                "createdAt": "2025-01-01T00:00:00Z", "endDate": "2025-02-01T00:00:00Z",
                "volumeNum": 1000}
        records = [{**base, "id": name, "umaResolutionStatus": status} for name, status in [
            ("resolved", "resolved"), ("legacy", None), ("proposed", "proposed"),
            ("disputed", "disputed"), ("unresolved", "unresolved"), ("invalid", 7),
        ]]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, out = root / "raw.jsonl", root / "processed" / "sample.csv"
            write_jsonl(raw, records)
            with patch.object(sample, "RAW", str(raw)), patch.object(sample, "OUT", str(out)), \
                    patch.object(sample, "ROOT", root), patch("builtins.print"):
                sample.main()
            result = pd.read_csv(out, dtype={"id": str})
            self.assertEqual(set(result.id), {"resolved", "legacy"})
            counts = json.loads((root / "results" / "sample_construction.json").read_text())
            self.assertEqual(counts["binary_yes_no"], 6)
            self.assertEqual(counts["clean_resolution"], 2)

    def test_snapshots_exclude_postclosure_and_precreation_quotes_and_future_siblings(self):
        base = pd.Timestamp("2025-01-01T00:00:00Z")
        def date(days):
            return base + pd.Timedelta(days=days)
        rows = []
        for identifier, created, closed, end in [("a", 0, 10, 12), ("b", 9.5, 13, 13)]:
            rows.append({"id": identifier, "yes_token": identifier, "event_id": "event",
                         "event_neg_risk": False, "negRisk": False, "y": 1, "cat": "other",
                         "question": "Will this occur?", "volumeNum": 1000,
                         "t_created": date(created), "t_res": date(closed), "t_end": date(end)})
        histories = []
        for identifier, start in [("a", -1), ("b", 9)]:
            history = [[date(day).timestamp(), .3 + day * .01] for day in range(start, 14)]
            histories.append({"id": identifier, "n": len(history), "history": history})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = root / "data" / "processed"
            processed.mkdir(parents=True)
            pd.DataFrame(rows).to_csv(processed / "sample_markets.csv", index=False)
            write_jsonl(root / "data" / "raw" / "price_histories.jsonl", histories)
            with patch.object(panel, "ROOT", root), patch.object(panel, "HORIZONS", [1, 3, 5]), \
                    patch.object(features, "ROOT", root), patch.object(features, "HORIZONS", [1, 3, 5]), \
                    patch("builtins.print"):
                panel.main()
                features.main()
            market = pd.read_csv(processed / "market_level.csv").set_index("id")
            self.assertTrue(pd.isna(market.at["a", "p_sched_1"]))  # tau=11, close=10
            self.assertTrue(pd.isna(market.at["b", "p_sched_5"]))  # tau=8, created=9.5
            dataset = pd.read_parquet(processed / "ml_dataset.parquet")
            for row in dataset.itertuples():
                original = next(r for r in rows if r["id"] == row.id)
                self.assertGreaterEqual(row.snap_ts, original["t_created"].timestamp())
                self.assertLess(row.snap_ts, original["t_res"].timestamp())
                self.assertEqual(row.event_n_markets, 1 if row.snap_ts < date(9.5).timestamp() else 2)
                self.assertGreaterEqual(row.p_start, .3 if row.id == "a" else .4)


    def test_invalid_scheduled_lifetime_is_missing_while_valid_late_closure_is_preserved(self):
        base = pd.Timestamp("2025-01-01T00:00:00Z")
        rows = []
        histories = []
        for identifier, schedule_days in [("before", -1), ("equal", 0), ("short", .05), ("late", 2)]:
            rows.append({"id": identifier, "yes_token": identifier, "event_id": identifier,
                         "event_neg_risk": False, "negRisk": False, "y": 1, "cat": "other",
                         "question": "Will this occur?", "volumeNum": 1000,
                         "t_created": base, "t_res": base + pd.Timedelta(days=10),
                         "t_end": base + pd.Timedelta(days=schedule_days)})
            history = [[(base + pd.Timedelta(days=day)).timestamp(), .3] for day in range(10)]
            histories.append({"id": identifier, "n": len(history), "history": history})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = root / "data" / "processed"
            processed.mkdir(parents=True)
            pd.DataFrame(rows).to_csv(processed / "sample_markets.csv", index=False)
            write_jsonl(root / "data" / "raw" / "price_histories.jsonl", histories)
            with patch.object(features, "ROOT", root), patch.object(features, "HORIZONS", [1]), \
                    patch("builtins.print"):
                features.main()
            dataset = pd.read_parquet(processed / "ml_dataset.parquet")
            resolved = dataset[dataset.anchor == "res"].set_index("id")
            self.assertEqual(set(resolved.index), {"before", "equal", "short", "late"})
            for identifier in ["before", "equal"]:
                self.assertTrue(pd.isna(resolved.at[identifier, "sched_life_days"]))
                self.assertTrue(pd.isna(resolved.at[identifier, "frac_life"]))
            self.assertAlmostEqual(resolved.at["short", "frac_life"], 9 / .05)
            self.assertAlmostEqual(resolved.at["late", "frac_life"], 9 / 2)


if __name__ == "__main__":
    unittest.main()
