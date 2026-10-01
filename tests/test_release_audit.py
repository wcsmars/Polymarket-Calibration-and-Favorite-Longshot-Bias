"""Check isolated sample-run validation and portable sample generation."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "sample" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("run_sample")
generator = load("make_sample")
spec = importlib.util.spec_from_file_location("release_data_io", ROOT / "code" / "data_io.py")
data_io = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_io)


class ResultWritingTests(unittest.TestCase):
    def test_results_are_strict_json_with_numpy_scalars_and_missing_estimates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "results.json"
            data_io.write_json(path, {"n": np.int64(3), "valid": np.bool_(True),
                                     "value": np.float64(.25), "values": np.array([np.nan, np.inf, -np.inf]),
                                     np.int64(7): (np.float32(.5), None)})
            def reject(token):
                raise AssertionError(f"nonstandard JSON constant: {token}")
            result = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
            self.assertEqual(result, {"n": 3, "valid": True, "value": .25,
                                      "values": [None, None, None], "7": [.5, None]})
            self.assertIs(type(result["n"]), int)

    def test_serialization_failure_preserves_existing_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            path.write_text('{"saved": true}', encoding="utf-8")
            with self.assertRaises(TypeError):
                data_io.write_json(path, {"unsupported": object()})
            self.assertEqual(path.read_text(encoding="utf-8"), '{"saved": true}')
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_failed_atomic_replace_preserves_previous_result_and_cleans_temporary_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            path.write_text('{"saved": true}', encoding="utf-8")
            def fail_replace(temporary, destination):
                self.assertEqual(Path(destination), path)
                self.assertEqual(json.loads(Path(temporary).read_text(encoding="utf-8")), {"new": 1})
                self.assertEqual(path.read_text(encoding="utf-8"), '{"saved": true}')
                raise OSError("simulated replacement failure")
            with patch.object(data_io.os, "replace", side_effect=fail_replace):
                with self.assertRaises(OSError):
                    data_io.write_json(path, {"new": 1})
            self.assertEqual(path.read_text(encoding="utf-8"), '{"saved": true}')
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_calibration_tables_render_null_regressions(self):
        results = json.loads((ROOT / "results/analysis.json").read_text(encoding="utf-8"))
        for horizon in results["horizon"].values():
            for regression in ("linear", "logodds"):
                horizon["regressions"][regression] = dict.fromkeys(horizon["regressions"][regression])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "code").mkdir()
            shutil.copyfile(ROOT / "code/07_tables.py", root / "code/07_tables.py")
            data_io.write_json(root / "results/analysis.json", results)
            result = subprocess.run([sys.executable, str(root / "code/07_tables.py")],
                                    capture_output=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stderr)
            tables = (root / "results/tables.md").read_text(encoding="utf-8")
            self.assertIn("—", tables)
            self.assertNotIn("nan", tables)


class SampleOperationTests(unittest.TestCase):
    def test_generator_creates_a_new_nested_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "new sample" / "inputs"
            self.assertEqual(generator.write(output), (293, 231))
            for name in runner.INPUTS:
                saved = (ROOT / "sample" / name).read_bytes().replace(b"\r\n", b"\n")
                self.assertEqual((output / name).read_bytes(), saved)

    def test_runner_does_not_overwrite_an_existing_workdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            sentinel = Path(tmp) / "keep.txt"
            sentinel.write_text("saved", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not empty"):
                runner.run(tmp)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "saved")
            self.assertEqual(list(Path(tmp).iterdir()), [sentinel])

    def test_runner_rejects_file_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "file"
            file.touch()
            with self.assertRaisesRegex(ValueError, "not a directory"):
                runner.run(file)

    def test_runner_rejects_unsupported_stages_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "work"
            for stage in ("01_fetch_markets", "09_model", "../outside"):
                with self.subTest(stage=stage), self.assertRaisesRegex(ValueError, "sample stages"):
                    runner.run(work, stages=[stage])
                self.assertFalse(work.exists())

    def test_missing_inputs_fail_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "work"
            with patch.object(runner, "HERE", Path(tmp)):
                with self.assertRaisesRegex(ValueError, "sample input is missing"):
                    runner.run(work)
            self.assertFalse(work.exists())


if __name__ == "__main__":
    unittest.main()
