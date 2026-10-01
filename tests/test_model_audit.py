"""Regression checks for fold eligibility, preprocessing and full-data refitting."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("model_audit", ROOT / "code/09_model.py")
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


class ModelTimingTests(unittest.TestCase):
    def test_training_excludes_postclosure_and_future_snapshots(self):
        frame = pd.DataFrame({
            "snap_ts": [pd.Timestamp(v, tz="UTC").timestamp() for v in
                        ["2024-01-01", "2024-02-01", "2024-01-15", "2024-01-03"]],
            "t_res": pd.to_datetime(["2024-01-20", "2024-01-20", "2024-02-01", "2024-01-03"], utc=True),
        })
        self.assertEqual(model.training_eligible(frame, "2024-02-01").tolist(), [True, False, False, False])
        self.assertEqual(model.eligible_snapshots(frame).index.tolist(), [0, 2])

    def test_early_stopping_retains_fit_event_when_validation_event_is_large(self):
        fit, valid = model.es_split(np.arange(11), ["old"] + ["recent"] * 10, frac=.99)
        self.assertEqual(fit.sum(), 1)
        self.assertEqual(valid.sum(), 10)
        with self.assertRaisesRegex(ValueError, "two nonmissing"):
            model.es_split([1, 2], ["one", "one"])

    def test_text_features_do_not_learn_future_only_words(self):
        burn = pd.DataFrame({"question": ["same forecast question"] * 30 + ["different forecast subject"] * 30})
        frame = pd.concat([burn, pd.DataFrame({"question": ["futureexclusive token"]})], ignore_index=True)
        encoded, cols = model.add_text_features(frame, burn)
        self.assertEqual(len(cols), model.N_SVD)
        np.testing.assert_array_equal(encoded.loc[len(frame)-1, cols].to_numpy(dtype=float), np.zeros(model.N_SVD))
        self.assertGreater(np.linalg.norm(encoded.loc[0, cols].to_numpy(dtype=float)), 0)

    def test_text_free_burn_in_has_finite_zero_features(self):
        burn = pd.DataFrame({"question": [""] * 30})
        frame, cols = model.add_text_features(burn, burn)
        self.assertEqual(frame[cols].shape, (30, model.N_SVD))
        self.assertTrue((frame[cols].values == 0).all())

    def test_selected_tree_count_is_refit_on_every_training_row(self):
        selected, final = MagicMock(), MagicMock()
        selected.best_iteration_ = 7
        X = pd.DataFrame({"p": [.1, .2, .3, .4]})
        y = pd.Series([0, 1, 0, 1])
        with patch.object(model.lgb, "LGBMClassifier", side_effect=[selected, final]) as cls:
            result = model.fit_gbm_model(X, y, np.arange(4), ["a", "b", "c", "d"])
        self.assertIs(result, final)
        self.assertEqual(cls.call_args.kwargs["n_estimators"], 7)
        self.assertIs(final.fit.call_args.args[0], X)
        self.assertIs(final.fit.call_args.args[1], y)
        self.assertLess(len(selected.fit.call_args.args[0]), len(X))

    def test_single_class_probabilities_keep_the_original_outcome_encoding(self):
        X = pd.DataFrame({"p": [.1, .2, .3]})
        for outcome in [0, 1]:
            prediction = model.fit_gbm(X, pd.Series([outcome] * 3), np.arange(3),
                                       ["a", "b", "c"], X)
            np.testing.assert_array_equal(prediction, np.full(3, outcome))

    def test_single_class_fit_partition_uses_all_training_rows(self):
        X = pd.DataFrame({"p": [.1, .2, .3, .9]})
        y = pd.Series([0, 0, 0, 1])
        with patch.object(model.lgb, "LGBMClassifier") as cls:
            model.fit_gbm_model(X, y, np.arange(4), ["a", "b", "c", "d"])
        cls.assert_called_once()
        self.assertIs(cls.return_value.fit.call_args.args[0], X)
        self.assertNotIn("eval_set", cls.return_value.fit.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
