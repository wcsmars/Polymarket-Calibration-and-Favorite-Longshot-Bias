"""Regression checks for fold eligibility, preprocessing and full-data refitting."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("model_audit", ROOT / "code/09_model.py")
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
spec = importlib.util.spec_from_file_location("interpret_audit", ROOT / "code/10_interpret.py")
interpret = importlib.util.module_from_spec(spec)
spec.loader.exec_module(interpret)


def walk_forward_frame(months=8, per_month=60, seed=3):
    """Small synthetic model dataset whose scheduled snapshots tie heavily in time."""
    rng = np.random.default_rng(seed)
    questions = ["Will the price close above the target", "Will the candidate win the vote",
                 "Will the team win the final match"]
    rows = []
    for month in range(months):
        start = pd.Timestamp("2024-01-01", tz="UTC") + pd.DateOffset(months=month)
        for k in range(per_month):
            # Three scheduled ends a month, so many rows share each snapshot time.
            end = start + pd.Timedelta(days=int(rng.choice([8, 15, 22])))
            truth = rng.uniform(.05, .95)
            y = int(rng.random() < truth)
            for h in (1, 7):
                p = float(np.clip(truth + rng.normal(0, .15), .01, .99))
                rows.append({"anchor": "sched", "id": f"{month}-{k:02d}", "h": h,
                             "event_id": f"{month}-{k // 3}", "cat": ["crypto", "politics", "sports"][k % 3],
                             "question": questions[k % 3], "y": y, "p": p, "logit_p": np.log(p / (1 - p)),
                             "snap_ts": (end - pd.Timedelta(days=h)).timestamp(), "t_res": end,
                             **{name: rng.normal() for name in model.PATH_FEATS + model.STRUCT_FEATS}})
    return pd.DataFrame(rows)


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
        # Future words need min_df support, or a fit on every row would drop them too.
        future = pd.DataFrame({"question": ["futureexclusive token"] * 30 + ["same subject futureexclusive"] * 30})
        frame = pd.concat([burn, future], ignore_index=True)
        encoded, cols = model.add_text_features(frame, burn)
        alone, _ = model.add_text_features(burn, burn)
        self.assertEqual(len(cols), model.N_SVD)
        np.testing.assert_array_equal(encoded.loc[len(burn):len(burn) + 29, cols].to_numpy(dtype=float),
                                      np.zeros((30, model.N_SVD)))
        # Later rows must not move the vocabulary, IDF weights or SVD basis.
        np.testing.assert_allclose(encoded.loc[:len(burn) - 1, cols].to_numpy(dtype=float),
                                   alone[cols].to_numpy(dtype=float), rtol=0, atol=1e-12)
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


class WalkForwardTests(unittest.TestCase):
    def predictions(self, frame):
        # One thread and a fixed histogram layout keep LightGBM's own arithmetic
        # reproducible, so a difference can only come from the rows a fold sees.
        with tempfile.TemporaryDirectory() as tmp, patch.object(model, "ROOT", Path(tmp)), \
                patch.dict(model.LGB_PARAMS, n_jobs=1, force_col_wise=True), patch("builtins.print"):
            model.run_anchor(frame, "sched")
            saved = pd.read_parquet(Path(tmp) / "results/models/predictions_sched.parquet")
        return saved.sort_values(["id", "h"]).set_index(["id", "h"])

    def test_fold_predictions_ignore_later_months_and_input_row_order(self):
        # Bagging samples rows by position, so tied snapshot times need a total order.
        frame = walk_forward_frame()
        full = self.predictions(frame)
        first = full["snap_month"].min()
        self.assertGreater(full["snap_month"].nunique(), 1)
        later = pd.Timestamp(first + "-01", tz="UTC") + pd.DateOffset(months=1)
        alone = self.predictions(frame[frame["snap_ts"] < later.timestamp()])
        pd.testing.assert_frame_equal(alone, full[full["snap_month"] == first], check_exact=True)
        shuffled = self.predictions(frame.sample(frac=1, random_state=7))
        pd.testing.assert_frame_equal(shuffled, full, check_exact=True)



class InterpretationTests(unittest.TestCase):
    def interpretation(self, dataset, predictions):
        with tempfile.TemporaryDirectory() as tmp, patch.object(interpret, "ROOT", Path(tmp)), \
                patch.dict(interpret.m09.LGB_PARAMS, n_jobs=1, force_col_wise=True), patch("builtins.print"):
            for folder, name, frame in [("data/processed", "ml_dataset", dataset),
                                        ("results/models", "predictions_sched", predictions)]:
                (Path(tmp) / folder).mkdir(parents=True)
                frame.to_parquet(Path(tmp) / folder / f"{name}.parquet")
            interpret.main()
            return (Path(tmp) / "results/models/interpretation.json").read_text(encoding="utf-8")

    def test_interpretation_model_ignores_input_row_order(self):
        # Fourteen months: the model fits on 2024 and is examined on 2025.
        frame = walk_forward_frame(months=14)
        frame["snap_month"] = pd.to_datetime(frame["snap_ts"], unit="s", utc=True).dt.strftime("%Y-%m")
        noise = np.random.default_rng(5).normal(0, .05, len(frame))
        predictions = frame.assign(pred_gbm_full=np.clip(frame["p"] + noise, .01, .99))
        self.assertEqual(self.interpretation(frame.sample(frac=1, random_state=7), predictions),
                         self.interpretation(frame, predictions))
if __name__ == "__main__":
    unittest.main()
