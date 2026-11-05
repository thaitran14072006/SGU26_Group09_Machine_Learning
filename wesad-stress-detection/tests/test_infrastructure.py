"""Kiểm tra cửa sổ, bảng đặc trưng, chia tập theo người và resume."""

from contextlib import redirect_stdout
import io
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wesad.data import load_feature_table, validate_features
from wesad.evaluation import run_experiment
from wesad.extraction import run_extraction
from wesad.models import build_pipeline
from wesad.preprocessing import iter_windows
from wesad.utils import read_config, read_json, write_csv, write_json


def feature_frame(subjects=("S2", "S3", "S4", "S5")):
    rows = []
    for j, subject in enumerate(subjects):
        for i in range(12):
            label = i % 3 + 1
            rows.append({"subject_id": subject, "window_id": f"{subject}_{i}",
                         "start_s": i * 5, "end_s": i * 5 + 60,
                         "label_original": label, "quality_flag": "test_fixture",
                         "signal": j * 100.0 + label, "missing": np.nan if i % 2 else float(label)})
    return pd.DataFrame(rows)


class WindowTests(unittest.TestCase):
    def windows(self, labels, **kwargs):
        return list(iter_windows({"x": np.arange(len(labels), dtype=float)}, np.array(labels),
                                 sampling_rate_hz=1, window_seconds=10, step_seconds=10, **kwargs))

    def test_majority_includes_exact_threshold_and_invalid_labels_in_denominator(self):
        accepted = self.windows([1] * 9 + [0], label_rule="majority")
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0].label_fraction, .9)
        self.assertEqual(len(self.windows([1] * 8 + [0] * 2, label_rule="majority")), 0)

    def test_invalid_dominant_label_and_homogeneous_boundary(self):
        self.assertFalse(self.windows([4] * 10, label_rule="majority"))
        self.assertFalse(self.windows([1] * 9 + [2], label_rule="homogeneous"))
        self.assertEqual(len(self.windows([1] * 10)), 1)

    def test_no_concatenation_and_discard_short_tail(self):
        labels = [1] * 10 + [0] * 10 + [2] * 10 + [3] * 4
        windows = self.windows(labels)
        self.assertEqual([x.start_s for x in windows], [0, 20])

    def test_nan_and_length_mismatch(self):
        with self.assertRaises(ValueError):
            list(iter_windows({"x": np.zeros(9)}, np.ones(10)))
        self.assertFalse(list(iter_windows({"x": np.full(10, np.nan)}, np.ones(10),
                                          sampling_rate_hz=1, window_seconds=10)))


class FeatureTests(unittest.TestCase):
    def test_metadata_never_features_and_missing_features_allowed(self):
        frame = feature_frame()
        self.assertEqual(validate_features(frame), ["signal", "missing"])
        with self.assertRaises(ValueError):
            validate_features(frame, ["signal", "subject_id"])

    def test_duplicates_inf_bad_labels_and_times(self):
        for column, value in [("signal", np.inf), ("label_original", 4), ("end_s", -1)]:
            frame = feature_frame()
            frame.loc[0, column] = value
            with self.assertRaises(ValueError):
                validate_features(frame)
        with self.assertRaises(ValueError):
            validate_features(pd.concat([feature_frame(), feature_frame().iloc[:1]]))

    def test_schema_orders_features(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "features.csv"
            write_csv(path, feature_frame())
            write_json(path.parent / "feature_schema.json", {"feature_columns": ["missing", "signal"]})
            _, columns = load_feature_table(path)
            self.assertEqual(columns, ["missing", "signal"])

    def test_local_config_cannot_override_method(self):
        with tempfile.TemporaryDirectory() as folder:
            config, local = Path(folder) / "c.yaml", Path(folder) / "l.yaml"
            config.write_text("label_rule: homogeneous\n")
            local.write_text("label_rule: majority\n")
            with self.assertRaises(ValueError):
                read_config(config, local)


class ExtractionTests(unittest.TestCase):
    def test_resume_verifies_csv_and_processing_identity(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            source = root / "S2" / "S2.pkl"
            source.parent.mkdir()
            values = np.arange(42000, dtype=float)
            with source.open("wb") as stream:
                pickle.dump({"label": np.ones(42000, dtype=int),
                             "signal": {"chest": {"ECG": values}}}, stream)
            config = {"data_root": str(root), "output_dir": str(root / "features"), "subjects": ["S2"],
                      "device": "chest", "sensors": ["ECG"], "sampling_rate_hz": 700,
                      "labels": [1, 2, 3], "window_seconds": 60, "step_seconds": 5,
                      "label_rule": "majority", "feature_version": "test"}
            run_extraction(config)
            output = root / "features" / "S2.csv"
            before = output.stat().st_mtime_ns
            with patch("wesad.extraction.process_subject", side_effect=AssertionError("must skip")):
                run_extraction(config, resume=True)
            self.assertEqual(output.stat().st_mtime_ns, before)
            output.write_text("broken\n")
            run_extraction(config, resume=True)
            self.assertEqual(len(pd.read_csv(output)), 1)
            with self.assertRaises(ValueError):
                run_extraction({**config, "step_seconds": 10}, resume=True)
            self.assertFalse(list((root / "features").glob("*.tmp")))


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.frame = feature_frame()
        self.path = self.root / "features.csv"
        write_csv(self.path, self.frame)
        self.config = {"features": str(self.path), "output_dir": str(self.root / "run"),
                       "models": ["dummy", "lr"], "tasks": ["multiclass", "binary"], "seed": 42}

    def tearDown(self):
        self.directory.cleanup()

    def run_check(self, config=None, **kwargs):
        with redirect_stdout(io.StringIO()):
            return run_experiment(config or self.config, **kwargs)

    def test_complete_requires_15_and_partial_predictions_are_unique(self):
        with self.assertRaises(ValueError):
            self.run_check()
        result = self.run_check(allow_partial=True)
        self.assertEqual(result["status"], "partial")
        predictions = pd.read_csv(self.root / "run/predictions.csv")
        self.assertEqual(len(predictions), 4 * len(self.frame))
        self.assertFalse(predictions.duplicated(["task", "model", "subject_id", "window_id"]).any())
        for split in read_json(self.root / "run/splits.json").values():
            self.assertFalse(set(split["outer_train_subjects"]) & set(split["outer_test_subjects"]))

    def test_full_15_subjects_completes_and_quick_stays_partial(self):
        from wesad.data import SUBJECTS
        write_csv(self.path, feature_frame(SUBJECTS))
        config = {**self.config, "models": ["dummy"], "tasks": ["multiclass"]}
        self.assertEqual(self.run_check(config, outer_subjects=["S2"])["status"], "partial")
        self.assertEqual(self.run_check(config, resume=True)["status"], "complete")

    def test_scaler_fits_outer_train_only(self):
        means = []

        class RecordingScaler(StandardScaler):
            def fit(self, X, y=None, **kwargs):
                means.append(float(np.asarray(X)[:, 0].mean()))
                return super().fit(X, y, **kwargs)

        config = {**self.config, "models": ["dummy"], "tasks": ["multiclass"]}
        with patch("wesad.evaluation.build_pipeline", return_value=Pipeline([
                ("scaler", RecordingScaler()), ("model", DummyClassifier())])):
            self.run_check(config, allow_partial=True)
        expected = [self.frame.loc[self.frame.subject_id != s, "signal"].mean() for s in sorted(self.frame.subject_id.unique())]
        np.testing.assert_allclose(means, expected)

    def test_nested_inner_groups_and_checkpoint_resume(self):
        config = {**self.config, "models": ["knn", "svm", "rf", "mlp"], "tasks": ["multiclass"],
                  "nested": True, "inner_splits": 3, "model_params": {"mlp": {"max_iter": 5}},
                  "search_spaces": {"knn": {"model__n_neighbors": [3]}, "svm": {"model__C": [1]},
                                    "rf": {"model__n_estimators": [3]}, "mlp": {"model__hidden_layer_sizes": [[4]]}}}
        self.run_check(config, allow_partial=True, outer_subjects=["S2"])
        for split in read_json(self.root / "run/splits.json").values():
            for inner in split["inner_splits"]:
                self.assertNotIn("S2", inner["train_subjects"] + inner["validation_subjects"])
                self.assertFalse(set(inner["train_subjects"]) & set(inner["validation_subjects"]))
        with patch("wesad.evaluation.clone", side_effect=AssertionError("must resume")):
            self.run_check(config, allow_partial=True, outer_subjects=["S2"], resume=True)
        corrupt = self.root / "run/folds/multiclass/knn/S2/predictions.csv"
        corrupt.write_text("broken\n")
        self.run_check(config, allow_partial=True, outer_subjects=["S2"], resume=True)
        self.assertEqual(len(pd.read_csv(corrupt)), 12)
        with self.assertRaises(ValueError):
            self.run_check({**config, "seed": 123}, allow_partial=True, resume=True)

    def test_balancing_sampler_only_changes_training(self):
        frame = self.frame.iloc[:12].copy()
        pipeline = build_pipeline("knn", {"balance": True})
        pipeline.fit(frame[["signal", "missing"]], frame.label_original)
        self.assertEqual(len(pipeline.predict(frame[["signal", "missing"]])), len(frame))
        self.assertEqual(build_pipeline("lr", {"balance": True})["model"].class_weight, "balanced")


if __name__ == "__main__":
    unittest.main()
