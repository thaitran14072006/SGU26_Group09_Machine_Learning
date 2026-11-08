"""Default settings and train-only preprocessing for Nhu's RF/MLP factories."""

import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from wesad.evaluation import run_experiment
from wesad.models import build_pipeline
from wesad.models_rf_mlp import build_default_pipeline
from wesad.utils import read_config, read_json, write_csv


class NhuModelsTests(unittest.TestCase):
    def test_defaults_match_locked_library_except_seed_and_rf_jobs(self):
        for name, reference in [("rf", RandomForestClassifier(random_state=42, n_jobs=1)),
                                ("mlp", MLPClassifier(random_state=42))]:
            pipeline = build_default_pipeline(name, {})
            self.assertEqual(pipeline["model"].get_params(), reference.get_params())
            self.assertEqual([key for key, _ in pipeline.steps], ["imputer", "scaler", "model"])

    def test_config_registers_factories_for_group_choice_without_assuming_a_file(self):
        config = read_config("configs/g3_nhu_defaults.yaml")
        self.assertIsNone(config["features"])
        self.assertEqual(config["feature_version"], "physiological_68_v1")
        self.assertTrue(config["input_feature_family_confirmed_by_group"])
        for name in config["models"]:
            self.assertEqual(build_pipeline(name, config)["model"].random_state, 42)

    def test_quick_input_rejects_raw112_and_accepts_the_68_contract(self):
        import importlib.util
        from wesad.utils import PROJECT_ROOT, write_json
        spec = importlib.util.spec_from_file_location("nhu_g3_quick", PROJECT_ROOT / "scripts/run_nhu_g3_quick.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        contract = read_json(PROJECT_ROOT / "configs/feature_schema.json")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "features.csv"
            schema = root / "feature_schema.json"
            frame = pd.DataFrame([{"subject_id": "S4", "window_id": "S4_0", "start_s": 0,
                "end_s": 60, "label_original": 1, "label_binary": 0, "label_fraction": 1.,
                "quality_flag": "synthetic_test_only", **dict.fromkeys(contract["feature_columns"], 0.)}],
                columns=contract["output_column_order"])
            write_csv(path, frame)
            write_json(schema, contract)
            self.assertEqual(module.check_input(path)["n_features"], 68)
            write_json(schema, {**contract, "feature_version": "all_sensors_raw_stats_v1"})
            with self.assertRaisesRegex(ValueError, "group selected"):
                module.check_input(path)

    def test_default_factory_rejects_nondefault_experiments(self):
        for change in [{"nested": True}, {"balance": True}, {"variance_threshold": 0},
                       {"model_params": {"mlp": {"max_iter": 5}}}, {"seed": 1}]:
            with self.assertRaises(ValueError):
                build_default_pipeline("mlp", change)

    def test_imputer_and_scaler_do_not_fit_test_distribution(self):
        train = pd.DataFrame({"x": [0., 2., np.nan, 4.], "empty": [np.nan] * 4})
        test = pd.DataFrame({"x": [1000., np.nan], "empty": [500., np.nan]})
        for name in ["rf", "mlp"]:
            pipeline = build_default_pipeline(name, {})
            pipeline.fit(train, [0, 1, 0, 1])
            before = pipeline["imputer"].statistics_.copy(), pipeline["scaler"].mean_.copy()
            prediction = pipeline.predict(test)
            self.assertEqual(len(prediction), 2)
            np.testing.assert_allclose(before[0], [2., 0.])
            np.testing.assert_allclose(before[1], [2., 0.])
            np.testing.assert_equal(before[0], pipeline["imputer"].statistics_)
            np.testing.assert_equal(before[1], pipeline["scaler"].mean_)

    def test_one_subject_check_for_both_models_and_tasks_stays_partial(self):
        # Deliberately synthetic; these tiny fixtures never constitute model results.
        rows = []
        for subject in ["S2", "S3", "S4"]:
            for i in range(18):
                label = i % 3 + 1
                rows.append({"subject_id": subject, "window_id": f"{subject}_{i * 3500}",
                             "start_s": i * 5, "end_s": i * 5 + 60, "label_original": label,
                             "label_binary": int(label == 2), "label_fraction": 1.,
                             "quality_flag": "synthetic_test_only", "signal": float(label),
                             "missing": np.nan if i % 4 == 0 else float(i % 2)})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_csv(root / "features.csv", pd.DataFrame(rows))
            config = read_config("configs/g3_nhu_defaults.yaml")
            config.update(features=str(root / "features.csv"), output_dir=str(root / "run"))
            with redirect_stdout(io.StringIO()):
                result = run_experiment(config, outer_subjects=["S4"], allow_partial=True)
            self.assertEqual(result["status"], "partial")
            predictions = pd.read_csv(root / "run/predictions.csv")
            self.assertEqual(len(predictions), 18 * 4)
            self.assertFalse(predictions.duplicated(["task", "model", "window_id"]).any())
            for detail in (root / "run/folds").glob("*/*/S4/details.json"):
                record = read_json(detail)
                self.assertNotIn("S4", record["outer_train_subjects"])
                self.assertEqual(record["outer_test_subjects"], ["S4"])
                self.assertIn("warnings", record)
            # Resume must reuse all four validated fold outputs without fitting again.
            with redirect_stdout(io.StringIO()), patch("wesad.evaluation.clone", side_effect=AssertionError("Unexpected refit")):
                self.assertEqual(run_experiment(config, resume=True, outer_subjects=["S4"], allow_partial=True)["status"], "partial")


if __name__ == "__main__":
    unittest.main()
