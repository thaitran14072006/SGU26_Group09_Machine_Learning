"""G1 acceptance checks must catch mismatched data and cannot certify four machines."""

import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from wesad.utils import PROJECT_ROOT, file_hash, read_json, write_json, read_config

spec = importlib.util.spec_from_file_location("check_g1_sample", PROJECT_ROOT / "scripts/check_g1_sample.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class G1SampleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.a, self.b = Path(self.temp.name) / "a", Path(self.temp.name) / "b"
        self.a.mkdir()
        rows = []
        for i in range(3):
            row = dict(zip(checker.METADATA, ["S2", f"S2_{i*3500}", float(i*5), float(i*5+60),
                                            i+1, int(i==1), .95, "finite_only_not_physiological_QA"]))
            row.update({name: float(i + j + 1) for j, name in enumerate(checker.expected_feature_columns())})
            rows.append(row)
        pd.DataFrame(rows).to_csv(self.a / "S2.csv", index=False)
        features = list(rows[0])[8:]
        settings = {k:v for k,v in read_config("configs/features_smoke.yaml").items()
                    if k not in {"data_root", "subjects", "output_dir"}}
        write_json(self.a / "config.json", {"signature": "test", "processing": settings,
                   "code_sha256": "a"*64, "environment": {"python": "3.12.14 test build", "packages": {"numpy": "test"}}})
        write_json(self.a / "feature_schema.json", {"feature_columns": features, "processing_signature": "test", "feature_version": "all_sensors_raw_stats_v1", "sensor_groups": checker.expected_sensor_groups(features)})
        write_json(self.a / "run_status.json", {"status": "complete", "subjects": ["S2"]})
        write_json(self.a / "S2.json", {"status": "complete", "signature": "test", "n_windows": 3,
                   "csv_sha256": file_hash(self.a / "S2.csv"), "source_sha256": "b"*64, "feature_columns": features})
        shutil.copytree(self.a, self.b)

    def edit_csv(self, change):
        frame = pd.read_csv(self.b / "S2.csv")
        change(frame)
        frame.to_csv(self.b / "S2.csv", index=False)
        manifest = read_json(self.b / "S2.json")
        manifest["csv_sha256"] = file_hash(self.b / "S2.csv")
        write_json(self.b / "S2.json", manifest)

    def test_single_package_is_not_four_machine_acceptance(self):
        report = checker.check_sample(self.a)
        self.assertEqual(report["n_windows"], 3)  # do not force real S2's observed 395
        self.assertFalse(report["four_machine_acceptance"])
        self.assertEqual(report["n_features"], 112)
        self.assertEqual(report["sensor_group_counts"], {"chest":63,"wrist":49,"combined":112})
        with self.assertRaises(ValueError):
            checker.check_sample(self.a, self.a)

    def test_tolerance_applies_to_features_only(self):
        self.edit_csv(lambda f: f.__setitem__("chest_ecg_mean", f.chest_ecg_mean + 1e-10))
        report = checker.check_sample(self.a, self.b)
        self.assertEqual(report["scope"], "g1_raw_stats_pairwise_packages")
        self.edit_csv(lambda f: f.__setitem__("chest_ecg_mean", f.chest_ecg_mean + 1))
        with self.assertRaises(ValueError):
            checker.check_sample(self.a, self.b)

    def test_old_21_feature_version_is_rejected(self):
        snapshot = read_json(self.b / "config.json")
        snapshot["processing"].update(device="chest", sensors=["ECG", "EDA", "Resp"],
                                      feature_version="chest_raw_stats_majority_v1")
        snapshot["processing"].pop("subject_processor")
        write_json(self.b / "config.json", snapshot)
        with self.assertRaises(ValueError):
            checker.check_sample(self.b)

    def test_incorrect_sensor_groups_are_rejected(self):
        schema = read_json(self.b / "feature_schema.json")
        schema["sensor_groups"]["wrist"] = schema["sensor_groups"]["wrist"][:-1]
        write_json(self.b / "feature_schema.json", schema)
        with self.assertRaises(ValueError):
            checker.check_sample(self.b)

    def test_window_grid_and_label_purity_are_checked(self):
        self.edit_csv(lambda f: f.__setitem__("label_fraction", .89))
        with self.assertRaises(ValueError):
            checker.check_sample(self.b)
        self.edit_csv(lambda f: (f.__setitem__("label_fraction", .95), f.__setitem__("start_s", f.start_s + 1)))
        with self.assertRaises(ValueError):
            checker.check_sample(self.b)

    def test_changed_environment_and_reference_metadata_are_rejected(self):
        snapshot = read_json(self.b / "config.json")
        snapshot["environment"]["python"] = "3.13.9 other build"
        write_json(self.b / "config.json", snapshot)
        with self.assertRaises(ValueError):
            checker.check_sample(self.a, self.b)
        shutil.copyfile(self.a / "config.json", self.b / "config.json")
        self.edit_csv(lambda f: f.__setitem__("label_fraction", .96))
        with self.assertRaises(ValueError):
            checker.check_sample(self.a, self.b)

    def test_corrupt_csv_and_feature_order_are_rejected(self):
        with (self.b / "S2.csv").open("a") as stream:
            stream.write("\n")
        with self.assertRaises(ValueError):
            checker.check_sample(self.b)
        frame = pd.read_csv(self.a / "S2.csv")
        frame = frame[list(frame.columns[:8]) + list(reversed(frame.columns[8:]))]
        frame.to_csv(self.b / "S2.csv", index=False)
        manifest = read_json(self.b / "S2.json")
        manifest["csv_sha256"] = file_hash(self.b / "S2.csv")
        write_json(self.b / "S2.json", manifest)
        with self.assertRaises(ValueError):
            checker.check_sample(self.b)
        self.edit_csv(lambda f: f.__setitem__("chest_ecg_mean", np.nan))
        with self.assertRaises(ValueError):
            checker.check_sample(self.b)


if __name__ == "__main__":
    unittest.main()
