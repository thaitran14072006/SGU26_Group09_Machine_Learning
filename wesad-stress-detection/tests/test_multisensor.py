"""Native-rate alignment, common windows and extraction handoff (synthetic PKLs)."""

import copy
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from wesad.data import load_feature_table
from wesad.extraction import process_subject as legacy_process, run_extraction
from wesad.multisensor import SENSOR_SPECS, process_subject
from wesad.utils import read_config, read_json


class MultisensorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "S2" / "S2.pkl"
        self.path.parent.mkdir()
        self.config = read_config("configs/features_smoke.yaml")
        self.config.update(data_root=str(self.root), subjects=["S2"], output_dir=str(self.root / "features"))
        self.raw = {"label": np.ones(65 * 700), "signal": {}}
        for device, specs in SENSOR_SPECS.items():
            self.raw["signal"][device] = {}
            for sensor, (rate, axes) in specs.items():
                values = np.arange(65 * rate, dtype=float).reshape(-1, 1)
                if axes == 3:
                    values = np.tile([1., 2., 2.], (65 * rate, 1))
                self.raw["signal"][device][sensor] = values
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        with self.path.open("wb") as stream:
            pickle.dump(self.raw, stream)

    def test_all_channels_are_native_rate_aligned_and_acc_axes_separate(self):
        frame = process_subject(self.path, self.config)
        self.assertEqual(frame.shape, (2, 120))  # 112 features + 8 metadata
        self.assertEqual(frame.window_id.tolist(), ["S2_0", "S2_3500"])
        for device, specs in SENSOR_SPECS.items():
            for sensor, (rate, axes) in specs.items():
                prefix = f"{device}_{sensor.lower()}"
                if axes == 1:
                    self.assertAlmostEqual(frame.iloc[0][prefix + "_mean"], (60 * rate - 1) / 2)
                    self.assertAlmostEqual(frame.iloc[1][prefix + "_mean"], (5 * rate + 65 * rate - 1) / 2)
                else:
                    for axis, expected in (("x", 1), ("y", 2), ("z", 2), ("magnitude", 3)):
                        self.assertEqual(frame.iloc[1][f"{prefix}_{axis}_mean"], expected)
        self.assertEqual(frame.attrs["extraction_qa"]["n_kept_windows"], 2)

    def test_nan_on_end_boundary_only_excludes_windows_containing_it(self):
        self.raw["signal"]["wrist"]["EDA"][240] = np.nan  # exactly 60s, not part of [0,60)
        self.save()
        frame = process_subject(self.path, self.config)
        self.assertEqual(frame.start_s.tolist(), [0.])
        self.assertEqual(frame.attrs["extraction_qa"]["n_rejected_nonfinite_signal"], 1)
        self.assertTrue(np.isfinite(frame.filter(regex="^(chest|wrist)_").to_numpy()).all())

    def test_short_sensor_excludes_common_window_instead_of_truncating(self):
        self.raw["signal"]["wrist"]["BVP"] = self.raw["signal"]["wrist"]["BVP"][:60 * 64]
        self.save()
        frame = process_subject(self.path, self.config)
        self.assertEqual(frame.start_s.tolist(), [0.])
        self.assertEqual(frame.attrs["extraction_qa"]["n_rejected_incomplete_signal"], 1)

    def test_majority_counts_excluded_labels_in_denominator(self):
        self.raw["label"] = np.r_[np.full(37800, 2), np.zeros(4200)]
        self.save()
        frame = process_subject(self.path, self.config)
        self.assertEqual(frame.label_original.tolist(), [2])
        self.assertEqual(frame.label_binary.tolist(), [1])
        self.assertEqual(frame.label_fraction.tolist(), [.9])
        self.raw["label"][37799] = 0
        self.save()
        with self.assertRaisesRegex(ValueError, "no common windows"):
            process_subject(self.path, self.config)

    def test_original_timeline_preserved_across_removed_labels(self):
        self.raw["label"] = np.r_[np.ones(42000), np.zeros(42000), np.full(42000, 3)]
        for device, specs in SENSOR_SPECS.items():
            for sensor, (rate, axes) in specs.items():
                self.raw["signal"][device][sensor] = np.ones((180 * rate, axes))
        self.save()
        frame = process_subject(self.path, self.config)
        self.assertIn("S2_84000", frame.window_id.tolist())
        qa = frame.attrs["extraction_qa"]
        self.assertEqual(qa["n_candidate_windows"], qa["n_kept_windows"] + qa["n_rejected_label"])
        self.assertTrue((frame.label_fraction >= .9).all())

    def test_missing_sensor_wrong_axes_and_empty_selection_fail(self):
        original = copy.deepcopy(self.raw)
        del self.raw["signal"]["wrist"]["TEMP"]
        self.save()
        with self.assertRaisesRegex(ValueError, "missing required wrist/TEMP"):
            process_subject(self.path, self.config)
        self.raw = original
        self.raw["signal"]["chest"]["ACC"] = np.ones((65 * 700, 1))
        self.save()
        with self.assertRaisesRegex(ValueError, "expected nonempty numeric"):
            process_subject(self.path, self.config)
        with self.assertRaisesRegex(ValueError, "Invalid chest sensors"):
            process_subject(self.path, {**self.config, "sensors": {"chest": []}})

    def test_export_resume_and_generated_groups(self):
        first = run_extraction(self.config)
        out = Path(self.config["output_dir"])
        schema = read_json(out / "feature_schema.json")
        self.assertEqual(len(schema["sensor_groups"]["chest"]), 63)
        self.assertEqual(len(schema["sensor_groups"]["wrist"]), 49)
        self.assertEqual(len(schema["feature_columns"]), 112)
        frame, columns = load_feature_table(out / "all_subjects.csv", expected_subjects=["S2"])
        self.assertEqual(len(columns), 112)
        with patch("wesad.multisensor.process_subject", side_effect=AssertionError("must use verified cache")):
            second = run_extraction(self.config, resume=True)
        pd.testing.assert_frame_equal(first, second, check_exact=False)
        report = pd.read_csv(out / "quality_report.csv")
        self.assertTrue(report.resumed.iloc[0])
        self.assertEqual(report.n_candidate_windows.iloc[0], 2)
        self.assertEqual(report.n_kept_windows.iloc[0], 2)
        self.assertEqual(read_json(out / "run_status.json")["subjects"], ["S2"])

    def test_old_raw_stat_config_still_produces_21_features(self):
        config = {"device": "chest", "sampling_rate_hz": 700, "sensors": ["ECG", "EDA", "Resp"],
                  "labels": [1, 2, 3], "window_seconds": 60, "step_seconds": 5,
                  "label_rule": "majority", "min_label_fraction": .9}
        frame = legacy_process(self.path, config)
        self.assertEqual(frame.shape, (2, 29))
        self.assertIn("ecg_mean", frame)
        self.assertNotIn("wrist_bvp_mean", frame)


if __name__ == "__main__":
    unittest.main()
