"""Check the G1 feature contract against the existing infrastructure APIs."""

from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from wesad.data import METADATA, REQUIRED, SUBJECTS, load_feature_table, validate_features
from wesad.evaluation import task_target
from wesad.extraction import process_subject
from wesad.utils import PROJECT_ROOT


class G1ContractTests(unittest.TestCase):
    def setUp(self):
        self.schema_path = PROJECT_ROOT / "configs/feature_schema.json"
        self.schema = json.loads(self.schema_path.read_text(encoding="utf-8"))
        rows = []
        for i, label in enumerate((1, 2, 3)):
            rows.append({"subject_id": "S2", "window_id": f"S2_{i * 3500}",
                         "start_s": i * 5., "end_s": i * 5. + 60,
                         "label_original": label, "label_binary": int(label == 2),
                         "label_fraction": .9, "quality_flag": "fixture_not_real_QA",
                         **{name: float(i) for name in self.schema["feature_columns"]}})
        self.frame = pd.DataFrame(rows)

    def test_subjects_metadata_and_feature_groups_match(self):
        self.assertEqual(self.schema["expected_subjects"], list(SUBJECTS))
        self.assertTrue(set(REQUIRED) <= set(self.schema["metadata_columns"]) <= METADATA)
        self.assertFalse(set(self.schema["feature_columns"]) & METADATA)
        self.assertEqual(self.schema["feature_columns"],
                         self.schema["sensor_groups"]["chest"] + self.schema["sensor_groups"]["wrist"])

    def test_all_planned_columns_load_in_schema_order_and_nan_is_preserved(self):
        self.frame.loc[0, self.schema["feature_columns"][0]] = np.nan
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "features.csv"
            self.frame[self.frame.columns[::-1]].to_csv(path, index=False)
            loaded, columns = load_feature_table(path, self.schema_path, ["S2"])
        self.assertEqual(columns, self.schema["feature_columns"])
        self.assertTrue(np.isnan(loaded.loc[0, columns[0]]))

    def test_binary_mapping_matches_runner(self):
        target, labels = task_target(self.frame, "binary")
        self.assertEqual(target.tolist(), [0, 1, 0])
        self.assertEqual(labels, [0, 1])
        self.frame.loc[0, "label_binary"] = 1
        with self.assertRaises(ValueError):
            validate_features(self.frame, self.schema["feature_columns"])

    def test_invalid_labels_duplicate_keys_and_inf_are_rejected(self):
        for change in ("label", "duplicate", "inf"):
            with self.subTest(change=change):
                frame = self.frame.copy()
                if change == "label":
                    frame.loc[0, "label_original"] = 4
                elif change == "duplicate":
                    frame.loc[1, "window_id"] = frame.loc[0, "window_id"]
                else:
                    frame.loc[0, self.schema["feature_columns"][0]] = np.inf
                with self.assertRaises(ValueError):
                    validate_features(frame, self.schema["feature_columns"])

    def test_existing_processor_emits_matching_metadata_but_raw_features_are_separate(self):
        config = {"device": "chest", "sampling_rate_hz": 700, "sensors": ["ECG", "EDA", "Resp"],
                  "window_seconds": 60, "step_seconds": 5, "labels": [1, 2, 3],
                  "label_rule": "majority", "min_label_fraction": .9}
        data = {"label": np.ones(65 * 700, dtype=int),
                "signal": {"chest": {name: np.ones((65 * 700, 1)) for name in config["sensors"]}}}
        with patch("wesad.extraction.load_subject", return_value=data):
            frame = process_subject(Path("S2.pkl"), config)
        self.assertEqual(frame.columns[:8].tolist(), self.schema["metadata_columns"])
        self.assertEqual(frame.window_id.tolist(), ["S2_0", "S2_3500"])
        self.assertEqual(frame.start_s.tolist(), [0., 5.])
        self.assertEqual(frame.label_binary.tolist(), [0, 0])
        self.assertEqual(len(validate_features(frame)), 21)
        with self.assertRaises(ValueError):
            validate_features(frame, self.schema["feature_columns"])


if __name__ == "__main__":
    unittest.main()
