"""G1: test nhãn trên cửa sổ thật 60 giây ở 700 Hz, không triển khai G2."""

import runpy
import sys
import unittest
from unittest.mock import patch

import numpy as np

from wesad.preprocessing import iter_windows
from wesad.utils import PROJECT_ROOT


class G1WindowingTests(unittest.TestCase):
    def windows(self, labels):
        labels = np.asarray(labels)
        return list(iter_windows({"ECG": np.ones(len(labels))}, labels))

    def test_defaults_use_60_second_windows_and_5_second_steps(self):
        windows = self.windows(np.ones(70 * 700 + 100))
        self.assertEqual([w.start_index for w in windows], [0, 3500, 7000])
        self.assertEqual([w.start_s for w in windows], [0, 5, 10])
        self.assertTrue(all(w.end_s - w.start_s == 60 for w in windows))
        self.assertTrue(all(len(w.signals["ECG"]) == 42000 for w in windows))

    def test_exact_90_percent_is_accepted_in_default_majority_mode(self):
        labels = np.r_[np.full(37800, 2), np.zeros(4200)]
        windows = self.windows(labels)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].label, 2)
        self.assertEqual(windows[0].label_fraction, .9)

    def test_one_sample_below_90_percent_is_rejected(self):
        self.assertFalse(self.windows(np.r_[np.full(37799, 2), np.zeros(4201)]))

    def test_excluded_labels_never_become_target_class(self):
        for label in (0, 4, 5, 6, 7):
            with self.subTest(label=label):
                self.assertFalse(self.windows(np.r_[np.full(37800, label), np.ones(4200)]))

    def test_removed_label_segments_do_not_shift_original_timeline(self):
        labels = np.r_[np.ones(60 * 700), np.zeros(60 * 700), np.full(60 * 700, 3)]
        windows = self.windows(labels)
        self.assertIn((120, 180, 3), [(w.start_s, w.end_s, w.label) for w in windows])
        self.assertTrue(all(w.label_fraction >= .9 for w in windows))
        self.assertFalse(any(10 <= w.start_s <= 110 for w in windows))

    def test_cli_default_uses_shared_config_without_loading_raw(self):
        argv = ["extract_features.py", "--subjects", "S2", "--output-dir", "outputs/g1/test_only"]
        with patch.object(sys, "argv", argv), patch("wesad.extraction.run_extraction") as run:
            runpy.run_path(str(PROJECT_ROOT / "scripts/extract_features.py"), run_name="__main__")
        run.assert_called_once()
        config, subjects, resume = run.call_args.args
        self.assertEqual(subjects, ["S2"])
        self.assertFalse(resume)
        self.assertEqual(config["output_dir"], "outputs/g1/test_only")
        self.assertEqual(config["window_seconds"], 60)
        self.assertEqual(config["step_seconds"], 5)
        self.assertEqual(config["label_rule"], "majority")
        self.assertEqual(config["min_label_fraction"], .9)
        self.assertEqual(config["device"], "combined")
        self.assertEqual(config["subject_processor"], "wesad.multisensor:process_subject")


if __name__ == "__main__":
    unittest.main()
