"""Behavior checks for G2 filters, common windows and five-second aggregation."""

import unittest

import numpy as np

from wesad.multisensor import SENSOR_SPECS
from wesad.preprocessing import Window
from wesad.preprocessing_g2 import (acc_emg_block_statistics, filter_finite_segments,
                                    iter_preprocessed_windows)
from wesad.utils import read_config


class G2PreprocessingTests(unittest.TestCase):
    def test_lowpass_attenuates_high_frequency_and_preserves_low_frequency(self):
        rate = 100
        t = np.arange(20000) / rate
        low, high = np.sin(2 * np.pi * t), np.sin(2 * np.pi * 20 * t)
        y = filter_finite_segments(low + high, rate,
                                  {"kind": "lowpass", "cutoff_hz": 5, "edge_seconds": 10})
        good = np.isfinite(y)
        self.assertLess(np.sqrt(np.mean((y[good] - low[good]) ** 2)), .02)
        self.assertTrue(np.isnan(y[:1000]).all())
        self.assertTrue(np.isnan(y[-1000:]).all())

    def test_bandpass_retains_resp_frequency_and_rejects_drift(self):
        rate = 20
        t = np.arange(12000) / rate
        breathing = np.sin(2 * np.pi * .2 * t)
        x = breathing + np.sin(2 * np.pi * .01 * t) + np.sin(2 * np.pi * 2 * t)
        y = filter_finite_segments(x, rate,
                                  {"kind": "bandpass", "cutoff_hz": [.1, .35], "edge_seconds": 30})
        good = np.isfinite(y)
        self.assertLess(np.sqrt(np.mean((y[good] - breathing[good]) ** 2)), .03)

    def test_gap_is_not_bridged_and_short_segments_are_unusable(self):
        spec = {"kind": "lowpass", "cutoff_hz": 1, "edge_seconds": 2}
        x = np.r_[np.ones(200), np.nan, np.ones(200) * 3]
        y = filter_finite_segments(x, 4, spec)
        changed = x.copy()
        changed[:200] = 100
        other = filter_finite_segments(changed, 4, spec)
        np.testing.assert_equal(y[201:], other[201:])
        self.assertTrue(np.isnan(y[192:209]).all())
        self.assertTrue(np.isnan(filter_finite_segments(np.ones(10), 4, spec)).all())

    def test_native_rates_edges_and_original_window_keys(self):
        config = read_config("configs/preprocessing_g2_basic.yaml")
        data = {"label": np.ones(120 * 700, dtype=np.int8), "signal": {}}
        for device, specs in SENSOR_SPECS.items():
            data["signal"][device] = {sensor: np.ones((120 * rate, axes))
                                      for sensor, (rate, axes) in specs.items()}
        qa = {}
        windows = list(iter_preprocessed_windows(data, config, qa))
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].start_index, 30 * 700)
        self.assertEqual(len(windows[0].signals["wrist_bvp"]), 60 * 64)
        self.assertEqual(len(windows[0].signals["wrist_acc_x"]), 60 * 32)
        self.assertEqual(qa["n_rejected_filter_edge_or_short_segment"], 12)
        self.assertEqual(qa["n_candidate_windows"], 13)

    def test_exact_90_percent_keeps_original_label_denominator(self):
        config = read_config("configs/preprocessing_g2_basic.yaml")
        data = {"label": np.ones(120 * 700, dtype=np.int8), "signal": {}}
        for device, specs in SENSOR_SPECS.items():
            data["signal"][device] = {s: np.ones((120 * rate, axes))
                                      for s, (rate, axes) in specs.items()}
        data["label"][30 * 700:36 * 700] = 0
        windows = list(iter_preprocessed_windows(data, config, {}))
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].label_fraction, .9)
        data["label"][36 * 700] = 0
        self.assertEqual(list(iter_preprocessed_windows(data, config, {})), [])

    def test_short_or_nonfinite_required_channel_rejects_common_window(self):
        config = read_config("configs/preprocessing_g2_basic.yaml")
        for failure in ("short", "nan"):
            data = {"label": np.ones(120 * 700, dtype=np.int8), "signal": {}}
            for device, specs in SENSOR_SPECS.items():
                data["signal"][device] = {s: np.ones((120 * rate, axes))
                                          for s, (rate, axes) in specs.items()}
            if failure == "short":
                data["signal"]["wrist"]["BVP"] = np.ones((64 * 80, 1))
            else:
                data["signal"]["wrist"]["BVP"][64 * 40] = np.nan
            qa = {}
            self.assertEqual(list(iter_preprocessed_windows(data, config, qa)), [])
            self.assertGreater(qa["n_rejected_incomplete_signal" if failure == "short"
                                  else "n_rejected_nonfinite_signal"], 0)
            self.assertEqual(qa["n_candidate_windows"], qa["n_kept_windows"] + qa["n_rejected_label"]
                             + qa["n_rejected_filter_edge_or_short_segment"]
                             + qa["n_rejected_incomplete_signal"] + qa["n_rejected_nonfinite_signal"])

    def test_block_mean_is_not_whole_window_standard_deviation(self):
        # Each block is constant, although means differ between blocks.
        x = np.repeat(np.arange(12), 5 * 700)
        window = Window(0, 42000, 0, 60, 1, {"chest_emg": x})
        stats = acc_emg_block_statistics(window)
        self.assertEqual(stats["chest_emg_std_block5_mean"], 0)
        self.assertGreater(x.std(), 0)
        self.assertEqual(stats["chest_emg_mean_block5_mean"], 5.5)
        self.assertEqual(stats["chest_emg_iav_block5_mean"], 27.5)

    def test_incomplete_blocks_are_not_silently_averaged(self):
        with self.assertRaises(ValueError):
            acc_emg_block_statistics(Window(0, 42000, 0, 60, 1, {"wrist_acc_x": np.ones(1919)}))


if __name__ == "__main__":
    unittest.main()
