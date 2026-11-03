"""Kiểm tra số học, đồng bộ và lỗi dữ liệu; không thay nghiệm thu raw."""

import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from wesad.preprocessing import filter_signal, five_second_features, prepare_signals
from wesad.utils import PROJECT_ROOT, read_config
from wesad.windowing import SampledSignal, WindowAudit, iter_multirate_windows


class NativeWindowsTests(unittest.TestCase):
    def test_native_rates_time_grid_and_original_key(self):
        signals = {str(fs): SampledSignal(np.arange(70 * fs), fs) for fs in (700, 64, 32, 4)}
        windows = list(iter_multirate_windows(signals, np.ones(70 * 700)))
        self.assertEqual([w.start_s for w in windows], [0, 5, 10])
        for fs in (700, 64, 32, 4):
            np.testing.assert_array_equal(windows[1].signals[str(fs)].values, np.arange(5 * fs, 65 * fs))
        self.assertEqual(windows[1].metadata("S2")["window_id"], "S2_3500")
        self.assertEqual(windows[1].metadata("S2")["label_binary"], 0)

    def test_threshold_and_excluded_labels_in_denominator(self):
        signals = {"eda": SampledSignal(np.ones(60 * 4), 4)}
        for count, accepted in ((37800, 1), (37799, 0)):
            labels = np.r_[np.full(count, 2), np.zeros(42000 - count)]
            windows = list(iter_multirate_windows(signals, labels))
            self.assertEqual(len(windows), accepted)
            if accepted:
                self.assertEqual(windows[0].label_fraction, .9)
                self.assertEqual(windows[0].metadata("S2")["label_binary"], 1)
        self.assertFalse(list(iter_multirate_windows(signals, np.full(42000, 4))))

    def test_common_windows_do_not_bridge_missing_samples(self):
        x = np.ones(180 * 4)
        x[70 * 4] = np.nan
        audit = WindowAudit()
        windows = list(iter_multirate_windows({"eda": SampledSignal(x, 4)},
                       np.ones(180 * 700), audit=audit))
        starts = [w.start_s for w in windows]
        self.assertIn(10, starts)  # [10,70) does not contain sample at 70 s
        self.assertNotIn(15, starts)
        self.assertIn(75, starts)
        self.assertEqual(audit.candidates, audit.accepted + audit.label_rejected + audit.signal_rejected)
        self.assertEqual(audit.reasons["eda:nonfinite_or_filter_edge"], 12)

    def test_short_channel_not_silently_truncated(self):
        audit = WindowAudit()
        windows = list(iter_multirate_windows({"wrist": SampledSignal(np.ones(239), 4)},
                       np.ones(42000), audit=audit))
        self.assertFalse(windows)
        self.assertEqual(audit.reasons, {"wrist:incomplete": 1})

    def test_bad_labels_and_noninteger_native_grid_raise(self):
        for labels in (np.full(42000, np.nan), np.full(42000, 1.5)):
            with self.assertRaises(ValueError):
                list(iter_multirate_windows({"x": SampledSignal(np.ones(240), 4)}, labels))
        with self.assertRaises(ValueError):
            list(iter_multirate_windows({"x": SampledSignal(np.ones(240), 4)},
                                       np.ones(42000), step_seconds=.1))


class FilterTests(unittest.TestCase):
    def test_lowpass_keeps_slow_component_and_removes_high_frequency(self):
        fs = 100
        time = np.arange(60 * fs) / fs
        slow = np.sin(2 * np.pi * time)
        mixed = slow + np.sin(2 * np.pi * 30 * time)
        original = mixed.copy()
        filtered = filter_signal(mixed, fs, kind="lowpass", cutoff_hz=5, edge_seconds=2)
        inside = np.isfinite(filtered)
        self.assertLess(np.sqrt(np.mean((filtered[inside] - slow[inside]) ** 2)), .01)
        self.assertTrue(np.isnan(filtered[:200]).all())
        np.testing.assert_array_equal(mixed, original)

    def test_resp_bandpass_preserves_breathing_frequency(self):
        fs = 10
        time = np.arange(300 * fs) / fs
        slow = np.sin(2 * np.pi * .2 * time)
        mixed = slow + np.sin(2 * np.pi * time) + 3
        filtered = filter_signal(mixed, fs, kind="bandpass", cutoff_hz=[.1, .35], edge_seconds=30)
        inside = np.isfinite(filtered)
        self.assertLess(np.sqrt(np.mean((filtered[inside] - slow[inside]) ** 2)), .05)

    def test_filter_segments_are_independent_and_short_segment_is_missing(self):
        fs = 20
        x = np.r_[np.ones(200), np.nan, np.full(200, 8), np.inf, np.ones(2)]
        filtered = filter_signal(x, fs, kind="lowpass", cutoff_hz=2, edge_seconds=1)
        np.testing.assert_allclose(filtered[20:180], 1, atol=1e-10)
        np.testing.assert_allclose(filtered[221:381], 8, atol=1e-10)
        self.assertTrue(np.isnan(filtered[180:221]).all())
        self.assertTrue(np.isnan(filtered[-3:]).all())

    def test_nyquist_guard_and_axis_zero(self):
        with self.assertRaises(ValueError):
            filter_signal(np.ones(200), 4, kind="lowpass", cutoff_hz=5, edge_seconds=0)
        x = np.column_stack([np.ones(200), np.full(200, 2)])
        filtered = filter_signal(x, 20, kind="lowpass", cutoff_hz=2, edge_seconds=1)
        np.testing.assert_allclose(filtered[20:-20], x[20:-20], atol=1e-10)


class BlockFeaturesTests(unittest.TestCase):
    def test_acc_integration_magnitude_and_mean_of_block_std(self):
        # Different constant blocks: std per block is zero, global std is not.
        x = np.zeros((60 * 4, 3))
        x[:, 0] = np.repeat(np.arange(1, 13), 5 * 4)
        f = five_second_features(x, 4, sensor="acc")
        self.assertAlmostEqual(f["acc_x_5s_mean_mean"], 6.5)
        self.assertEqual(f["acc_x_5s_std_mean"], 0)
        self.assertAlmostEqual(f["acc_magnitude_5s_abs_integral_mean"], 32.5)
        self.assertEqual(len(f), 12)

    def test_emg_statistics_and_waveform_length_ignore_block_boundaries(self):
        x = np.repeat(np.arange(1, 13), 5 * 10)
        f = five_second_features(x, 10, sensor="emg")
        self.assertEqual(f["emg_5s_rms_mean"], 6.5)
        self.assertEqual(f["emg_5s_mav_mean"], 6.5)
        self.assertEqual(f["emg_5s_abs_integral_mean"], 32.5)
        self.assertEqual(f["emg_5s_waveform_length_per_s_mean"], 0)
        # Within each block, alternating 0/1 gives 49 transitions per 5 s.
        f = five_second_features(np.tile([0., 1.], 300), 10, sensor="emg")
        self.assertAlmostEqual(f["emg_5s_waveform_length_per_s_mean"], 49 / 5)

    def test_feature_names_match_mai_draft_and_missing_is_not_zero_filled(self):
        schema = json.loads((PROJECT_ROOT / "configs/feature_schema.json").read_text(encoding="utf-8"))
        for prefix in ("chest", "wrist"):
            f = five_second_features(np.ones((240, 3)), 4, sensor="acc")
            self.assertTrue({f"{prefix}_{key}" for key in f} <= set(schema["feature_columns"]))
        f = five_second_features(np.ones(240), 4, sensor="emg")
        self.assertTrue({f"chest_{key}" for key in f} <= set(schema["feature_columns"]))
        bad = np.ones((240, 3))
        bad[1, 0] = np.nan
        with self.assertRaises(ValueError):
            five_second_features(bad, 4, sensor="acc")
        with self.assertRaises(ValueError):
            five_second_features(np.ones(239), 4, sensor="emg")


class PreparationTests(unittest.TestCase):
    def test_prepare_preserves_input_units_and_rejects_wrong_axes(self):
        config = {"channels": {"acc": {"device": "wrist", "sensor": "ACC", "axes": 3,
                                       "sampling_rate_hz": 32}}}
        data = {"signal": {"wrist": {"ACC": np.full((1920, 3), 64)}}}
        np.testing.assert_array_equal(prepare_signals(data, config)["acc"].values, data["signal"]["wrist"]["ACC"])
        data["signal"]["wrist"]["ACC"] = np.ones((1920, 2))
        with self.assertRaises(ValueError):
            prepare_signals(data, config)

    def test_diagnostic_cli_produces_traceable_metadata_not_training_features(self):
        help_result = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts/check_preprocessing.py"),
                                      "--help"], cwd=PROJECT_ROOT, capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        config = read_config("configs/preprocessing_nhu_draft.yaml")
        data = {"label": np.ones(180 * 700), "signal": {"chest": {}, "wrist": {}}}
        for spec in config["channels"].values():
            data["signal"][spec["device"]][spec["sensor"]] = np.ones((180 * spec["sampling_rate_hz"], spec["axes"]))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw/S2"
            raw.mkdir(parents=True)
            with (raw / "S2.pkl").open("wb") as stream:
                pickle.dump(data, stream)
            local = root / "local.yaml"
            local.write_text("data_root: " + json.dumps(str(raw.parent)), encoding="utf-8")
            command = [sys.executable, str(PROJECT_ROOT / "scripts/check_preprocessing.py"),
                       "--local", str(local), "--output", str(root / "out")]
            result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            status = json.loads((root / "out/run_status.json").read_text())
            self.assertEqual(status["status"], "complete")
            self.assertFalse(status["ready_for_training"])
            quality = json.loads((root / "out/S2_quality.json").read_text())
            self.assertEqual(quality["audit"]["accepted"], 13)
            self.assertTrue(quality["channels"]["chest_acc"]["raw_flat"])
            snapshot = json.loads((root / "out/config.json").read_text())
            self.assertNotIn("data_root", snapshot["processing"])
            repeated = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)


if __name__ == "__main__":
    unittest.main()
