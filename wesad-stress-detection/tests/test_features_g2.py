"""Independent numerical/QA invariants for Mai's G2 physiological contract."""
import copy
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from wesad.features import (Channel, band_powers, block_features, detect_scr,
                            interval_features, process_subject, read_contract, window_features)
from wesad.multisensor import SENSOR_SPECS
from wesad.utils import read_config, read_json, resolve_path


class PhysiologicalFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = read_config("configs/features_g2.yaml")
        cls.contract = read_json(resolve_path(cls.config["feature_contract"]))

    def test_cardiac_rate_and_rr_variability_have_correct_units(self):
        rr = np.tile([1., 1.1], 25)
        peaks = np.r_[0, np.cumsum(rr) * 700].round().astype(int)
        values, reason = interval_features(peaks, 700)
        self.assertIsNone(reason)
        self.assertAlmostEqual(values[0], np.mean(60 / rr))
        self.assertAlmostEqual(values[2], 1050)
        self.assertAlmostEqual(values[3], rr.std(ddof=1) * 1000)
        self.assertAlmostEqual(values[4], 100)
        self.assertEqual(values[5], 1.)

    def test_rejected_interval_is_not_bridged_for_rmssd(self):
        rr = np.r_[np.ones(25), .1, np.ones(25)]
        values, reason = interval_features(np.r_[0, np.cumsum(rr)*700], 700)
        self.assertIsNone(reason)
        self.assertAlmostEqual(values[4], 0)
        self.assertEqual(values[5], 0)
        values, reason = interval_features(np.arange(10)*700, 700)
        self.assertTrue(np.isnan(values).all())
        self.assertEqual(reason, "insufficient_or_unreliable_beats")

    def test_exactly_50_ms_does_not_count_in_pnn50(self):
        rr_samples = np.tile([700, 735], 25)
        values, reason = interval_features(np.r_[0, np.cumsum(rr_samples)], 700)
        self.assertIsNone(reason)
        self.assertEqual(values[5], 0)

    def test_psd_integrated_power_of_known_sine(self):
        fs = 700
        x = 2*np.sin(2*np.pi*7.5*np.arange(fs*60)/fs)
        power = band_powers(x, fs, [[5, 10], [10, 20], [20, 40]])
        self.assertAlmostEqual(power[0], 2., places=9)
        self.assertLess(power[1] + power[2], 1e-20)

    def test_acc_and_emg_aggregate_all_twelve_blocks(self):
        x = np.repeat(np.arange(1., 13), 5*32)
        mean, std, integral = block_features(x, 32)
        self.assertEqual(mean, 6.5)
        self.assertEqual(std, 0)
        self.assertEqual(integral, 32.5)
        emg = block_features(np.full(60*700, 2.), 700, emg=True)
        np.testing.assert_allclose(emg, [2, 2, 0, 10])
        with self.assertRaises(ValueError):
            block_features(x[:-1], 32)

    def test_scr_onset_to_peak_amplitude_and_full_event_boundaries(self):
        fs = 4
        x = np.zeros(80)
        x[20:25] = [0, .01, .02, .04, .02]
        events = detect_scr(x, fs, np.ones(len(x), dtype=bool))
        self.assertEqual(events.shape, (1, 3))
        self.assertAlmostEqual(events[0, 2], .04)
        self.assertLess(events[0, 0], events[0, 1])
        mask = np.ones(len(x), dtype=bool)
        mask[22] = False
        self.assertEqual(len(detect_scr(x, fs, mask)), 0)

    def prepared(self):
        channels = {}
        for device, specs in SENSOR_SPECS.items():
            for sensor, (fs, axes) in specs.items():
                t = np.arange(180*fs)/fs
                x = np.ones((len(t), axes)) if axes == 3 else np.sin(2*np.pi*.2*t)
                key = f"{device}_{sensor.lower()}"
                covered = np.ones(len(t), dtype=bool)
                c = Channel(x, fs, covered, covered.copy())
                if sensor in ("ECG", "BVP"):
                    c.values = np.sin(2*np.pi*(10 if sensor == "ECG" else 1.5)*t)
                    c.peaks = np.arange(0, len(t), fs)
                elif sensor == "EDA":
                    c.values = 1 + .001*t
                    c.tonic = c.values.copy()
                    c.phasic = np.zeros(len(t))
                elif sensor == "Resp":
                    c.peaks = np.arange(fs, len(t), 5*fs)
                elif sensor in ("Temp", "TEMP"):
                    c.values = 30 + .01*t
                channels[key] = c
        return channels

    def test_exact_contract_no_scr_missingness_and_s2_contact_mask(self):
        channels = self.prepared()
        result, flags = window_features(channels, 60*700, 120*700, "S2", self.contract)
        self.assertEqual(set(result), set(self.contract["feature_columns"]))
        self.assertEqual(len(result), 68)
        self.assertEqual(result["wrist_eda_scr_count"], 0)
        self.assertTrue(np.isnan(result["wrist_eda_scr_amplitude_mean"]))
        self.assertIn("wrist_eda:no_scr_events", flags)
        self.assertTrue(np.isnan(result["chest_temp_mean"]))
        self.assertIn("chest_temp:suspect_contact", flags)
        self.assertAlmostEqual(result["wrist_temp_slope_per_s"], .01)
        self.assertAlmostEqual(result["chest_resp_rate_mean"], 12.)
        result, _ = window_features(channels, 60*700, 120*700, "S5", self.contract)
        self.assertTrue(np.isfinite(result["chest_temp_mean"]))

    def test_missing_or_removed_provider_fails_before_raw_loading(self):
        config = copy.deepcopy(self.config)
        config["preprocessing_provider"] = None
        with patch("wesad.features.load_subject") as loader:
            with self.assertRaisesRegex(ValueError, "preprocessing_provider"):
                process_subject(Path("S5.pkl"), config)
            loader.assert_not_called()
        config["preprocessing_provider"] = "wesad.features:prepare_subject"
        with patch("wesad.features.load_subject") as loader:
            with self.assertRaisesRegex(ValueError, "not available"):
                process_subject(Path("S5.pkl"), config)
            loader.assert_not_called()

    def test_already_calibrated_input_is_never_filtered_or_rescaled(self):
        channels = self.prepared()
        original = {k: c.values.copy() for k, c in channels.items()}
        # A second filter would both alter the input and violate ownership.
        with patch("scipy.signal.sosfiltfilt", side_effect=AssertionError("Unexpected second filter")):
            result, _ = window_features(channels, 60*700, 120*700, "S5", self.contract)
        self.assertAlmostEqual(result["wrist_acc_x_5s_mean_mean"], 1.)
        self.assertAlmostEqual(result["wrist_acc_magnitude_5s_mean_mean"], np.sqrt(3))
        for key, channel in channels.items():
            np.testing.assert_array_equal(channel.values, original[key])

    def test_subject_processor_uses_explicit_provider_once(self):
        config = copy.deepcopy(self.config)
        config["preprocessing_provider"] = "nhu.integration:prepare_subject"
        channels = self.prepared()
        data = {"label": np.ones(180*700), "signal": {"not_read_by_feature_layer": True}}
        provider = Mock(return_value=channels)
        with patch("wesad.features.import_callable", return_value=provider) as importer, \
             patch("wesad.features.load_subject", return_value=data), \
             patch("scipy.signal.sosfiltfilt", side_effect=AssertionError("Unexpected filter")):
            frame = process_subject(Path("S5.pkl"), config)
        importer.assert_called_once_with("nhu.integration:prepare_subject")
        provider.assert_called_once_with(data, config, self.contract)
        self.assertEqual(list(frame.columns), self.contract["output_column_order"])
        self.assertEqual(len(frame), 25)
        np.testing.assert_allclose(frame.wrist_acc_x_5s_mean_mean, 1.)

    def test_config_cannot_silently_change_contract_or_window_rule(self):
        read_contract(self.config)
        altered = copy.deepcopy(self.config)
        altered["feature_contract_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "contract changed"):
            read_contract(altered)
        altered = copy.deepcopy(self.config)
        altered["step_seconds"] = 60
        with self.assertRaisesRegex(ValueError, "step_seconds"):
            read_contract(altered)


if __name__ == "__main__":
    unittest.main()
