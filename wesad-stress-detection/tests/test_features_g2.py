"""Independent numerical/QA invariants for Mai's G2 physiological contract."""
import copy
import unittest
from pathlib import Path

import numpy as np

from wesad.features import (Channel, band_powers, block_features, detect_scr,
                            filter_segments, interval_features, prepare_subject,
                            read_contract, window_features)
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

    def test_filter_gaps_and_short_regions_are_not_bridged(self):
        fs = 4
        x = np.sin(np.arange(400)/fs)
        x[180:200] = np.nan
        y, valid = filter_segments(x, fs, 1, "lowpass", 10)
        self.assertFalse(valid[:40].any())
        self.assertFalse(valid[140:240].any())
        self.assertTrue(np.isnan(y[180:200]).all())
        short, mask = filter_segments(np.ones(20), fs, 1, "lowpass", 10)
        self.assertFalse(mask.any())
        self.assertTrue(np.isnan(short).all())

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

    def test_reference_adapter_converts_wrist_acc_once(self):
        data = {"label": np.ones(150*700), "signal": {"chest": {}, "wrist": {}}}
        for device, specs in SENSOR_SPECS.items():
            for sensor, (fs, axes) in specs.items():
                value = 64 if device == "wrist" and sensor == "ACC" else 1
                data["signal"][device][sensor] = np.full((150*fs, axes), value, dtype=float)
        channels = prepare_subject(data, self.config, self.contract)
        np.testing.assert_allclose(channels["wrist_acc"].values, 1.)
        np.testing.assert_allclose(data["signal"]["wrist"]["ACC"], 64.)
        self.assertFalse(channels["wrist_eda"].usable[:60*4].any())
        self.assertTrue(channels["wrist_eda"].usable[60*4:90*4].all())

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
