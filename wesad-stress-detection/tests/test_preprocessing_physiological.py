"""Provider invariants: ownership, calibration, segment masks and integration."""

import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from wesad.features import process_subject, read_contract
from wesad.multisensor import SENSOR_SPECS
from wesad.preprocessing_physiological import prepare_subject
from wesad.utils import read_config


class PhysiologicalProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = read_config("configs/features_g2.yaml")
        cls.contract = read_contract(cls.config)

    def sample(self, seconds=180):
        data = {"label": np.ones(seconds * 700, dtype=np.int8), "signal": {}}
        for device, sensors in SENSOR_SPECS.items():
            data["signal"][device] = {}
            for sensor, (rate, axes) in sensors.items():
                t = np.arange(seconds * rate) / rate
                if axes == 3:
                    values = np.tile([64., 0., 0.] if device == "wrist" else [1., 0., 0.], (len(t), 1))
                elif sensor == "ECG":
                    # One 10-Hz pulse burst per second, shaped for the ECG detector.
                    values = (np.sin(2 * np.pi * 10 * t) * np.exp(-((t % 1 - .5) / .04) ** 2))[:, None]
                elif sensor == "BVP":
                    values = np.sin(2 * np.pi * t)[:, None]
                elif sensor == "EMG":
                    values = np.sin(2 * np.pi * 50 * t)[:, None]
                elif sensor == "EDA":
                    values = (2 + .1 * np.sin(2 * np.pi * .1 * t))[:, None]
                elif sensor == "Resp":
                    values = np.sin(2 * np.pi * .2 * t)[:, None]
                else:
                    values = (30 + .01 * t)[:, None]
                data["signal"][device][sensor] = values
        return data

    def test_native_rate_calibration_once_and_input_not_modified(self):
        data = self.sample()
        channels = prepare_subject(data, self.config, self.contract)
        self.assertEqual(len(channels), 10)
        np.testing.assert_equal(data["signal"]["wrist"]["ACC"][:, 0], 64)
        np.testing.assert_equal(channels["wrist_acc"].values[:, 0], 1)
        np.testing.assert_equal(channels["chest_acc"].values[:, 0], 1)
        self.assertEqual(channels["wrist_bvp"].fs, 64)
        self.assertEqual(len(channels["wrist_bvp"].values), 180 * 64)
        self.assertTrue(channels["wrist_acc"].usable.all())

    def test_eda_60_second_margin_is_not_added_twice(self):
        channels = prepare_subject(self.sample(), self.config, self.contract)
        for name in ["chest_eda", "wrist_eda"]:
            c = channels[name]
            self.assertFalse(c.usable[:60 * c.fs].any())
            self.assertTrue(c.usable[60 * c.fs:120 * c.fs].all())
            self.assertFalse(c.usable[120 * c.fs:].any())
            np.testing.assert_allclose(c.values, c.tonic + c.phasic)

    def test_gaps_have_independent_filters_and_excluded_margins(self):
        data = self.sample(300)
        data["signal"]["wrist"]["EDA"][150 * 4] = np.nan
        channels = prepare_subject(data, self.config, self.contract)
        c = channels["wrist_eda"]
        self.assertFalse(c.covered[600])
        self.assertTrue(np.isnan(c.values[600]))
        self.assertFalse(c.usable[90 * 4:210 * 4 + 1].any())
        data["signal"]["wrist"]["EDA"][:600] *= 10
        other = prepare_subject(data, self.config, self.contract)["wrist_eda"]
        np.testing.assert_equal(c.values[601:], other.values[601:])
        np.testing.assert_equal(c.tonic[601:], other.tonic[601:])
        self.assertTrue(all(c.usable[int(a):int(b) + 1].all() for a, b, _ in c.events))

    def test_very_short_filter_segments_are_not_usable(self):
        channels = prepare_subject(self.sample(1), self.config, self.contract)
        self.assertFalse(channels["wrist_eda"].usable.any())
        self.assertFalse(channels["chest_resp"].usable.any())

    def test_synthetic_integration_preserves_window_and_68_column_contract(self):
        data = self.sample()
        with patch("wesad.features.load_subject", return_value=data):
            frame = process_subject(Path("S4.pkl"), self.config)
        self.assertEqual(list(frame.columns), self.contract["output_column_order"])
        self.assertEqual(frame.window_id.tolist(), ["S4_42000"])
        self.assertEqual(frame.label_fraction.iloc[0], 1.)
        self.assertAlmostEqual(frame.wrist_acc_x_5s_mean_mean.iloc[0], 1.)
        self.assertAlmostEqual(frame.chest_resp_rate_mean.iloc[0], 12., places=1)
        self.assertAlmostEqual(frame.wrist_bvp_pulse_rate_mean_bpm.iloc[0], 60., places=1)
        self.assertFalse(np.isinf(frame[self.contract["feature_columns"]].to_numpy()).any())

    def test_provider_rejects_undeclared_units_and_wrong_shapes(self):
        config = copy.deepcopy(self.config)
        config["input_units"] = "unknown"
        with self.assertRaisesRegex(ValueError, "official synchronized PKL"):
            prepare_subject(self.sample(), config, self.contract)
        data = self.sample()
        data["signal"]["chest"]["ACC"] = np.ones((100, 2))
        with self.assertRaisesRegex(ValueError, "chest_acc"):
            prepare_subject(data, self.config, self.contract)


if __name__ == "__main__":
    unittest.main()
