"""Missingness reports must preserve groups and expose inconsistent flags."""

import unittest

import numpy as np
import pandas as pd

from wesad.g2_bvp_quality import BVP_INTERVAL_COLUMNS, summarize_bvp


class BvpQualityTests(unittest.TestCase):
    def frame(self):
        frame = pd.DataFrame({"subject_id": ["S4"] * 3, "window_id": ["a", "b", "c"],
                              "label_original": [1, 1, 2], "quality_flag": ["ok", "wrist_bvp:insufficient_or_unreliable_beats", "wrist_bvp:flat_or_unusable"]})
        for name in BVP_INTERVAL_COLUMNS:
            frame[name] = [1., np.nan, np.nan]
        return frame

    def test_subject_label_counts_include_all_missing_group(self):
        rows = summarize_bvp(self.frame())
        self.assertEqual(rows[0]["n_windows"], 2)
        self.assertEqual(rows[0]["bvp_interval_missing_fraction"], .5)
        self.assertEqual(rows[1]["n_bvp_interval_available"], 0)
        self.assertEqual(rows[1]["bvp_interval_missing_fraction"], 1.)

    def test_missing_requires_exact_reason_token(self):
        frame = self.frame()
        frame.loc[1, "quality_flag"] = "not_wrist_bvp:insufficient_or_unreliable_beats"
        with self.assertRaisesRegex(ValueError, "quality reasons"):
            summarize_bvp(frame)

    def test_partial_interval_group_is_rejected(self):
        frame = self.frame()
        frame.loc[1, BVP_INTERVAL_COLUMNS[0]] = 60.
        with self.assertRaisesRegex(ValueError, "together"):
            summarize_bvp(frame)

    def test_duplicate_keys_and_infinity_are_rejected(self):
        frame = self.frame()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize_bvp(pd.concat([frame, frame.iloc[:1]], ignore_index=True))
        frame.loc[0, BVP_INTERVAL_COLUMNS[0]] = np.inf
        with self.assertRaisesRegex(ValueError, "Inf"):
            summarize_bvp(frame)


if __name__ == "__main__":
    unittest.main()
