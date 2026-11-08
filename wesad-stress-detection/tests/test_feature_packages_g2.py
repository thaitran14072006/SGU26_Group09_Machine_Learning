"""Package integrity, merge deduplication and refusal of incomplete 15-subject claims."""
from pathlib import Path
import tempfile
import unittest

import pandas as pd

import test_features_g2 as feature_tests
from wesad.feature_packages import merge_packages, read_package, validate_contract_frame
from wesad.features import META, window_features
from wesad.utils import environment, file_hash, fingerprint, read_config, read_json, resolve_path, write_csv, write_json


class G2PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.contract = read_json(resolve_path("configs/feature_schema.json"))
        helper = feature_tests.PhysiologicalFeatureTests()
        helper.contract = self.contract
        values, flags = window_features(helper.prepared(), 60*700, 120*700, "S2", self.contract)
        row = dict(zip(META, ["S2", "S2_42000", 60., 120., 1, 0, 1., ";".join(sorted(flags))]))
        row.update(values)
        self.frame = pd.DataFrame([row], columns=self.contract["output_column_order"])

    def package(self, name, frame=None, environment_tag="same"):
        directory = self.root/name
        directory.mkdir()
        frame = self.frame.copy() if frame is None else frame
        config = read_config("configs/features_g2.yaml")
        processing = {k: v for k, v in config.items() if k not in ("data_root", "subjects", "output_dir")}
        env = environment()
        env["python"] += " " + environment_tag
        code = "a"*64
        signature = fingerprint({"processing": processing, "environment": env, "code": code})
        for name in ("S2.csv", "all_subjects.csv"):
            write_csv(directory/name, frame)
        write_json(directory/"config.json", {"processing": processing, "environment": env,
                   "code_sha256": code, "signature": signature})
        write_json(directory/"feature_schema.json", {**self.contract, "processing_signature": signature})
        write_json(directory/"S2.json", {"status": "complete", "signature": signature,
                   "source_sha256": "b"*64, "csv_sha256": file_hash(directory/"S2.csv"),
                   "n_windows": 1, "feature_columns": self.contract["feature_columns"],
                   "extraction_qa": {"n_candidate_windows": 1, "n_label_valid_windows": 1,
                                     "n_rejected_label": 0, "n_kept_windows": 1}})
        write_json(directory/"checksums.json", {n: file_hash(directory/n) for n in
                   ("S2.csv", "all_subjects.csv", "feature_schema.json")})
        write_json(directory/"run_status.json", {"status": "complete", "subjects": ["S2"], "n_windows": 1})
        return directory

    def test_missing_values_require_specific_reasons(self):
        validate_contract_frame(self.frame, self.contract)
        changed = self.frame.copy()
        changed.loc[0, "wrist_acc_x_5s_mean_mean"] = float("nan")
        with self.assertRaisesRegex(ValueError, "lacks a permitted"):
            validate_contract_frame(changed, self.contract)
        changed = self.frame.copy()
        changed.loc[0, "wrist_eda_scr_amplitude_mean"] = 0.
        with self.assertRaisesRegex(ValueError, "No-SCR amplitudes"):
            validate_contract_frame(changed, self.contract)

    def test_changed_csv_is_not_accepted(self):
        package = self.package("a")
        read_package(package, self.contract)
        with (package/"S2.csv").open("a") as stream:
            stream.write("broken\n")
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            read_package(package, self.contract)

    def test_identical_s2_deduplicated_and_full_claim_refused(self):
        a, b = self.package("a"), self.package("b")
        output = self.root/"out"
        with self.assertRaisesRegex(ValueError, "missing"):
            merge_packages([a, b], output, self.contract)
        self.assertFalse(output.exists())
        report = merge_packages([a, b], output, self.contract, require_complete=False)
        self.assertEqual(report["n_windows"], 1)
        self.assertEqual(len(report["missing_subjects"]), 14)
        self.assertEqual(report["scope"], "selected_subjects_only")

    def test_environment_or_duplicate_values_cannot_be_mixed(self):
        a = self.package("a")
        b = self.package("b", environment_tag="different")
        with self.assertRaisesRegex(ValueError, "code/config/environment"):
            merge_packages([a, b], self.root/"mixed", self.contract, require_complete=False)
        changed = self.frame.copy()
        changed.loc[0, "wrist_temp_mean"] += 1
        c = self.package("c", changed)
        with self.assertRaises(AssertionError):
            merge_packages([a, c], self.root/"conflict", self.contract, require_complete=False)
        self.assertFalse((self.root/"conflict").exists())


if __name__ == "__main__":
    unittest.main()
