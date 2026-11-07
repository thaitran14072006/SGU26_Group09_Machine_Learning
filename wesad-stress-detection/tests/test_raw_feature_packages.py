"""Raw112 integrity/merge tests; no raw-signal processing or model training."""
import copy
from pathlib import Path
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from wesad.data import SUBJECTS
from wesad.raw_feature_packages import (FEATURES, GROUPS, META, PROCESSING, VERSION,
    audit_package, compatibility, extract_package, merge_complete)
from wesad.utils import file_hash, fingerprint, write_csv, write_json


class RawPackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def package(self, name, subjects, *, code="a"*64, changed=None, version=VERSION):
        path = self.root/name
        path.mkdir()
        processing = copy.deepcopy(PROCESSING)
        processing["feature_version"] = version
        environment = {"python": "3.12.14 fixture", "packages": {"numpy": "fixture"}}
        signature = fingerprint({"processing": processing, "code": code, "environment": environment})
        write_json(path/"config.json", {"signature": signature, "processing": processing,
                   "environment": environment, "code_sha256": code, "git_commit": None, "git_dirty": True})
        write_json(path/"feature_schema.json", {"feature_version": version, "feature_columns": FEATURES,
                   "sensor_groups": GROUPS, "processing_signature": signature})
        frames, qa = [], []
        for subject in subjects:
            row = dict(zip(META, [subject, f"{subject}_0", 0., 60., 1, 0, 1., "finite_only_not_physiological_QA"]))
            row.update({c: 1. for c in FEATURES})
            if changed:
                row.update(changed)
            frame = pd.DataFrame([row], columns=META+FEATURES)
            frames.append(frame)
            write_csv(path/f"{subject}.csv", frame)
            write_json(path/f"{subject}.json", {"status": "complete", "signature": signature,
                       "feature_columns": FEATURES, "n_windows": 1,
                       "source_sha256": fingerprint(subject), "csv_sha256": file_hash(path/f"{subject}.csv")})
            qa.append({"subject_id": subject, "n_windows": 1, "n_label_1": 1, "n_label_2": 0, "n_label_3": 0})
        write_csv(path/"all_subjects.csv", pd.concat(frames, ignore_index=True))
        write_csv(path/"quality_report.csv", pd.DataFrame(qa))
        write_json(path/"run_status.json", {"status": "complete", "subjects": subjects, "n_windows": len(subjects)})
        write_json(path/"checksums.json", {n: file_hash(path/n) for n in
                   ["all_subjects.csv", "feature_schema.json", *[f"{s}.csv" for s in subjects]]})
        return path

    def test_valid_package_and_corrupt_csv_checksum(self):
        path = self.package("a", ["S4", "S9"])
        report = audit_package(path).report
        self.assertEqual(report["n_windows"], 2)
        with (path/"S4.csv").open("a") as stream:
            stream.write("bad\n")
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            audit_package(path)

    def test_nan_rejected_even_when_all_checksums_are_consistent(self):
        path = self.package("a", ["S2"], changed={FEATURES[0]: float("nan")})
        with self.assertRaisesRegex(ValueError, "NaN"):
            audit_package(path)

    def test_version_and_provenance_mismatch_do_not_merge(self):
        wrong = self.package("wrong", ["S2"], version="physiological_68_v1")
        with self.assertRaisesRegex(ValueError, "Different feature version"):
            audit_package(wrong)
        a = audit_package(self.package("a", ["S2"]))
        b = audit_package(self.package("b", ["S4"], code="b"*64))
        self.assertFalse(compatibility(a, b)["eligible_for_strict_merge"])
        with self.assertRaisesRegex(ValueError, "Different code/config/environment"):
            merge_complete([a, b], self.root/"out")
        self.assertFalse((self.root/"out").exists())

    def test_incomplete_and_conflicting_duplicate_never_write_output(self):
        a = audit_package(self.package("a", ["S2"]))
        with self.assertRaisesRegex(ValueError, "missing"):
            merge_complete([a], self.root/"missing")
        b = audit_package(self.package("b", ["S2"], changed={FEATURES[0]: 2.}))
        with self.assertRaisesRegex(ValueError, "feature values differ"):
            merge_complete([a, b], self.root/"conflict")
        self.assertFalse((self.root/"missing").exists())
        self.assertFalse((self.root/"conflict").exists())

    def test_complete_merge_keeps_s2_once_and_not_all_subjects_twice(self):
        groups = [["S2", "S7", "S13", "S15"], ["S3", "S8", "S14", "S16"],
                  ["S4", "S9", "S11", "S17"], ["S2", "S5", "S6", "S10"]]
        packages = [audit_package(self.package(str(i), subjects)) for i, subjects in enumerate(groups)]
        result = merge_complete(packages, self.root/"complete")
        self.assertEqual(set(result.frames), set(SUBJECTS))
        self.assertEqual(result.report["n_windows"], 15)
        actual = pd.read_csv(self.root/"complete/all_subjects.csv")
        self.assertEqual(actual.subject_id.eq("S2").sum(), 1)
        self.assertFalse(actual.duplicated(["subject_id", "window_id"]).any())

    def test_unsafe_zip_is_rejected_before_extraction(self):
        archive = self.root/"bad.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            stream.writestr("../outside.txt", "bad")
        with self.assertRaisesRegex(ValueError, "Unsafe archive"):
            extract_package(archive, self.root/"unpacked")
        self.assertFalse((self.root/"unpacked").exists())
        self.assertFalse((self.root/"outside.txt").exists())

    def test_rar_members_written_as_regular_files_with_checked_names(self):
        listing = "\n".join(f"Name: package/{n}\nType: File\nSize: 2\n" for n in ("config.json", "run_status.json"))
        outputs = [SimpleNamespace(stdout=listing), SimpleNamespace(stdout=b"{}"), SimpleNamespace(stdout=b"{}")]
        with patch("wesad.raw_feature_packages.shutil.which", return_value="unrar"), \
             patch("wesad.raw_feature_packages.subprocess.run", side_effect=outputs) as runner:
            folder = extract_package(self.root/"input.rar", self.root/"unpacked")
        self.assertEqual((folder/"config.json").read_text(), "{}")
        self.assertEqual(runner.call_args_list[1].args[0][1], "p")

    def test_unsafe_rar_path_rejected_without_creating_destination(self):
        listing = "Name: ../outside.txt\nType: File\nSize: 3\n"
        with patch("wesad.raw_feature_packages.shutil.which", return_value="unrar"), \
             patch("wesad.raw_feature_packages.subprocess.run", return_value=SimpleNamespace(stdout=listing)):
            with self.assertRaisesRegex(ValueError, "Unsafe archive"):
                extract_package(self.root/"bad.rar", self.root/"unpacked")
        self.assertFalse((self.root/"unpacked").exists())


if __name__ == "__main__":
    unittest.main()
