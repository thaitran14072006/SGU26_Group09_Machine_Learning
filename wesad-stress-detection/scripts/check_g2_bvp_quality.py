"""Summarize BVP missingness from checked physiological68 package directories.

This does not merge training tables or modify received packages. Each subject
must appear in just one package; metadata preserves the actual source run.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from wesad.data import SUBJECTS, load_feature_table
from wesad.features import META, read_contract
from wesad.g2_bvp_quality import summarize_bvp
from wesad.utils import file_hash, read_config, read_json, resolve_path, write_csv, write_json


def checked_package(directory, contract, config):
    directory = Path(directory)
    snapshot = read_json(directory / "config.json")
    schema = read_json(directory / "feature_schema.json")
    status = read_json(directory / "run_status.json")
    if status.get("status") != "complete":
        raise ValueError("Package is incomplete.")
    subjects = status["subjects"]
    if not subjects or len(subjects) != len(set(subjects)) or set(subjects) - set(SUBJECTS):
        raise ValueError("Invalid package subjects.")
    processing = snapshot["processing"]
    for key in ["subject_processor", "preprocessing_provider", "input_units", "feature_version",
                "feature_contract_sha256", "window_seconds", "step_seconds", "labels",
                "label_rule", "min_label_fraction"]:
        if processing.get(key) != config[key]:
            raise ValueError(f"Package differs from configured physiological definition: {key}")
    for key in ["feature_version", "feature_columns", "sensor_groups"]:
        if schema[key] != contract[key]:
            raise ValueError(f"Package schema differs: {key}")
    if schema["processing_signature"] != snapshot["signature"]:
        raise ValueError("Schema processing signature differs.")
    if file_hash(directory / "physiological_contract.json") != config["feature_contract_sha256"]:
        raise ValueError("Bundled physiological contract differs.")
    for filename in ["checksums.json", "handoff_checksums.json"]:
        for name, digest in read_json(directory / filename).items():
            if Path(name).name != name or name in {".", ".."} or "/" in name or "\\" in name or ":" in name:
                raise ValueError("Checksum entry must name a package file.")
            if file_hash(directory / name) != digest:
                raise ValueError(f"Package checksum mismatch: {name}")
    frames = []
    for subject in subjects:
        frame, columns = load_feature_table(directory / f"{subject}.csv", expected_subjects=[subject])
        manifest = read_json(directory / f"{subject}.json")
        if (list(frame) != contract["output_column_order"] or columns != contract["feature_columns"]
                or manifest["status"] != "complete" or manifest["signature"] != snapshot["signature"]
                or manifest["n_windows"] != len(frame) or manifest["feature_columns"] != columns
                or manifest["csv_sha256"] != file_hash(directory / f"{subject}.csv")):
            raise ValueError(f"{subject}: inconsistent manifest or columns.")
        if (frame[META].isna().any().any() or not frame.label_fraction.between(.9, 1).all()
                or not np.allclose(frame.end_s - frame.start_s, 60, rtol=0, atol=1e-9)
                or not np.allclose(frame.start_s / 5, np.round(frame.start_s / 5), rtol=0, atol=1e-9)
                or frame.window_id.tolist() != [f"{subject}_{int(round(t * 700))}" for t in frame.start_s]):
            raise ValueError(f"{subject}: invalid metadata/grid/keys.")
        for column in columns:
            missing = frame[column].isna()
            sensor = "_".join(column.split("_")[:2])
            if missing.any() and not frame.loc[missing, "quality_flag"].str.contains(sensor + ":", regex=False).all():
                raise ValueError(f"{subject}/{column}: missing without a sensor quality reason.")
        qa = manifest["extraction_qa"]
        if (qa["n_kept_windows"] != len(frame)
                or qa["n_missing_feature_values"] != int(frame[columns].isna().sum().sum())
                or qa["n_candidate_windows"] != len(frame) + sum(v for k, v in qa.items() if k.startswith("n_rejected_"))):
            raise ValueError(f"{subject}: QA counts differ.")
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    aggregate, _ = load_feature_table(directory / "all_subjects.csv", expected_subjects=subjects)
    pd.testing.assert_frame_equal(frame, aggregate)
    if status["n_windows"] != len(frame):
        raise ValueError("Aggregate window count differs.")
    return frame, {"package": directory.name, "subjects": subjects,
                   "code_sha256": snapshot["code_sha256"], "environment": snapshot["environment"],
                   "processing": processing, "signature": snapshot["signature"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages", nargs="+", required=True, help="Extracted package directories")
    parser.add_argument("--config", default="configs/features_g2.yaml")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = resolve_path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new output directory to preserve evidence.")
    config = read_config(args.config)
    contract = read_contract(config)
    frames, sources, seen = [], [], set()
    for package in args.packages:
        frame, source = checked_package(resolve_path(package), contract, config)
        if seen.intersection(source["subjects"]):
            raise ValueError("Subject overlap between packages; no automatic duplication or merging.")
        seen.update(source["subjects"])
        frames.append(frame)
        sources.append(source)
    combined = pd.concat(frames, ignore_index=True)
    rows = summarize_bvp(combined)
    write_csv(out / "bvp_missing_by_subject_label.csv", pd.DataFrame(rows))
    report = {"status": "passed_package_and_quality_consistency_checks", "scope": "descriptive_BVP_QA_only",
              "n_windows": len(combined), "subjects": sorted(seen, key=lambda s: int(s[1:])),
              "not_received_subjects": [s for s in SUBJECTS if s not in seen],
              "n_bvp_interval_missing": sum(r["n_bvp_interval_missing"] for r in rows),
              "n_missing_feature_values": int(combined[contract["feature_columns"]].isna().sum().sum()),
              "by_subject_label": rows, "source_packages": sources,
              "code_hashes_match": len({s["code_sha256"] for s in sources}) == 1,
              "full_environments_match": all(s["environment"] == sources[0]["environment"] for s in sources),
              "physiological_acceptance": False, "merged_training_table_created": False}
    write_json(out / "verification.json", report)
    print(f"Checked {len(seen)} subjects, {len(combined)} windows; BVP interval missing: {report['n_bvp_interval_missing']}")


if __name__ == "__main__":
    main()
