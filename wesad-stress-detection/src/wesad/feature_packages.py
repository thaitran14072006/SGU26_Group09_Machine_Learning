"""Validate/merge complete physiological packages; never silently mix versions."""
from pathlib import Path
import re

import numpy as np
import pandas as pd

from wesad.data import SUBJECTS, validate_features
from wesad.features import META, OK, VERSION
from wesad.utils import PROJECT_ROOT, file_hash, fingerprint, read_json, write_csv, write_json


def validate_contract_frame(frame, contract):
    columns = contract["feature_columns"]
    validate_features(frame, columns)
    if list(frame.columns) != contract["output_column_order"]:
        raise ValueError("Metadata/feature order or extra columns differ from contract")
    if frame[META].isna().any().any():
        raise ValueError("Metadata cannot be missing")
    if not np.allclose(frame.end_s-frame.start_s, 60, rtol=0, atol=1e-9):
        raise ValueError("Window duration differs from 60 seconds")
    if not np.allclose(frame.start_s/5, np.round(frame.start_s/5), rtol=0, atol=1e-9):
        raise ValueError("Window start is outside the 5-second grid")
    ids = [f"{s}_{int(round(t*700))}" for s, t in zip(frame.subject_id, frame.start_s)]
    if ids != frame.window_id.tolist() or not frame.label_fraction.between(.9, 1).all():
        raise ValueError("Invalid original window IDs or label purity")
    numeric = frame[columns].to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise ValueError("Infinite feature value")
    for i, (_, row) in enumerate(frame.iterrows()):
        flag = str(row.quality_flag)
        flags = set(flag.split(";"))
        if flag != OK and (flag != ";".join(sorted(flags)) or any(":" not in f for f in flags)):
            raise ValueError("Quality flags must be sorted, unique sensor:reason codes")
        for j in np.flatnonzero(np.isnan(numeric[i])):
            name = columns[j]
            sensor = "_".join(name.split("_")[:2])
            relevant = {f.split(":", 1)[1] for f in flags if f.startswith(sensor+":")}
            allowed = {"flat_or_unusable", "detector_failure"}
            if any(s in name for s in ("_hr_", "_mean_rr_", "_sd_rr_", "_rmssd", "_pnn50", "_pulse_rate_", "_mean_ibi_", "_sd_ibi_")):
                allowed.add("insufficient_or_unreliable_beats")
            if "_eda_scr_amplitude_" in name:
                allowed.add("no_scr_events")
            if name.startswith("chest_resp_"):
                allowed.add("insufficient_cycles")
            if name.startswith("chest_temp_") and row.subject_id == "S2":
                allowed.add("suspect_contact")
            if not (relevant & allowed):
                raise ValueError(f"Missing {name} lacks a permitted sensor-specific reason")
    for device in ("chest", "wrist"):
        count = frame[f"{device}_eda_scr_count"]
        rate = frame[f"{device}_eda_scr_rate_per_min"]
        if not np.allclose(count, rate, equal_nan=True):
            raise ValueError("SCR count/rate differ for a 60-second window")
        if not np.allclose(count.dropna(), np.round(count.dropna())) or (count.dropna() < 0).any():
            raise ValueError("SCR counts must be nonnegative integers")
        no_events = count.eq(0)
        amps = [f"{device}_eda_scr_amplitude_{s}" for s in ("mean", "max")]
        if not frame.loc[no_events, amps].isna().all().all():
            raise ValueError("No-SCR amplitudes must be NaN, not zero")
    s2 = frame.subject_id.eq("S2")
    if s2.any() and not frame.loc[s2, ["chest_temp_mean", "chest_temp_slope_per_s"]].isna().all().all():
        raise ValueError("S2 known suspect-contact TEMP must be explicitly missing")
    for col in columns:
        finite = frame[col].dropna()
        if any(t in col for t in ("_std", "_sd_", "_rmssd", "_power_", "_rms_", "_mav_", "_abs_integral_", "_waveform_length_")) and (finite < -1e-12).any():
            raise ValueError(f"Negative nonnegative statistic: {col}")
        if "_pnn50" in col and not finite.between(0, 1).all():
            raise ValueError("pNN50 is a fraction in [0,1]")
    return frame


def read_package(directory, contract):
    directory = Path(directory)
    snapshot, schema, status = [read_json(directory/n) for n in ("config.json", "feature_schema.json", "run_status.json")]
    if status.get("status") != "complete" or not status.get("subjects"):
        raise ValueError("Incomplete package")
    if len(set(status["subjects"])) != len(status["subjects"]) or set(status["subjects"]) - set(SUBJECTS):
        raise ValueError("Invalid/duplicate package subject list")
    if snapshot["processing"].get("feature_version") != VERSION or schema.get("feature_version") != VERSION:
        raise ValueError("Not physiological_68_v1; raw112/old trials cannot be merged")
    for key, value in {"window_seconds": 60, "step_seconds": 5, "label_rule": "majority",
                       "min_label_fraction": .9, "labels": [1, 2, 3],
                       "input_units": "official_wesad_synchronized_pkl"}.items():
        if snapshot["processing"].get(key) != value:
            raise ValueError(f"Package processing differs from contract: {key}")
    for key in ("feature_columns", "metadata_columns", "output_column_order", "features", "sensor_groups", "unit_policy", "methods"):
        if schema.get(key) != contract[key]:
            raise ValueError(f"Package contract differs: {key}")
    if schema.get("processing_signature") != snapshot["signature"]:
        raise ValueError("Schema/signature mismatch")
    expected_signature = fingerprint({"processing": snapshot["processing"], "code": snapshot["code_sha256"],
                                      "environment": snapshot["environment"]})
    if expected_signature != snapshot["signature"]:
        raise ValueError("Invalid code/config/environment signature")
    if not re.match(r"3\.12\.14(?:\s|$)", snapshot["environment"]["python"]):
        raise ValueError("G2 package requires Python 3.12.14")
    locked = dict(line.split("==") for line in (PROJECT_ROOT/"requirements.lock.txt").read_text().splitlines() if "==" in line)
    for name in ("numpy", "pandas", "PyYAML", "scipy", "scikit-learn", "imbalanced-learn"):
        if snapshot["environment"]["packages"].get(name) != locked[name]:
            raise ValueError(f"Package environment differs from locked library: {name}")
    checksums = read_json(directory/"checksums.json")
    if set(checksums) != {"all_subjects.csv", "feature_schema.json", *[f"{s}.csv" for s in status["subjects"]]}:
        raise ValueError("Missing/unexpected checksum entries")
    for name, digest in checksums.items():
        if file_hash(directory/name) != digest:
            raise ValueError(f"Checksum mismatch: {name}")
    frames, manifests = [], {}
    for subject in status["subjects"]:
        manifest = read_json(directory/f"{subject}.json")
        path = directory/f"{subject}.csv"
        if (manifest.get("status") != "complete" or manifest.get("signature") != snapshot["signature"]
                or manifest.get("csv_sha256") != file_hash(path)
                or manifest.get("feature_columns") != contract["feature_columns"]
                or not re.fullmatch(r"[0-9a-f]{64}", manifest.get("source_sha256", ""))):
            raise ValueError(f"Invalid manifest for {subject}")
        frame = pd.read_csv(path, dtype={"subject_id": str, "window_id": str}, float_precision="round_trip")
        validate_contract_frame(frame, contract)
        if set(frame.subject_id) != {subject} or len(frame) != manifest["n_windows"]:
            raise ValueError("Subject/row count differs from manifest")
        qa = manifest.get("extraction_qa", {})
        if (qa.get("n_kept_windows") != len(frame)
                or qa.get("n_candidate_windows") != qa.get("n_rejected_label", 0) + qa.get("n_label_valid_windows", 0)
                or qa.get("n_label_valid_windows") != len(frame) + sum(qa.get(k, 0) for k in
                    ("n_rejected_incomplete_signal", "n_rejected_nonfinite_signal", "n_rejected_filter_edge"))):
            raise ValueError("Invalid extraction QA accounting")
        frames.append(frame)
        manifests[subject] = manifest
    combined = pd.concat(frames, ignore_index=True)
    actual = pd.read_csv(directory/"all_subjects.csv", dtype={"subject_id": str, "window_id": str}, float_precision="round_trip")
    pd.testing.assert_frame_equal(combined, actual)
    if len(combined) != status["n_windows"]:
        raise ValueError("Package row count differs from run_status")
    return combined, snapshot, manifests


def package_report(directory, contract, *, complete=False):
    frame, snapshot, manifests = read_package(directory, contract)
    subjects = sorted(set(frame.subject_id), key=lambda s: int(s[1:]))
    missing = sorted(set(SUBJECTS)-set(subjects), key=lambda s: int(s[1:]))
    if complete and missing:
        raise ValueError(f"Missing subjects for 15-subject acceptance: {missing}")
    nan_counts = frame[contract["feature_columns"]].isna().sum()
    return {"status": "passed", "scope": "complete_15_subjects" if not missing else "selected_subjects_only",
            "feature_version": VERSION, "subjects": subjects, "missing_subjects": missing,
            "n_windows": len(frame), "n_features": len(contract["feature_columns"]),
            "feature_missing_counts": {k: int(v) for k, v in nan_counts.items() if v},
            "label_counts": {str(k): int(v) for k, v in frame.label_original.value_counts().sort_index().items()},
            "processing_signature": snapshot["signature"], "code_sha256": snapshot["code_sha256"],
            "environment": snapshot["environment"], "preprocessing_provider": snapshot["processing"].get("preprocessing_provider"),
            "adapter_status": snapshot["processing"].get("adapter_status"),
            "cross_machine_acceptance": False, "nhu_runtime_integration_confirmed": False,
            "physiological_signal_validity_established": False}


def merge_packages(directories, output, contract, *, require_complete=True):
    """Merge per-subject files only; do not append package all_subjects twice."""
    packages = [read_package(d, contract) for d in directories]
    first = packages[0][1]
    frames, manifests, qa_rows = {}, {}, []
    for frame, snapshot, subjects in packages:
        if any(snapshot[k] != first[k] for k in ("processing", "environment", "code_sha256", "signature")):
            raise ValueError("Packages differ in code/config/environment; do not merge")
        for subject, manifest in subjects.items():
            part = frame.loc[frame.subject_id.eq(subject)].reset_index(drop=True)
            if subject in frames:
                # S2 may occur in every package. Accept one only when identical.
                if manifest["source_sha256"] != manifests[subject]["source_sha256"]:
                    raise ValueError(f"Conflicting raw hash for duplicate {subject}")
                pd.testing.assert_frame_equal(part, frames[subject])
                continue
            frames[subject], manifests[subject] = part, manifest
    ordered = sorted(frames, key=lambda s: int(s[1:]))
    missing = sorted(set(SUBJECTS)-set(ordered), key=lambda s: int(s[1:]))
    if require_complete and missing:
        raise ValueError(f"Cannot declare full G2; missing {missing}")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Merge output must be new/empty; do not overwrite existing package")
    output.mkdir(parents=True, exist_ok=True)
    # All source packages/duplicates are checked before any output is written.
    combined = pd.concat([frames[s] for s in ordered], ignore_index=True)
    validate_contract_frame(combined, contract)
    for subject in ordered:
        write_csv(output/f"{subject}.csv", frames[subject])
        manifest = dict(manifests[subject], csv_sha256=file_hash(output/f"{subject}.csv"))
        write_json(output/f"{subject}.json", manifest)
        qa_rows.append({"subject_id": subject, "n_windows": len(frames[subject]), **manifest.get("extraction_qa", {})})
    write_csv(output/"all_subjects.csv", combined)
    write_csv(output/"quality_report.csv", pd.DataFrame(qa_rows))
    schema = {**contract, "processing_signature": first["signature"]}
    write_json(output/"feature_schema.json", schema)
    write_json(output/"config.json", {**first, "merged_from": [str(Path(d)) for d in directories]})
    write_json(output/"checksums.json", {n: file_hash(output/n) for n in
               ["all_subjects.csv", "feature_schema.json", *[f"{s}.csv" for s in ordered]]})
    write_json(output/"run_status.json", {"status": "complete", "subjects": ordered, "n_windows": len(combined),
               "scope": "complete_15_subjects" if not missing else "selected_subjects_only"})
    return package_report(output, contract, complete=require_complete)
