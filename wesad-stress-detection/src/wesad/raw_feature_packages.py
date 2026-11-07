"""Basic raw112 package audit/merge. Does not apply physiological preprocessing."""
import csv
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import zipfile

import numpy as np
import pandas as pd

from wesad.data import SUBJECTS, validate_features
from wesad.multisensor import SENSOR_SPECS, STATS, channel_names
from wesad.utils import file_hash, fingerprint, read_json, write_csv, write_json

VERSION = "all_sensors_raw_stats_v1"
META = ["subject_id", "window_id", "start_s", "end_s", "label_original",
        "label_binary", "label_fraction", "quality_flag"]
FEATURES = [f"{name}_{metric}" for device, specs in SENSOR_SPECS.items()
            for sensor, (_, axes) in specs.items()
            for name in channel_names(device, sensor, axes) for metric in STATS]
GROUPS = {"chest": FEATURES[:63], "wrist": FEATURES[63:], "combined": FEATURES}
PROCESSING = {"device": "combined", "sampling_rate_hz": 700,
              "sensors": {d: list(s) for d, s in SENSOR_SPECS.items()},
              "subject_processor": "wesad.multisensor:process_subject", "labels": [1, 2, 3],
              "window_seconds": 60, "step_seconds": 5, "label_rule": "majority",
              "min_label_fraction": .90, "feature_version": VERSION}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def extract_package(archive, destination):
    """Extract to a new directory after checking all member paths and sizes."""
    if Path(archive).suffix.lower() == ".rar":
        return extract_rar_package(archive, destination)
    destination = Path(destination).resolve()
    require(not destination.exists(), "Extraction destination must be new")
    with zipfile.ZipFile(archive) as stream:
        entries, seen = [], set()
        require(sum(i.file_size for i in stream.infolist()) <= 512*1024**2, "Archive too large for a basic feature package")
        for info in stream.infolist():
            name = info.filename.replace("\\", "/")
            relative = PurePosixPath(name)
            require(not relative.is_absolute() and ".." not in relative.parts and ":" not in name,
                    "Unsafe archive member path")
            target = destination.joinpath(*relative.parts).resolve()
            require(target.is_relative_to(destination), "Archive member leaves destination")
            require(not stat.S_ISLNK(info.external_attr >> 16), "Linked archive members are unsupported")
            require(name.rstrip("/").casefold() not in seen, "Duplicate archive member")
            seen.add(name.rstrip("/").casefold())
            entries.append((info, target))
        require(entries, "Empty archive")
        destination.mkdir(parents=True)
        for info, target in entries:
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with stream.open(info) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
    roots = [p.parent for p in destination.rglob("config.json")
             if (p.parent/"run_status.json").exists()]
    require(len(roots) == 1, "ZIP must contain exactly one feature package")
    return roots[0]


def extract_rar_package(archive, destination):
    """Read RAR members to regular files; UnRAR never chooses output paths."""
    executable = shutil.which("unrar") or shutil.which("UnRAR")
    if not executable:
        installed = Path("C:/Program Files/WinRAR/UnRAR.exe")
        require(installed.is_file(), "RAR input requires an installed UnRAR CLI")
        executable = str(installed)
    destination = Path(destination).resolve()
    require(not destination.exists(), "Extraction destination must be new")
    listing = subprocess.run([executable, "lt", "-c-", str(archive)], check=True,
                             capture_output=True, text=True).stdout
    entries, seen = [], set()
    for block in re.split(r"(?m)^\s*Name: ", listing)[1:]:
        name = block.splitlines()[0].strip()
        kind = re.search(r"(?m)^\s*Type:\s*(.+)$", block)
        size = re.search(r"(?m)^\s*Size:\s*(\d+)\s*$", block)
        require(kind is not None and kind.group(1).strip() in ("File", "Directory"), "Unsupported RAR member type")
        relative = PurePosixPath(name.replace("\\", "/"))
        require(not relative.is_absolute() and ".." not in relative.parts
                and not any(c in name for c in (":", "*", "?")), "Unsafe archive member path")
        target = destination.joinpath(*relative.parts).resolve()
        require(target.is_relative_to(destination), "RAR member leaves destination")
        key = str(relative).casefold()
        require(key not in seen, "Duplicate RAR member")
        seen.add(key)
        is_file = kind.group(1).strip() == "File"
        require(not is_file or size is not None, "Missing RAR member size")
        entries.append((name, target, is_file, int(size.group(1)) if size else 0))
    require(entries and sum(item[3] for item in entries) <= 512*1024**2, "Empty/oversized RAR feature package")
    destination.mkdir(parents=True)
    for name, target, is_file, size in entries:
        if not is_file:
            target.mkdir(parents=True, exist_ok=True)
            continue
        result = subprocess.run([executable, "p", "-inul", str(archive), name], check=True, capture_output=True)
        require(len(result.stdout) == size, "RAR member length differs from listing")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as output:
            output.write(result.stdout)
    roots = [p.parent for p in destination.rglob("config.json") if (p.parent/"run_status.json").exists()]
    require(len(roots) == 1, "RAR must contain exactly one feature package")
    return roots[0]


def read_table(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    require(header == META + FEATURES, f"Wrong/missing/duplicate columns: {Path(path).name}")
    return pd.read_csv(path, dtype={"subject_id": str, "window_id": str, "quality_flag": str},
                       float_precision="round_trip")


def validate_table(frame, subject=None):
    validate_features(frame, FEATURES, [subject] if subject else None)
    require(list(frame.columns) == META + FEATURES, "Raw112 column order differs")
    require(not frame.isna().any().any(), "Raw112 metadata/features contain NaN")
    numeric = frame[[c for c in frame if c not in ("subject_id", "window_id", "quality_flag")]]
    require(np.isfinite(numeric.to_numpy(dtype=float)).all(), "Raw112 contains Inf/nonfinite values")
    require(np.allclose(frame.end_s-frame.start_s, 60, rtol=0, atol=1e-9), "Invalid duration")
    require(np.allclose(frame.start_s/5, np.round(frame.start_s/5), rtol=0, atol=1e-9), "Invalid 5-second grid")
    require(frame.label_fraction.between(.9, 1).all(), "Invalid label majority fraction")
    ids = [f"{s}_{int(round(t*700))}" for s, t in zip(frame.subject_id, frame.start_s)]
    require(ids == frame.window_id.tolist(), "window_id does not match original label-grid index")
    require(frame.quality_flag.eq("finite_only_not_physiological_QA").all(),
            "Package claims preprocessing/quality different from raw112")
    for _, part in frame.groupby("subject_id", sort=False):
        require(part.start_s.is_monotonic_increasing, "Subject windows not in increasing time order")


def compare_subject(left, right, rtol=1e-7, atol=1e-9):
    a = left.sort_values("window_id").reset_index(drop=True)
    b = right.sort_values("window_id").reset_index(drop=True)
    require(a[META].equals(b[META]), "Duplicate subject metadata/keys differ")
    x, y = a[FEATURES].to_numpy(), b[FEATURES].to_numpy()
    require(np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=False), "Duplicate subject feature values differ")
    return float(np.max(np.abs(x-y)))


@dataclass
class Package:
    path: Path
    snapshot: dict
    schema: dict
    frames: dict
    manifests: dict
    quality: pd.DataFrame
    report: dict


def audit_package(directory, raw_root=None):
    directory = Path(directory).resolve()
    snapshot, schema, status, checksums = [read_json(directory/n) for n in
        ("config.json", "feature_schema.json", "run_status.json", "checksums.json")]
    processing = snapshot["processing"]
    require(processing.get("feature_version") == VERSION, "Different feature version: raw112 required")
    for key, expected in PROCESSING.items():
        require(processing.get(key) == expected, f"Different raw112 feature definition/config: {key}")
    extra = set(processing) - set(PROCESSING) - {"seed"}
    require(not extra, f"Unknown processing settings require definition review: {sorted(extra)}")
    if "seed" in processing:
        require(processing["seed"] == 42, "Different seed")
    require(schema.get("feature_version") == VERSION and schema.get("feature_columns") == FEATURES
            and schema.get("sensor_groups") == GROUPS, "Generated schema does not match canonical raw112")
    require(not set(schema)-{"feature_version", "feature_columns", "sensor_groups", "description", "processing_signature"},
            "Unexpected feature-definition metadata in raw schema; review before merging")
    require(schema.get("processing_signature") == snapshot["signature"], "Schema/signature mismatch")
    require(re.fullmatch(r"[0-9a-f]{64}", snapshot["code_sha256"]) is not None, "Invalid source code hash")
    require(fingerprint({"processing": processing, "code": snapshot["code_sha256"],
                        "environment": snapshot["environment"]}) == snapshot["signature"], "Invalid processing signature")
    require(status.get("status") == "complete", "Run did not complete")
    subjects = status["subjects"]
    require(subjects and len(set(subjects)) == len(subjects) and not set(subjects)-set(SUBJECTS), "Invalid subject list")
    disk_csv = {p.stem for p in directory.glob("S*.csv") if re.fullmatch(r"S\d+", p.stem)}
    disk_json = {p.stem for p in directory.glob("S*.json") if re.fullmatch(r"S\d+", p.stem)}
    require(disk_csv == disk_json == set(subjects), "Unlisted or missing subject CSV/manifests")
    required_hashes = {"all_subjects.csv", "feature_schema.json", *[f"{s}.csv" for s in subjects]}
    allowed_files = required_hashes | {"config.json", "run_status.json", "quality_report.csv", "run.log", *[f"{s}.json" for s in subjects]}
    require(required_hashes <= set(checksums) <= allowed_files, "Missing/unexpected checksum entries")
    for name, digest in checksums.items():
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None, "Invalid checksum digest")
        require(file_hash(directory/name) == digest, f"Checksum mismatch: {name}")
    quality = pd.read_csv(directory/"quality_report.csv", dtype={"subject_id": str})
    require(set(quality.subject_id) == set(subjects) and not quality.subject_id.duplicated().any(), "Invalid quality report subjects")
    frames, manifests, summaries = {}, {}, []
    for subject in subjects:
        path = directory/f"{subject}.csv"
        frame, manifest = read_table(path), read_json(directory/f"{subject}.json")
        validate_table(frame, subject)
        require(manifest.get("status") == "complete" and manifest.get("signature") == snapshot["signature"], f"Invalid {subject} manifest status/signature")
        require(manifest.get("feature_columns") == FEATURES and manifest.get("n_windows") == len(frame), f"Invalid {subject} manifest schema/count")
        require(manifest.get("csv_sha256") == file_hash(path), f"Invalid {subject} CSV hash")
        require(re.fullmatch(r"[0-9a-f]{64}", manifest.get("source_sha256", "")) is not None, f"Missing {subject} raw hash")
        raw_path = Path(raw_root)/subject/f"{subject}.pkl" if raw_root else None
        verified_raw = bool(raw_path and raw_path.is_file())
        if verified_raw:
            require(file_hash(raw_path) == manifest["source_sha256"], f"{subject} local raw hash differs")
        row = quality.loc[quality.subject_id.eq(subject)].iloc[0]
        require(int(row.n_windows) == len(frame), f"{subject} quality count differs")
        counts = {str(k): int(v) for k, v in frame.label_original.value_counts().sort_index().items()}
        for label in (1, 2, 3):
            require(f"n_label_{label}" in row and int(row[f"n_label_{label}"]) == counts.get(str(label), 0), f"{subject} quality label counts differ")
        qa = manifest.get("extraction_qa")
        if qa is not None:
            rejected = sum(qa.get(k, 0) for k in ("n_rejected_incomplete_signal", "n_rejected_nonfinite_signal"))
            require(qa["n_kept_windows"] == len(frame)
                    and qa["n_candidate_windows"] == qa["n_label_valid_windows"]+qa["n_rejected_label"]
                    and qa["n_label_valid_windows"] == len(frame)+rejected, f"{subject} QA accounting differs")
        frames[subject], manifests[subject] = frame, manifest
        summaries.append({"subject_id": subject, "n_windows": len(frame), "label_counts": counts,
                          "nan_values": 0, "inf_values": 0, "duplicate_keys": 0,
                          "csv_sha256": manifest["csv_sha256"], "source_sha256": manifest["source_sha256"],
                          "local_raw_hash_verified": verified_raw})
    # all_subjects is verified against the subject tables, never appended as a second input.
    merged = read_table(directory/"all_subjects.csv")
    validate_table(merged)
    require(set(merged.subject_id) == set(subjects), "Merged table has wrong subjects")
    require(len(merged) == sum(len(f) for f in frames.values()) == status["n_windows"], "Total window count differs")
    for subject in subjects:
        compare_subject(frames[subject], merged.loc[merged.subject_id.eq(subject)])
    report = {"status": "valid", "path": str(directory), "feature_version": VERSION,
              "n_features": 112, "n_metadata": 8, "sensor_groups": {k: len(v) for k, v in GROUPS.items()},
              "subjects": summaries, "n_windows": len(merged),
              "label_counts": {str(k): int(v) for k, v in merged.label_original.value_counts().sort_index().items()},
              "processing": processing, "signature": snapshot["signature"], "code_sha256": snapshot["code_sha256"],
              "environment": snapshot["environment"], "git_commit": snapshot.get("git_commit"),
              "git_dirty": snapshot.get("git_dirty"), "checksums_verified": sorted(checksums),
              "scope": "raw112_no_filtering_no_5s_acc_emg_aggregation"}
    return Package(directory, snapshot, schema, frames, manifests, quality, report)


def compatibility(reference, candidate):
    reasons = []
    for key in ("processing", "code_sha256", "environment", "signature"):
        if reference.snapshot[key] != candidate.snapshot[key]:
            reasons.append(key)
    overlaps = []
    for subject in sorted(set(reference.frames)&set(candidate.frames), key=lambda s: int(s[1:])):
        require(reference.manifests[subject]["source_sha256"] == candidate.manifests[subject]["source_sha256"], f"Duplicate {subject} raw hashes differ")
        overlaps.append({"subject_id": subject, "max_abs_feature_difference": compare_subject(reference.frames[subject], candidate.frames[subject])})
    return {"eligible_for_strict_merge": not reasons, "different_fields": reasons, "overlapping_subjects": overlaps,
            "definition_metadata_matches": reference.snapshot["processing"] == candidate.snapshot["processing"]}


def select_subjects(packages):
    require(bool(packages), "No valid compatible packages")
    selected, duplicates = {}, []
    reference = packages[0]
    for package in packages:
        result = compatibility(reference, package)
        require(result["eligible_for_strict_merge"], "Different code/config/environment: do not mix packages")
        for subject, frame in package.frames.items():
            if subject in selected:
                original = selected[subject]
                require(original.manifests[subject]["source_sha256"] == package.manifests[subject]["source_sha256"], f"Conflicting {subject} raw hash")
                delta = compare_subject(original.frames[subject], frame)
                duplicates.append({"subject_id": subject, "kept_package": str(original.path),
                                   "duplicate_package": str(package.path), "max_abs_feature_difference": delta})
            else:
                selected[subject] = package
    return selected, duplicates


def merge_complete(packages, destination):
    selected, duplicates = select_subjects(packages)
    missing = sorted(set(SUBJECTS)-set(selected), key=lambda s: int(s[1:]))
    require(not missing, f"Cannot merge full 15-subject dataset; missing {missing}")
    destination = Path(destination)
    require(not destination.exists(), "Merge destination must be new; existing data preserved")
    destination.mkdir(parents=True)
    subjects = list(SUBJECTS)
    reference = packages[0]
    # Copy each original subject once, preserving CSV and original manifest bytes.
    for subject in subjects:
        for extension in ("csv", "json"):
            shutil.copyfile(selected[subject].path/f"{subject}.{extension}", destination/f"{subject}.{extension}")
    frame = pd.concat([selected[s].frames[s] for s in subjects], ignore_index=True)
    validate_table(frame)
    write_csv(destination/"all_subjects.csv", frame)
    quality = pd.concat([selected[s].quality.loc[selected[s].quality.subject_id.eq(s)] for s in subjects], ignore_index=True)
    write_csv(destination/"quality_report.csv", quality)
    write_json(destination/"feature_schema.json", reference.schema)
    write_json(destination/"config.json", {**reference.snapshot,
        "merge_provenance": {"sources": [p.report for p in packages], "duplicate_subjects_not_appended": duplicates}})
    write_json(destination/"checksums.json", {n: file_hash(destination/n) for n in
        ["all_subjects.csv", "feature_schema.json", *[f"{s}.csv" for s in subjects]]})
    write_json(destination/"run_status.json", {"status": "complete", "subjects": subjects,
        "n_windows": len(frame), "scope": "complete_15_subjects"})
    return audit_package(destination)
