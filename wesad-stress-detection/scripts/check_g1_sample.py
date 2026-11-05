"""Validate a G1 raw-stat S2 package; optionally compare another machine's CSV."""

import argparse
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from wesad.data import load_feature_table
from wesad.multisensor import SENSOR_SPECS, STATS, channel_names
from wesad.utils import file_hash, read_json, resolve_path, write_json

METADATA = ["subject_id", "window_id", "start_s", "end_s", "label_original",
            "label_binary", "label_fraction", "quality_flag"]


def expected_feature_columns():
    return [f"{name}_{stat}" for device, specs in SENSOR_SPECS.items()
            for sensor, (_, axes) in specs.items()
            for name in channel_names(device, sensor, axes) for stat in STATS]


def expected_sensor_groups(features):
    return {"chest": [c for c in features if c.startswith("chest_")],
            "wrist": [c for c in features if c.startswith("wrist_")], "combined": features}


def read_sample(directory):
    directory = Path(directory)
    snapshot = read_json(directory / "config.json")
    manifest = read_json(directory / "S2.json")
    schema = read_json(directory / "feature_schema.json")
    status = read_json(directory / "run_status.json")
    frame, features = load_feature_table(directory / "S2.csv", expected_subjects=["S2"])
    settings = snapshot["processing"]
    expected = {"device": "combined", "sampling_rate_hz": 700, "window_seconds": 60,
                "step_seconds": 5, "label_rule": "majority", "min_label_fraction": .9}
    if any(settings.get(key) != value for key, value in expected.items()):
        raise ValueError("Package is not the agreed G1 combined raw-stat 60/5 majority-90% sample.")
    if (settings.get("sensors") != {device: list(specs) for device, specs in SENSOR_SPECS.items()}
            or settings.get("labels") != [1, 2, 3]
            or settings.get("subject_processor") != "wesad.multisensor:process_subject"
            or settings.get("feature_version") != "all_sensors_raw_stats_v1"):
        raise ValueError("G1 requires all 112 combined raw-stat features and the shared multisensor processor.")
    if (status.get("status") != "complete" or "S2" not in status.get("subjects", [])
            or manifest.get("status") != "complete"
            or manifest.get("n_windows") != len(frame)
            or manifest.get("csv_sha256") != file_hash(directory / "S2.csv")
            or manifest.get("signature") != snapshot.get("signature")
            or schema.get("processing_signature") != snapshot.get("signature")
            or manifest.get("feature_columns") != features):
        raise ValueError("Incomplete or inconsistent package/manifest/checksum.")
    expected_features = expected_feature_columns()
    if list(frame.columns) != METADATA + expected_features or features != expected_features:
        raise ValueError("Metadata or feature names/order differ from the G1 raw-stat schema.")
    if (schema.get("sensor_groups") != expected_sensor_groups(features)
            or schema.get("feature_version") != "all_sensors_raw_stats_v1"):
        raise ValueError("Generated schema must identify 63 chest and 49 wrist features in canonical order.")
    if (frame[METADATA].isna().any().any() or not np.isfinite(frame[features].to_numpy()).all()
            or not np.allclose(frame.end_s - frame.start_s, 60, rtol=0, atol=1e-9)
            or not np.allclose(frame.start_s / 5, np.round(frame.start_s / 5), rtol=0, atol=1e-9)
            or not frame.label_fraction.between(.9, 1).all()):
        raise ValueError("Invalid window grid, duration, label purity, metadata or nonfinite features.")
    expected_ids = [f"S2_{int(round(t * 700))}" for t in frame.start_s]
    if list(frame.window_id) != expected_ids:
        raise ValueError("window_id must use the original 700-Hz label start index.")
    python_match = re.match(r"\d+\.\d+\.\d+", snapshot["environment"]["python"])
    if not python_match:
        raise ValueError("Missing Python patch version in environment snapshot.")
    identity = {"source_sha256": manifest["source_sha256"], "code_sha256": snapshot["code_sha256"],
                "processing": settings, "python": python_match.group(),
                "packages": snapshot["environment"]["packages"]}
    if not all(isinstance(identity[key], str) and re.fullmatch(r"[0-9a-f]{64}", identity[key])
               for key in ("source_sha256", "code_sha256")):
        raise ValueError("Missing raw/source hashes.")
    return frame, features, identity


def check_sample(directory, reference=None, *, rtol=1e-7, atol=1e-9):
    if not np.isfinite([rtol, atol]).all() or min(rtol, atol) < 0:
        raise ValueError("Tolerances must be finite and nonnegative.")
    frame, features, identity = read_sample(directory)
    target_python = (Path(__file__).resolve().parents[1] / ".python-version").read_text().strip()
    if identity["python"] != target_python:
        raise ValueError(f"G1 target Python is {target_python}; package used {identity['python']}.")
    result = {"status": "passed", "scope": "g1_raw_stats_single_package", "identity": identity,
              "n_windows": len(frame), "n_features": len(features),
              "sensor_group_counts": {"chest": 63, "wrist": 49, "combined": 112},
              "label_counts": {str(k): int(v) for k, v in frame.label_original.value_counts().sort_index().items()},
              "rtol": rtol, "atol": atol, "four_machine_acceptance": False}
    if reference is not None:
        if Path(directory).resolve() == Path(reference).resolve():
            raise ValueError("Reference must be a separate package; self-comparison is not acceptance.")
        other, other_features, other_identity = read_sample(reference)
        if identity != other_identity or features != other_features:
            raise ValueError("Reference raw/source/config/Python/package versions or feature order differ.")
        left, right = frame.set_index("window_id").sort_index(), other.set_index("window_id").sort_index()
        if not left.index.equals(right.index):
            raise ValueError("Reference window keys differ.")
        exact = [c for c in METADATA if c != "window_id"]
        if not left[exact].equals(right[exact]):
            raise ValueError("Reference labels, times, quality flags or metadata differ.")
        a, b = left[features].to_numpy(), right[features].to_numpy()
        if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=False):
            raise ValueError("Reference raw-stat values exceed the declared tolerances.")
        result.update(scope="g1_raw_stats_pairwise_packages", max_abs_feature_difference=float(np.abs(a - b).max()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--reference-dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument("--atol", type=float, default=1e-9)
    args = parser.parse_args()
    try:
        result = check_sample(resolve_path(args.features_dir),
                              resolve_path(args.reference_dir) if args.reference_dir else None,
                              rtol=args.rtol, atol=args.atol)
    except (ValueError, KeyError, OSError) as exc:
        write_json(resolve_path(args.output), {"status": "failed", "error": str(exc),
                   "four_machine_acceptance": False})
        print(str(exc), file=sys.stderr)
        return 1
    write_json(resolve_path(args.output), result)
    print(f"G1 package check passed: {result['n_windows']} windows; {result['scope']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
