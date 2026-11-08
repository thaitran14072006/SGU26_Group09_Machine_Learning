"""Nhu G3 preparation: one outer subject only, never a final 15-subject result."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wesad.evaluation import run_experiment
from wesad.data import load_feature_table
from wesad.utils import read_config, read_json, resolve_path, write_json


def check_input(features, schema=None):
    """Reject raw112 or another physiological definition before starting fits."""
    path = resolve_path(features)
    schema_path = resolve_path(schema) if schema else path.parent / "feature_schema.json"
    supplied = read_json(schema_path)
    contract = read_json(resolve_path("configs/feature_schema.json"))
    if (supplied.get("feature_version") != "physiological_68_v1"
            or supplied.get("feature_columns") != contract["feature_columns"]
            or supplied.get("sensor_groups") != contract["sensor_groups"]):
        raise ValueError("G3 group selected physiological_68_v1; use its 68-feature schema, not raw112.")
    frame, columns = load_feature_table(path, schema_path)
    if list(frame) != contract["output_column_order"]:
        raise ValueError("Input columns/order differ from the physiological68 contract.")
    return {"n_input_windows": len(frame), "n_features": len(columns),
            "available_subjects": sorted(set(frame.subject_id), key=lambda s: int(s[1:]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="Explicit input table; no default dataset choice")
    parser.add_argument("--schema", help="Optional schema, otherwise use the CSV's companion schema")
    parser.add_argument("--test-subject", required=True)
    parser.add_argument("--sensor-set", default="combined_physiological68", choices=["combined_physiological68"])
    parser.add_argument("--output", required=True, help="Dedicated quick output directory")
    args = parser.parse_args()
    config = read_config("configs/g3_nhu_defaults.yaml")
    config.update(features=args.features, output_dir=args.output, sensor_set=args.sensor_set)
    if args.schema:
        config["schema"] = args.schema
    if not resolve_path(args.features).is_file():
        parser.error("Feature table does not exist; obtain the agreed group package first.")
    try:
        input_info = check_input(args.features, args.schema)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    out = resolve_path(args.output)
    if out.exists() and any(out.iterdir()):
        parser.error("Use a new quick output directory to keep this check separate.")
    result = run_experiment(config, outer_subjects=[args.test_subject], allow_partial=True)
    # Preserve the core runner's partial status and add an explicit user-facing scope.
    details = {}
    for task in config["tasks"]:
        for name in config["models"]:
            folder = out / "folds" / task / name / args.test_subject
            metrics = read_json(folder / "metrics.json")
            detail = read_json(folder / "details.json")
            details[f"{task}/{name}"] = {"seconds_fit_and_predict": metrics["seconds"],
                "warnings": detail["warnings"], "train_subjects": detail["outer_train_subjects"],
                "test_subjects": detail["outer_test_subjects"]}
    write_json(out / "quick_check.json", {"scope": "quick_one_outer_subject_not_official_G3",
        "run_status": result["status"], "input_feature_family_confirmed_by_group": True,
        "feature_version": config["feature_version"], **input_info,
        "requested_test_subject": args.test_subject, "models": details})
    print("Quick check only: inspect fold warnings/time; this is not the official G3 score table.")


if __name__ == "__main__":
    main()
