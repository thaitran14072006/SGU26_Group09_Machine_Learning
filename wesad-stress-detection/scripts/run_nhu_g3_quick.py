"""Nhu G3 preparation: one outer subject only, never a final 15-subject result."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wesad.evaluation import run_experiment
from wesad.utils import read_config, read_json, resolve_path, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="Explicit input table; no default dataset choice")
    parser.add_argument("--schema", help="Optional schema, otherwise use the CSV's companion schema")
    parser.add_argument("--test-subject", required=True)
    parser.add_argument("--sensor-set", required=True, help="Actual feature definition, e.g. combined_raw112")
    parser.add_argument("--output", required=True, help="Dedicated quick output directory")
    args = parser.parse_args()
    config = read_config("configs/g3_nhu_defaults.yaml")
    config.update(features=args.features, output_dir=args.output, sensor_set=args.sensor_set)
    if args.schema:
        config["schema"] = args.schema
    if not resolve_path(args.features).is_file():
        parser.error("Feature table does not exist; obtain the agreed group package first.")
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
        "run_status": result["status"], "input_selection_confirmed_by_group": False,
        "requested_test_subject": args.test_subject, "models": details})
    print("Quick check only: inspect fold warnings/time; this is not the official G3 score table.")


if __name__ == "__main__":
    main()
