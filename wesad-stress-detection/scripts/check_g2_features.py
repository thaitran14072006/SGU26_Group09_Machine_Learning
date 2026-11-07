"""Validate a physiological G2 package or strictly merge members' packages."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from wesad.feature_packages import merge_packages, package_report
from wesad.utils import read_json, resolve_path, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", nargs="+", required=True)
    parser.add_argument("--contract", default="configs/feature_schema.json")
    parser.add_argument("--report", required=True)
    parser.add_argument("--merge-output")
    parser.add_argument("--require-all-subjects", action="store_true")
    parser.add_argument("--allow-partial-merge", action="store_true", help="Explicitly label an incomplete merge")
    args = parser.parse_args()
    contract = read_json(resolve_path(args.contract))
    directories = [resolve_path(d) for d in args.features_dir]
    try:
        if args.merge_output:
            result = merge_packages(directories, resolve_path(args.merge_output), contract,
                                    require_complete=not args.allow_partial_merge)
        elif len(directories) == 1:
            result = package_report(directories[0], contract, complete=args.require_all_subjects)
        else:
            raise ValueError("Multiple packages require --merge-output")
    except (ValueError, KeyError, OSError, AssertionError) as exc:
        write_json(resolve_path(args.report), {"status": "failed", "error": str(exc)})
        print(str(exc), file=sys.stderr)
        return 1
    write_json(resolve_path(args.report), result)
    print(f"G2 package check passed: {result['n_windows']} windows, {result['scope']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
