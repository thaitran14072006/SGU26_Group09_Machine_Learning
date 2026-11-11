"""Run and compare balanced variants using the parameters selected in G4."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from wesad.evaluation import run_experiment
from wesad.g4 import KEY, inputs
from wesad.utils import file_hash, read_config, read_json, write_json

reference = Path("outputs/g4_group_tuning_complete_20261110")
output = Path("outputs/g4_balanced_tuned_20261111")
config = read_config("configs/g4_rf_mlp_additional.yaml")
inputs(config)
summary = read_json(reference / "summary.json")
assert summary["status"] == "complete"
assert summary["data_sha256"] == config["feature_sha256"]
params = read_json(reference / "best_params.json")
assert len(params) == 120
for key, digest in read_json(reference / "handoff_checksums.json").items():
    if key in {"best_params.json", "predictions.csv", "metrics_by_subject.csv", "summary.json"}:
        assert file_hash(reference / key) == digest
config.pop("outer_subjects")
config.pop("search_spaces")
config.update(protocol="g4_balanced_fixed_tuned_v1", run_kind="balanced_comparison",
              output_dir=str(output), models=["knn", "svm", "rf", "mlp"],
              balance=True, nested=False, fixed_fold_params=params,
              reference_params_sha256=file_hash(reference / "best_params.json"))
config["source_paths"][-1] = "scripts/run_g4_balanced_tuned.py"
result = run_experiment(config, resume=True)
assert result["status"] == "complete"
assert read_json(output / "best_params.json") == params

plain = pd.read_csv(reference / "predictions.csv")
balanced = pd.read_csv(output / "predictions.csv")
paired = plain.merge(balanced, on=KEY, how="outer", validate="one_to_one",
                     suffixes=("_plain", "_balanced"), indicator=True)
assert len(paired) == 49512 and paired["_merge"].eq("both").all()
assert paired.y_true_plain.eq(paired.y_true_balanced).all()

keys = ["task", "model", "subject_id"]
comparison = pd.read_csv(reference / "metrics_by_subject.csv").merge(
    pd.read_csv(output / "metrics_by_subject.csv"), on=keys, validate="one_to_one",
    suffixes=("_plain", "_balanced"))
assert len(comparison) == 120 and comparison.n_test_plain.eq(comparison.n_test_balanced).all()
scores = ["macro_f1", "balanced_accuracy"]
for score in scores:
    comparison[f"{score}_delta"] = comparison[f"{score}_balanced"] - comparison[f"{score}_plain"]
reports = Path("reports/g4_balanced_tuned")
reports.mkdir(parents=True, exist_ok=True)
comparison.to_csv(reports / "comparison_by_subject.csv", index=False)
columns = [f"{score}_{suffix}" for score in scores for suffix in ("plain", "balanced", "delta")]
means = comparison.groupby(["task", "model"])[columns].mean().reset_index()
write_json(reports / "summary.json", dict(
    status="complete", folds=len(comparison), predictions=len(paired),
    data_sha256=config["feature_sha256"], reference_params_sha256=config["reference_params_sha256"],
    methods=dict(knn="RandomOverSampler", mlp="RandomOverSampler",
                 svm="class_weight=balanced", rf="class_weight=balanced"),
    results=means.to_dict("records")))
print(means.to_string(index=False))
