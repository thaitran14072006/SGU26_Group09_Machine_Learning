"""Compare and combine the four submitted Challenge1 models."""

import hashlib
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
MEMBERS = ROOT / "members"
SUBMISSIONS = ROOT / "results/submissions"
NAMES = ["Mai", "Nam", "Nhu", "Thai"]


def align(path, ids):
    frame = pd.read_csv(path).set_index("Id", verify_integrity=True)
    if set(frame.index) != set(ids):
        raise ValueError(f"Id không khớp: {path}")
    return frame.loc[ids]


def checked_oof(path, development, actual_column, prediction_column, log_scale):
    rows = align(path, development.index)
    actual = rows[actual_column] if log_scale else np.log(rows[actual_column])
    np.testing.assert_allclose(actual, np.log(development["SalePrice"]))
    if "cv_fold" in rows:
        np.testing.assert_array_equal(rows["cv_fold"], development["cv_fold"])
    predicted = rows[prediction_column] if log_scale else np.log(rows[prediction_column])
    if not np.isfinite(predicted).all():
        raise ValueError(f"Dự đoán OOF không hợp lệ: {path}")
    return predicted


def main():
    sys.path.insert(0, str(MEMBERS / "Nam/src"))
    import workflow as svr

    train_path = ROOT / "data/raw/train.csv"
    split_path = ROOT / "data/splits/split_seed7.csv"
    train = pd.read_csv(train_path).set_index("Id", verify_integrity=True)
    split = align(split_path, train.index)
    data = train.join(split)
    dev = data.loc[data["split"].eq("development")]
    hold = data.loc[data["split"].eq("holdout")]
    mai = MEMBERS / "Mai/outputs/predictions"
    nam = MEMBERS / "Nam/outputs"
    nhu = MEMBERS / "Nhu/outputs/experiments/20261115_171330_237145"
    thai = MEMBERS / "Thai/outputs"
    thai_run = thai / "experiments/Thai_EXP-010/20261115T090441429482"
    thai_result = json.loads((thai_run / "result.json").read_text(encoding="utf-8"))
    train_lf = train_path.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(train_lf).hexdigest() != thai_result["train_sha256"]:
        raise ValueError("Hash train của Thái không khớp sau khi chuẩn hóa xuống dòng.")

    nam_report = json.loads((nam / "v5_holdout_report.json").read_text(encoding="utf-8"))
    nam_dev = pd.read_csv(ROOT / "data/splits/development_ids.csv")["Id"]
    for fold, (_, validation) in enumerate(svr.CV.split(nam_dev)):
        np.testing.assert_array_equal(dev.loc[nam_dev.iloc[validation], "cv_fold"], fold)

    oof = np.column_stack([
        checked_oof(mai / "oof_Mai.csv", dev, "SalePrice", "PredictedSalePrice", False),
        checked_oof(nam / "predictions/v4_oof_error_analysis.csv",
                    dev, "actual_log", "oof_pred_log", True),
        checked_oof(nhu / "predictions/Nhu_EXP-003_oof.csv",
                    dev, "true_log", "predicted_log", True),
        checked_oof(thai_run / "oof_predictions.csv", dev, "actual_log", "predicted_log", True),
    ])
    actual_log = np.log(dev["SalePrice"].to_numpy())
    folds = dev["cv_fold"].to_numpy()

    def score(weights):
        squared = (actual_log - oof @ weights) ** 2
        return np.mean([np.sqrt(squared[folds == fold].mean()) for fold in range(5)])

    expected_cv = [0.12863228211634378, nam_report["development_cv_rmse"],
                   0.14517989419560304, thai_result["cv_rmse_log_mean"]]
    np.testing.assert_allclose([score(w) for w in np.eye(4)], expected_cv, atol=1e-8)
    # Include equal weights as a baseline. Select exclusively on development OOF.
    candidates = [np.full(4, .25)]
    candidates.extend(np.array(parts) / 10 for parts in product(range(11), repeat=4)
                      if sum(parts) == 10)
    scores = pd.DataFrame(candidates, columns=NAMES)
    scores["cv_rmse_log"] = [score(w) for w in candidates]
    best = scores.loc[scores["cv_rmse_log"].idxmin()]
    weights = best[NAMES].to_numpy()

    # SVR publishes a holdout summary. Reproduce its per-house predictions
    # with the published configuration fit on development only.
    columns = train.columns.drop("SalePrice")
    X_nam = dev.loc[nam_dev, columns]
    pipeline, _ = svr.build_pipeline(X_nam, nam_report["mode"], nam_report["params"])
    pipeline.fit(X_nam, np.log(dev.loc[nam_dev, "SalePrice"]))
    nam_hold_log = pipeline.predict(hold[columns])
    hold_log = np.log(hold["SalePrice"].to_numpy())
    np.testing.assert_allclose(np.sqrt(np.mean((hold_log - nam_hold_log) ** 2)),
                               nam_report["holdout_rmse_log"], atol=1e-8)

    hold_predictions = np.column_stack([
        np.log(align(mai / "holdout_errors.csv", hold.index)["PredictedSalePrice"]),
        nam_hold_log,
        np.log(align(nhu / "predictions/holdout_errors.csv", hold.index)["predicted_price"]),
        np.log(align(thai / "predictions/holdout_errors.csv", hold.index)["predicted"]),
    ])
    hold_blend = hold_predictions @ weights
    hold_rmse = np.sqrt(np.mean((hold_log - hold_blend) ** 2))
    hold_mae = np.mean(np.abs(hold["SalePrice"] - np.exp(hold_blend)))

    sample = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    submission_files = ["submission_Mai.csv", "submission_Nam_SVR.csv",
                        "submission_Nhu.csv", "submission_Thai.csv"]
    test_prices = np.column_stack([
        align(SUBMISSIONS / filename, sample["Id"])["SalePrice"] for filename in submission_files
    ])
    if not np.isfinite(test_prices).all() or not (test_prices > 0).all():
        raise ValueError("Giá trong submission phải hữu hạn và dương.")
    sample["SalePrice"] = np.exp(np.log(test_prices) @ weights)
    sample.to_csv(SUBMISSIONS / "submission_ensemble.csv", index=False)
    scores.sort_values("cv_rmse_log", kind="stable").to_csv(
        ROOT / "ensemble/oof_comparison.csv", index=False,
    )
    print(f"Weights: {dict(zip(NAMES, weights.tolist()))}")
    print(f"Equal-weight CV: {scores.iloc[0].cv_rmse_log:.5f} | Selected CV: {best.cv_rmse_log:.5f}")
    print(f"Holdout RMSE log: {hold_rmse:.5f} | Holdout MAE: {hold_mae:,.0f}")


if __name__ == "__main__":
    main()
