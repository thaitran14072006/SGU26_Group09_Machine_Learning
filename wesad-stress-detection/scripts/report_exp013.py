"""Export EXP-013 summary, evidence and PNG from the project's existing runner."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/03_balancing/EXP-013_SVM"

def export(source=None):
    output = Path(source) if source else EXPERIMENT / "run_outputs/main"
    if not output.is_absolute(): output = ROOT / output
    summary_path = output / "summary.json"
    manifest_path = output / "config.json"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Chưa chạy chính thức: thiếu summary.json hoặc config.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = json.loads(manifest_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or len(summary.get("results", [])) != 2:
        raise ValueError("Chỉ được xuất kết quả khi cả 15 người và 2 bài toán đã hoàn tất.")
    settings = config.get("settings", {})
    if settings.get("models") != ["svm"] or settings.get("balance") is not True or settings.get("nested") is not False:
        raise ValueError("Manifest không phải cấu hình EXP-013 SVM balanced.")
    if set(settings.get("tasks", [])) != {"binary", "multiclass"}:
        raise ValueError("Thiếu bài toán binary/multiclass.")
    metrics = pd.read_csv(output / "metrics_by_subject.csv", dtype={"subject_id":str})
    predictions = pd.read_csv(output / "predictions.csv", dtype={"subject_id":str,"window_id":str})
    if len(metrics) != 30 or set(metrics.task) != {"binary", "multiclass"} or any(metrics.groupby("task").subject_id.nunique() != 15):
        raise ValueError("Không đủ 30 fold / 15 người mỗi bài toán.")
    if predictions.duplicated(["subject_id","window_id","task","model"]).any():
        raise ValueError("Dự đoán bị trùng khóa.")
    if not set(predictions.model)=={"svm"}:
        raise ValueError("Dự đoán chứa mô hình khác.")
    for _,row in metrics.iterrows():
        part=predictions[(predictions.task==row.task)&(predictions.subject_id==row.subject_id)]
        if len(part)!=int(row.n_test): raise ValueError("Số dự đoán không khớp fold")
    # preserve runner's exact metrics, no invented measurements
    metrics.to_csv(EXPERIMENT / "results.csv", index=False)
    predictions.to_csv(EXPERIMENT / "predictions.csv", index=False)
    summaries = []
    for item in summary["results"]:
        if item.get("n_subjects")!=15 or not item.get("complete_15_subjects"):
            raise ValueError("Kết quả không đủ 15 người")
        summaries.append({"task": item["task"], "model":"svm", "n_subjects":item["n_subjects"],
                          "macro_f1_mean":item["macro_f1_mean"],"macro_f1_std":item["macro_f1_std"],
                          "balanced_accuracy_mean":item["balanced_accuracy_mean"],
                          "balanced_accuracy_std":item["balanced_accuracy_std"]})
    df_summary=pd.DataFrame(summaries)
    df_summary.to_csv(EXPERIMENT / "summary.csv", index=False)
    images=EXPERIMENT / "figures"
    images.mkdir(exist_ok=True)
    for task, labels, names in [("binary",[0,1],["Non-stress","Stress"]),
                               ("multiclass",[1,2,3],["Baseline","Stress","Amusement"])]:
        subset=predictions[predictions.task==task]
        matrix=confusion_matrix(subset.y_true,subset.y_pred,labels=labels)
        fig,ax=plt.subplots(figsize=(6,5))
        ax.imshow(matrix,cmap="Blues")
        ax.set_xticks(range(len(labels)),names)
        ax.set_yticks(range(len(labels)),names)
        ax.set(xlabel="Dự đoán",ylabel="Nhãn thật",title=f"EXP-013 SVM balanced | {task}")
        for (i,j),v in np.ndenumerate(matrix): ax.text(j,i,str(v),ha="center",va="center")
        fig.tight_layout(); fig.savefig(images/f"confusion_matrix_{task}.png",dpi=160);plt.close(fig)
        values=metrics[metrics.task==task].sort_values("subject_id",key=lambda s:s.str[1:].astype(int))
        fig,ax=plt.subplots(figsize=(11,4.5))
        ax.bar(values.subject_id,values.macro_f1)
        ax.set(ylim=(0,1),xlabel="Người tham gia",ylabel="Macro-F1",title=f"EXP-013 LOSO | {task}")
        fig.tight_layout();fig.savefig(images/f"macro_f1_by_subject_{task}.png",dpi=160);plt.close(fig)
    provenance={"experiment":"EXP-013", "scope":"main", "summary_status":summary["status"],
                "source_output":str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output),
                "signature":config.get("signature"),"data_sha256":config.get("data_sha256"),
                "reference":"SVM baseline EXP-004; only compare if datasets/protocols match"}
    (EXPERIMENT/"result_provenance.json").write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding="utf-8")
    # Compare only when EXP-004 provides verified matching 68-feature LOSO provenance.
    baseline_dir = ROOT / "experiments/01_baselines/EXP-004_SVM"
    baseline_summary = baseline_dir / "summary.csv"
    baseline_provenance = baseline_dir / "result_provenance.json"
    baseline_results = baseline_dir / "results.csv"
    if all(x.is_file() for x in (baseline_summary,baseline_provenance,baseline_results)):
        try:
            reference=json.loads(baseline_provenance.read_text(encoding="utf-8"))
            ref_sha = reference.get("data_sha256") or reference.get("settings",{}).get("feature_sha256")
            old=pd.read_csv(baseline_summary)
            old_metrics=pd.read_csv(baseline_results,dtype={"subject_id":str})
            if (ref_sha != config.get("data_sha256") or len(old_metrics)!=30 or
                set(old_metrics.task)!={"binary","multiclass"} or
                any(old_metrics.groupby("task").subject_id.nunique()!=15) or
                not {"task","macro_f1_mean","balanced_accuracy_mean"} <= set(old.columns)):
                raise ValueError("EXP-004 baseline không đủ bằng chứng khớp dữ liệu/LOSO")
            baseline_cols=old_metrics[["task","subject_id","macro_f1","balanced_accuracy","n_test"]].rename(columns={
                "macro_f1":"macro_f1_exp004", "balanced_accuracy":"balanced_accuracy_exp004",
                "n_test":"n_test_exp004"})
            balanced_cols=metrics[["task","subject_id","macro_f1","balanced_accuracy","n_test"]].rename(columns={
                "macro_f1":"macro_f1_exp013", "balanced_accuracy":"balanced_accuracy_exp013",
                "n_test":"n_test_exp013"})
            comp=balanced_cols.merge(baseline_cols,on=["task","subject_id"],validate="one_to_one")
            if len(comp)!=30 or not (comp.n_test_exp013==comp.n_test_exp004).all():
                raise ValueError("Số lượng cửa sổ trên fold không tương thích")
            comp["delta_macro_f1"]=comp.macro_f1_exp013-comp.macro_f1_exp004
            comp["delta_balanced_accuracy"]=comp.balanced_accuracy_exp013-comp.balanced_accuracy_exp004
            comp.to_csv(EXPERIMENT / "comparison_exp004.csv",index=False)
            fig,ax=plt.subplots(figsize=(7,4.5))
            means=comp.groupby("task")[["delta_macro_f1","delta_balanced_accuracy"]].mean()
            means.plot(kind="bar",ax=ax,rot=0)
            ax.set(ylabel="Chênh lệch điểm",title="EXP-013 trừ EXP-004 (cùng LOSO)")
            fig.tight_layout();fig.savefig(images / "comparison_exp004.png",dpi=160);plt.close(fig)
            provenance["comparison_exp004"]="paired 30 folds; same data SHA and test counts"
            (EXPERIMENT/"result_provenance.json").write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding="utf-8")
        except (ValueError,KeyError,TypeError) as err:
            print("Bỏ qua so sánh EXP-004 do chưa thể xác minh:",err)
    else:
        print("EXP-004 chưa có đủ CSV và provenance để so sánh; không tạo số liệu giả.")
    print(df_summary.to_string(index=False))
    print("Đã xuất:",EXPERIMENT)

if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--source"); args=p.parse_args(); export(args.source)
