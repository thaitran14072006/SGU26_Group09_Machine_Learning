"""Basic RF/MLP figures from a verified complete G3 run (no model fitting).

May run in a separate plotting environment with matplotlib; training results
are read from the saved JSON/CSV, never recomputed using that environment.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads((args.run / "verification.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete" or verification["scope"] != "main" or verification["status"] != "passed":
        raise ValueError("Plot only a verified full main run, never quick results.")
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [("multiclass", ["Baseline", "Stress", "Amusement"]),
             ("binary", ["Non-stress", "Stress"])]
    colors = {"rf": "#2378a8", "mlp": "#bd5b22"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
    for row, (task, labels) in enumerate(tasks):
        for col, model in enumerate(["rf", "mlp"]):
            entry = next(r for r in summary["results"] if r["task"] == task and r["model"] == model)
            if entry["n_subjects"] != 15 or not entry["complete_15_subjects"]:
                raise ValueError("Each plotted model/task must have all 15 held-out subjects.")
            matrix = entry["confusion_matrix"]
            ax = axes[row, col]
            maximum = max(max(values) for values in matrix)
            plotted = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
            for i, values in enumerate(matrix):
                for j, value in enumerate(values):
                    ax.text(j, i, str(value), ha="center", va="center",
                            color="white" if value > maximum * .55 else "#182c3d")
            ax.set(xticks=range(len(labels)), yticks=range(len(labels)),
                   xticklabels=labels, yticklabels=labels, xlabel="Predicted", ylabel="True",
                   title=f"{model.upper()} | {task}")
            fig.colorbar(plotted, ax=ax, shrink=.75, label="Windows")
    fig.suptitle("RF/MLP defaults | physiological68 | pooled out-of-fold predictions\n15-subject LOSO; windows overlap", fontsize=12)
    fig.savefig(args.output / "confusion_matrices.png", dpi=150)
    plt.close(fig)

    with (args.run / "metrics_by_subject.csv").open(encoding="utf-8", newline="") as stream:
        metrics = list(csv.DictReader(stream))
    subjects = sorted({r["subject_id"] for r in metrics}, key=lambda s: int(s[1:]))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
    for ax, (task, _) in zip(axes, tasks):
        for model in ["rf", "mlp"]:
            selected = {r["subject_id"]: float(r["macro_f1"]) for r in metrics if r["task"] == task and r["model"] == model}
            if set(selected) != set(subjects) or len(selected) != 15:
                raise ValueError("Missing subject scores.")
            ax.plot(subjects, [selected[s] for s in subjects], marker="o", markersize=4,
                    linewidth=1.4, color=colors[model], label=model.upper())
        ax.set(title=task, ylabel="Macro-F1", ylim=(0, 1.05))
        ax.grid(axis="y", alpha=.25)
        ax.legend(loc="lower right")
    axes[-1].set_xlabel("Held-out subject")
    fig.suptitle("RF/MLP defaults | physiological68 | one score per held-out subject", fontsize=12)
    fig.savefig(args.output / "macro_f1_by_subject.png", dpi=150)
    plt.close(fig)
    (args.output / "plot_environment.json").write_text(json.dumps({
        "matplotlib": matplotlib.__version__, "scope": "render_saved_metrics_only_no_training",
        "run": args.run.name, "figures": ["confusion_matrices.png", "macro_f1_by_subject.png"]}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
