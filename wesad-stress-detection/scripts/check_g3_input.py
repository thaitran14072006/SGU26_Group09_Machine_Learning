"""Kiểm tra môi trường và bảng sinh lý68 đủ 15 người trước khi chạy G3."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wesad.data import SUBJECTS, load_feature_table
from wesad.utils import file_hash, read_config, resolve_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_g3_knn_svm_68.yaml")
    args = parser.parse_args()
    if sys.version_info[:3] != (3, 12, 14):
        raise SystemExit(f"Cần Python 3.12.14; đang dùng {sys.version.split()[0]} tại {sys.executable}")
    config = read_config(args.config)
    schema_path = resolve_path(config["schema"])
    schema = json.loads(schema_path.read_text(encoding="utf-8-sig"))
    if schema.get("feature_version") != "physiological_68_v1":
        raise ValueError("Phải dùng schema physiological_68_v1 sinh cùng bảng CSV.")
    if len(schema.get("feature_columns", [])) != 68:
        raise ValueError("Schema phải chứa đúng 68 đặc trưng.")
    path = resolve_path(config["features"])
    frame, columns = load_feature_table(path, schema_path, SUBJECTS)
    if not ((frame.end_s - frame.start_s - 60).abs() < 1e-8).all():
        raise ValueError("Có cửa sổ không dài 60 giây.")
    if not ((frame.start_s / 5 - (frame.start_s / 5).round()).abs() < 1e-8).all():
        raise ValueError("Có điểm bắt đầu nằm ngoài lưới 5 giây.")
    if "label_fraction" not in frame or not frame.label_fraction.between(.90-1e-12, 1).all():
        raise ValueError("Thiếu hoặc sai tỷ lệ nhãn đa số >=90%.")
    print("PASS: Python 3.12.14; physiological_68_v1; đủ 15 người; không trùng khóa/Inf.")
    print(frame.groupby("subject_id").size().to_string())
    print(f"Windows={len(frame)}, features={len(columns)}, NaN={int(frame[columns].isna().sum().sum())}")
    print(f"CSV SHA256={file_hash(path)}")
    print("Kiểm tra này không thay nghiệm thu nguồn, bộ lọc hoặc chất lượng sinh lý của G2.")


if __name__ == "__main__":
    main()
