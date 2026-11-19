# WESAD Stress Detection

Dự án nhận diện stress từ đặc trưng cảm biến đeo. Công việc tổ chức theo **EXP-001–EXP-020**; xem [kế hoạch chung](docs/EXPERIMENT_PLAN_20.md).

## Bắt đầu

1. Dùng Python theo `.python-version`, tạo `.venv`, cài `requirements.lock.txt` và package editable.
2. Lấy dữ liệu theo [hướng dẫn dữ liệu](docs/DU_LIEU_VA_TRUY_CAP.md). Raw/feature CSV không nằm trong Git.
3. Dùng bảng physiological68 theo `configs/defaults.yaml`, kiểm tra checksum và schema trước khi train.
4. Mở notebook của EXP cần làm. Các notebook Dummy/LR mặc định `RUN_TRAIN=False` để đọc kết quả đã lưu; chỉ bật khi thực sự chạy lại.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

## Lưu file ở đâu?

| Thư mục | Vai trò |
|---|---|
| `src/wesad/`, `scripts/`, `configs/` | Mã dùng chung, lệnh chạy và cấu hình |
| `notebooks/experiments/` | Notebook làm việc |
| `experiments/<nhóm>/EXP-xxx/` | Hồ sơ kết quả đã chốt |
| `notebooks/04_evaluation/` | So sánh nhiều thí nghiệm |
| `outputs/` | Kết quả máy, checkpoint; không commit |
| `experiments/supplementary/` | Nghiên cứu bổ sung ngoài danh mục 20 hướng |

Mỗi kết quả giữ một hồ sơ: `experiment.ipynb`, `config.yaml`, `results.csv`, `figures/` và bằng chứng nguồn gốc. `config.json` nguyên bản là manifest lịch sử; `config.yaml` là cấu hình theo mã chung cho lần chạy mới. Không sửa manifest để giả như đã chạy lại.

## Các CSV khác nhau thế nào?

| File | Một dòng là gì? |
|---|---|
| `results.csv` | Một người test × task × model; bốn hồ sơ Dummy/LR đã chốt đều có 30 dòng |
| `predictions.csv` | Một cửa sổ test với nhãn thật và nhãn dự đoán |
| `comparison_by_subject.csv` | So sánh cùng người/task với đối chứng |
| `train_test_metrics.csv` | Điểm train/test theo EXP, task và outer fold |
| `train_test_summary.csv` | Mean/SD theo người cho một EXP và task |

Không cộng bản kết quả của nhiều run như các thí nghiệm độc lập. Dùng **một nhật ký trung tâm**: [experiments_logs.xlsx](experiments/experiments_logs.xlsx); không tạo Excel theo từng EXP.

## Hồ sơ Dummy/LR

| EXP | Notebook làm việc | Hồ sơ chốt |
|---|---|---|
| EXP-001 — Dummy | [Notebook](notebooks/experiments/EXP-001_working.ipynb) | [Run 14/11](experiments/01_baselines/EXP-001_Dummy/runs/20261114/experiment.ipynb) |
| EXP-002 — LR baseline | [Notebook](notebooks/experiments/EXP-002_working.ipynb) | [Kết quả](experiments/01_baselines/EXP-002_LR/experiment.ipynb) |
| EXP-011 — LR balanced | [Notebook](notebooks/experiments/EXP-011_working.ipynb) | [Kết quả](experiments/03_balancing/EXP-011_LR/experiment.ipynb) |
| EXP-016 — LR feature selection | [Notebook](notebooks/experiments/EXP-016_working.ipynb) | [Kết quả](experiments/04_additional/EXP-016_LR_Features/experiment.ipynb) |

Hồ sơ EXP-001 gốc của thành viên khác giữ nguyên; run 14/11 là nguồn riêng, không thay thế kết quả của người khác. [So sánh train–test](notebooks/04_evaluation/EXP_LR_train_test_comparison.ipynb). Trạng thái các EXP khác xem hồ sơ tương ứng, không suy ra hoàn tất từ việc có thư mục.

## Đánh giá

Binary và multiclass tách riêng; Macro-F1 và Balanced Accuracy lấy mean/SD theo subject. Dùng LOSO; chọn tham số trong inner GroupKFold của outer train. Imputation/scaling/selection/resampling chỉ fit trên train. Điểm train không phải bằng chứng tổng quát hóa.

Các bước tiếp theo: hoàn thiện phân tích lỗi, tổng hợp 20 hướng, chọn cấu hình; sau đó final fit/export và demo CSV. Không tự mở thêm chuỗi EXP ngoài kế hoạch.

## Tài liệu

- [Danh mục 20 hướng](docs/EXPERIMENT_PLAN_20.md), [registry](experiments/experiment_registry.json).
- [Lịch sử kỹ thuật G1–G4](docs/LEGACY_WORKFLOW.md): chỉ để truy nguồn, không phải quy trình hiện hành.
