# Kế hoạch thực nghiệm hiện hành — 20 hướng

Đơn vị công việc và tổng hợp là EXP-001–EXP-020. G1/G2/G3/G4 chỉ là tên nguồn dữ liệu, API hoặc báo cáo lịch sử; không dùng làm các mốc công việc hiện tại. Không sửa tên module, hash hoặc manifest cũ để đổi hình thức.

| ID | Mô hình | Hướng | Hồ sơ đã đồng bộ trong lượt này |
|---|---|---|---|
| EXP-001 | Dummy | Baseline | [EXP-001](../experiments/01_baselines/EXP-001_Dummy/runs/20261114/experiment.ipynb) |
| EXP-002 | LR | Baseline | [EXP-002](../experiments/01_baselines/EXP-002_LR/experiment.ipynb) |
| EXP-003 | KNN | Baseline | Giữ nguyên hồ sơ hiện có; chưa kiểm tra lại |
| EXP-004 | SVM | Baseline | Giữ nguyên hồ sơ hiện có; chưa kiểm tra lại |
| EXP-005 | RF | Baseline | Giữ nguyên hồ sơ hiện có; chưa kiểm tra lại |
| EXP-006 | MLP | Baseline | Giữ nguyên hồ sơ hiện có; chưa kiểm tra lại |
| EXP-007 | KNN | Tuning | Giữ nguyên hồ sơ hiện có; chưa kiểm tra lại |
| EXP-008 | SVM | Tuning | Giữ nguyên hồ sơ hiện có; chưa kiểm tra lại |
| EXP-009 | RF | Tuning | Giữ nguyên hồ sơ hiện có; chưa kiểm tra lại |
| EXP-010 | MLP | Tuning | Giữ nguyên hồ sơ hiện có; chưa kiểm tra lại |
| EXP-011 | LR | Class balancing | [EXP-011](../experiments/03_balancing/EXP-011_LR/experiment.ipynb) |
| EXP-012 | KNN | Class balancing | Giữ nguyên hồ sơ hiện có; chưa kiểm tra lại |
| EXP-013 | SVM | Class balancing | Giữ nguyên hồ sơ hiện có; chưa kiểm tra lại |
| EXP-014 | RF | Class balancing | Giữ nguyên hồ sơ hiện có; chưa kiểm tra lại |
| EXP-015 | MLP | Class balancing | Giữ nguyên hồ sơ hiện có; chưa kiểm tra lại |
| EXP-016 | LR | Feature selection | [EXP-016](../experiments/04_additional/EXP-016_LR_Features/experiment.ipynb) |
| EXP-017 | KNN | Feature selection | Giữ nguyên hồ sơ hiện có; chưa kiểm tra lại |
| EXP-018 | SVM | Feature selection | Giữ nguyên hồ sơ hiện có; chưa kiểm tra lại |
| EXP-019 | RF | Feature importance/selection | Giữ nguyên hồ sơ hiện có; chưa kiểm tra lại |
| EXP-020 | MLP | Architecture/regularization | Giữ nguyên hồ sơ hiện có; chưa kiểm tra lại |

## Quy ước chung

- Một EXP có nhiều run hoặc người thực hiện; run khác không tạo thêm hướng. Giữ ngày thực tế, nguồn kết quả và commit.
- Dùng mã EXP chung trong thư mục/notebook. Mã riêng cũ chỉ còn trong manifest nguyên bản; không cần thư mục alias trùng lặp.
- physiological68, cùng checksum/schema/subject/nhãn; LOSO ngoài, inner GroupKFold khi chọn tham số. Mọi biến đổi fit trong train.
- Hồ sơ chốt có experiment.ipynb, config.yaml, results.csv, figures và provenance; notebook đọc report phải ghi rõ không phải training mới.
- EXP-001 đã có hồ sơ của thành viên khác: giữ nguyên và thêm run ngày 14/11 dưới runs/20261114; không thay thế kết quả của người khác.
- Các hướng khác ngoài 001/002/011/016 chưa được nghiệm thu lại trong lượt đồng bộ này; không suy ra trạng thái từ việc có thư mục.

## Công việc tiếp theo

1. Hoàn thiện phân tích lỗi và kết luận của EXP-001/002/011/016.
2. Đối chiếu 20 hồ sơ của nhóm theo cùng dữ liệu và protocol, cập nhật nhật ký trung tâm.
3. So sánh ứng viên, chốt cấu hình; sau đó final fit/export, metadata/schema và demo upload CSV.
4. Không tự chạy thêm chuỗi THAI-EXP-006–010. LR tuning C đã có được giữ như phụ lục thăm dò ngoài 20 hướng.


## Lưu trữ tối thiểu

Một hồ sơ chốt cho mỗi kết quả; notebook làm việc giữ mã train, notebook trong experiments là bản báo cáo. Chỉ dùng experiments/experiments_logs.xlsx; phần Dummy/LR giữ 8 dòng kết quả chính, bỏ các dòng thử S2. Checkpoint máy trong outputs không phải hồ sơ nhóm. Kết quả bổ sung duy nhất ở experiments/supplementary/LR_Regularization.
