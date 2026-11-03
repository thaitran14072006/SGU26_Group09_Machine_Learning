# Kết quả kiểm tra S2 ngày 03 tháng 11 năm 2026

Đã chạy dữ liệu WESAD S2 thật trên máy Như bằng source commit `2a6d97e973174ade15b0b547146cd072c9fcd2d9`. Lượt thống kê raw và lượt chẩn đoán tiền xử lý đều hoàn tất. Kết quả xác nhận việc đọc PKL, chia cửa sổ và căn chỉnh theo thời gian trên máy này; chưa thay nghiệm thu bốn máy, chất lượng sinh lý hoặc bảng 68 feature của Mai.

## Cấu hình và bằng chứng

- Raw SHA256: `36ef5e8afc0f91998eefba7c12fc9fa97b7b07198cbec0126917d7abb436ca23`.
- Code SHA256: `b541a76d4898d72fbd6bc27e06051978449a369de013e58068a4a5ef1cc5e85d`.
- Python 3.13.9; thư viện theo `requirements.lock.txt`.
- Cửa sổ 60 giây, bước 5 giây; nhãn đa số >=90%, nhãn đích 1/2/3.
- Config raw: `configs/features_nhu_smoke.yaml`. Config lọc: `configs/preprocessing_nhu_draft.yaml`.
- Đường dẫn raw riêng nằm trong `configs/local.yaml` được Git bỏ qua. Không sao chép dữ liệu vào repository.

## Kết quả thực tế

| Chỉ tiêu | Kết quả |
| --- | ---: |
| Thời lượng nhãn và từng kênh được cấu hình | 6.079 giây |
| Cửa sổ đầy đủ trước kiểm tra nhãn | 1.204 |
| Loại do quy tắc nhãn | 809 |
| Loại thêm do thiếu mẫu, NaN/Inf hoặc vùng biên lọc, sau khi nhãn đạt | 0 |
| Cửa sổ giữ lại | 395 |
| Baseline (1) | 219 |
| Stress (2) | 113 |
| Amusement (3) | 63 |
| Thời gian extractor raw ghi nhận | 3,870 giây |
| Thời gian công cụ chẩn đoán ghi nhận | 3,783 giây |

Thời gian trên là thời gian xử lý từng subject do script ghi nhận, không gồm toàn bộ khởi động Python; chưa đo RAM đỉnh. Số liệu này không đại diện thời gian extractor sinh lý 68 feature.

Các kiểm tra bổ sung đã đạt:

- Hai bảng raw và sau lọc khớp toàn bộ 395 khóa, nhãn gốc/nhị phân, tỷ lệ nhãn, mốc bắt đầu/kết thúc.
- Mỗi cửa sổ dài đúng 60 giây, không trùng khóa.
- 21 cột thống kê raw đều hữu hạn.
- Cả 10 kênh cấu hình có cùng thời lượng với nhãn, chênh lệch 0 giây; shape phù hợp cấu hình. Không cần tự thêm offset hoặc nâng tần số.
- Raw không có NaN/Inf ở các kênh được kiểm tra. Sau lọc có NaN chủ động tại vùng biên: EDA ngực 14.000 mẫu, Resp 42.000 mẫu, EDA cổ tay 80 mẫu. Các vùng này không làm mất thêm cửa sổ đã đạt nhãn trên S2; không diễn giải rằng chúng không tồn tại.
- Các file PKL của đủ 15 subject dự kiến hiện có trên máy; lượt này chỉ xử lý S2.

## Kiểm tra hình trước và sau lọc

Đã tạo và xem hình đoạn Baseline 305–365 giây (cửa sổ hợp lệ đầu tiên), cho EDA ngực, EDA cổ tay và Resp. Hình so sánh đường raw và sau lọc tại tần số gốc. Trục tung vẫn ghi đơn vị đầu vào, không tự khẳng định đã xác minh scale.

Các file chỉ lưu trong output được Git bỏ qua:

```text
outputs/preprocessing/nhu_S2_draft_01/
  config.json
  run_status.json
  S2_quality.json
  S2_windows.csv
  S2_before_after_filter.png
  S2_before_after_filter.svg
  chest_eda_305_365.csv
  wrist_eda_305_365.csv
  chest_resp_305_365.csv
```

Bảng thống kê raw nằm trong `data/features/nhu_s2_chest_raw_stats_majority_v1/`, cũng được Git bỏ qua. Gói feature/chẩn đoán cần chia sẻ riêng khi đối chiếu nhóm.

## Vấn đề chất lượng và phần chưa nghiệm thu

File gốc `S2/S2_readme.txt` ghi cảm biến nhiệt độ ngực không được gắn sát hoàn toàn trong suốt quá trình. Việc Temp có đủ mẫu và giá trị hữu hạn chưa chứng minh chất lượng tốt. Cần Mai/Như thống nhất cách gắn cờ và xử lý ảnh hưởng khi so sánh cảm biến; chưa tự loại toàn bộ S2 hoặc đổi giá trị Temp.

Đã đọc `wesad_readme.pdf` đi kèm dữ liệu: mục II.3 xác nhận PKL chứa dữ liệu đã đồng bộ; mục II.2 ghi ACC E4 dạng 1/64 g, EDA µS và TEMP °C. Tuy nhiên, các công thức RespiBAN trong mục II.1 mô tả file TXT; cần đối chiếu dữ liệu PKL thực tế trước khi áp dụng lại chuyển đổi. Chưa sửa scale của PKL hoặc thay trạng thái đơn vị trong schema.

Vẫn giữ `physiological_qa_validated: false`, `ready_for_training: false` ở gói chẩn đoán. Cutoff EDA cổ tay, vùng biên, scale/đơn vị, detector ECG/BVP/SCR, dải PSD và QA sinh lý cần chốt với Mai. Schema Mai vẫn là bản nháp; chưa sinh bảng 68 feature, chưa nghiệm thu S2 bốn máy và chưa chạy lô S4/S9/S11/S17.

Bước tiếp theo: gửi checksum, cấu hình và kết quả S2 cho Mai đối chiếu; hoàn thiện các quyết định còn mở trong [bản bàn giao tiền xử lý](G2_NHU_BAN_GIAO_TIEN_XU_LY.md), sau đó tích hợp extractor G2 rồi mới chạy lô của Như.
