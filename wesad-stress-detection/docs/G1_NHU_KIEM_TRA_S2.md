# Kiểm tra mẫu S2 G1

Cập nhật ngày 05/11/2026 theo README mới của Nam. Lượt này chỉ đọc PKL S2, chạy thống kê raw có sẵn và kiểm tra G1; chưa chạy bộ lọc sinh lý hoặc extractor 68 feature.

## Nguồn và môi trường

- Python 3.12.14 trong `.venv-g1`, thư viện theo `requirements.lock.txt`; `pip check` đạt.
- Raw SHA256: `36ef5e8afc0f91998eefba7c12fc9fa97b7b07198cbec0126917d7abb436ca23`.
- Code SHA256: `43192da69227611ac1877f1fb8fb92c6435daa69b7d8e324d34895d89ee64c9d`.
- Config `configs/features_smoke.yaml`: ngực ECG/EDA/Resp, 700 Hz, cửa sổ 60 giây, bước 5 giây, majority >=90%.
- Chọn S2 bằng `--subjects S2`, output riêng `data/features/g1_s2_raw_stats/` đúng ví dụ README.

## Kết quả trên máy Như

| Nội dung | Kết quả |
| --- | ---: |
| Cửa sổ giữ lại | 395 |
| Baseline | 219 |
| Stress | 113 |
| Amusement | 63 |
| Metadata | 8 |
| Thống kê raw | 21 |
| Thời gian extractor ghi nhận | 5,143 giây |
| Test đạt | 25/25 |

Thời gian không gồm toàn bộ khởi động Python; chưa đo RAM đỉnh. `run_status.json` complete chỉ áp dụng S2. Gói đạt kiểm tra metadata, khóa, lưới thời gian, tỷ lệ nhãn, schema, manifest/checksum và tính hữu hạn. Số 395 là quan sát, không được mã hóa thành điều kiện bắt buộc.

25 test gồm 14 hạ tầng gốc, 5 kiểm tra gói G1 và 6 kiểm tra cửa sổ/CLI. Test mới sử dụng cửa sổ 42.000 mẫu ở 700 Hz: 37.800 mẫu nhãn hợp lệ được nhận, 37.799 bị loại; nhãn 0/4/5/6/7 không thành lớp đích; cửa sổ mặc định dịch 3.500 mẫu; giữ trục thời gian gốc và bỏ phần cuối thiếu độ dài.

Kiểm tra cấu trúc raw: cả 10 kênh ngực/cổ tay có shape đúng cấu hình, không có NaN/Inf và cùng thời lượng 6.079 giây với nhãn. Xác nhận cấu trúc không chứng minh tín hiệu đạt chất lượng sinh lý. Dữ liệu đã đồng bộ trong PKL; chưa tự thêm offset hoặc đổi scale/đơn vị.

Ghi chú nguồn `S2/S2_readme.txt` cho biết cảm biến nhiệt độ ngực không luôn được gắn sát. Ghi nhận vấn đề tiếp xúc để thiết kế QA; chưa tự sửa giá trị hoặc loại S2.

## Đối chiếu phần Nam vừa push

Đọc `notebooks/reports/tables/eda_step5/eda_summary.json` và `windows_by_subject_and_label.csv` từ main tại `21a9ef0`:

- Cấu hình xử lý trong báo cáo Nam trùng cấu hình lượt S2 này.
- Hàng S2 của Nam cũng có 219 Baseline, 113 Stress, 63 Amusement, tổng 395.
- Báo cáo Nam có 6.196 cửa sổ từ 15 người; đây là kết quả EDA của Nam, không phải lượt chạy 15 người trên máy Như.

So sánh trên chỉ ở mức cấu hình và số lượng. Bảng tóm tắt không chứa toàn bộ khóa, feature từng dòng, manifest/raw hash/môi trường của gói S2, nên chưa nghiệm thu cặp gói hoặc bốn máy.

## File bằng chứng và bước bàn giao G1

Các kết quả lưu ngoài Git:

```text
data/features/g1_s2_raw_stats/
outputs/g1/S2_acceptance.json
outputs/g1/S2_structure.json
outputs/g1/compare_nam_summary.json
```

Báo cáo nghiệm thu có `status: passed`, `scope: g1_raw_stats_single_package`, `four_machine_acceptance: false`. Báo cáo so Nam ghi rõ chỉ so tóm tắt và chưa so giá trị từng dòng. Những gói S2 từ các lượt cũ vẫn được giữ ngoài Git, không trộn với lượt source hiện tại.

Cách chạy lại và đối chiếu gói từ máy khác nằm trong [bàn giao G1](G1_NHU_BAN_GIAO.md). Phần G1 trên máy Như đã có đầu ra; còn chờ gói đầy đủ của các máy để nghiệm thu chung. Chưa triển khai G2.
