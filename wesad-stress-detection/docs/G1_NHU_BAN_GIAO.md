# Bàn giao G1 của Như sau cập nhật Mai

Ngày 03/11/2026. Phạm vi chỉ G1: môi trường, thiết kế đồng bộ/bộ lọc, cửa sổ 60/5 majority 90%, test biên nhãn và chạy mẫu S2. Chưa triển khai bộ lọc hoặc thuật toán feature G2.

## Đã khép các phần có thể làm trên máy Như

| Hạng mục | Bằng chứng |
| --- | --- |
| Mai chọn thiết kế của Như | Commit `b5a0cb8`, mục `preprocessing_design` trong schema Mai |
| Hợp đồng thiết kế có cấu trúc | `configs/preprocessing_g1_design.yaml`, stage G1, runtime_ready false |
| Bổ sung thiết kế còn thiếu | Biên EDA 10 s, Resp 30 s; QA cửa sổ chung; giữ trọng lực ACC; không đổi scale khi chưa có bằng chứng |
| Đồng nhất phiên bản Python trên máy Như với bản bàn giao Thái | `.venv-g1` Python 3.12.14; `.python-version`; cài đúng `requirements.lock.txt`, pip check đạt |
| Test hạ tầng/biên nhãn và công cụ nghiệm thu | 19/19 đạt, gồm 5 test mới cho đối chiếu gói G1 |
| Mẫu S2 trên môi trường mục tiêu | 395 cửa sổ: Baseline 219, Stress 113, Amusement 63; kiểm tra gói đạt |

Mai đã chọn thiết kế ban đầu; không suy ra Mai đã review phần biên/QA bổ sung này. Những giá trị thiết kế chưa được chứng minh chất lượng sinh lý. Các chi tiết detector/PSD/chuyển đơn vị gắn với feature vẫn cần Mai/Như thống nhất trước triển khai G2, không được tự đánh dấu đã hoàn thành.

## Cách chạy lại G1

Máy Như đã có `.venv-g1` Python 3.12.14, không cần cài lại. Máy khác tạo venv bằng Python đúng phiên bản này, cài `requirements.lock.txt`, cài editable như [thiết kế G1](G1_NHU_THIET_KE_TIEN_XU_LY.md).

Chạy tại thư mục `wesad-stress-detection`:

```powershell
.\.venv-g1\Scripts\python.exe scripts/extract_features.py --config configs/features_nhu_smoke.yaml --local configs/local.yaml --subjects S2 --resume
.\.venv-g1\Scripts\python.exe scripts/check_g1_sample.py --features-dir data/features/nhu_g1_S2_py312_raw_stats_majority_v1 --output outputs/g1/S2_py312_acceptance.json
```

Lượt này gọi extractor raw có sẵn, không đọc config thiết kế để lọc tín hiệu. Các config/header/code khác nhau cần output_dir mới; không sửa manifest để tái sử dụng output cũ.

## Đối chiếu mẫu giữa các máy

Mỗi thành viên dùng cùng source trên main, Python 3.12.14, thư viện khóa, raw S2 và config smoke. Giữ nguyên cả gói gồm `S2.csv`, `S2.json`, `config.json`, `feature_schema.json`, `run_status.json`; chia sẻ ngoài Git. Những gói cũ có source hash hoặc môi trường khác phải chạy lại, không bỏ qua kiểm tra chỉ vì cùng số cửa sổ.

Máy Như có thể đặt gói nhận từ máy khác vào `data/features/g1_reference_mai/` (hoặc tên thành viên tương ứng), rồi chạy:

```powershell
.\.venv-g1\Scripts\python.exe scripts/check_g1_sample.py --features-dir data/features/nhu_g1_S2_py312_raw_stats_majority_v1 --reference-dir data/features/g1_reference_mai --output outputs/g1/compare_nhu_mai.json
```

Đây là đường dẫn mẫu cho gói thực sự nhận được, chưa có gói Mai trên máy Như. Lặp lại với gói Thái và Nam khi có. Công cụ kiểm tra:

- Raw/source hash, thông số xử lý, Python patch version và thư viện trong snapshot phải trùng.
- Header metadata, tên/thứ tự 21 feature phải đúng; nhãn, mốc thời gian, quality flag và tỷ lệ nhãn phải khớp theo khóa `window_id`.
- Feature phải hữu hạn; so sánh `abs(a-b) <= 1e-9 + 1e-7*abs(b)`. Đây là dung sai bàn giao cho thống kê raw G1; không dùng làm dung sai mặc định cho 68 feature sinh lý.
- Mỗi gói phải complete, CSV khớp checksum/manifest; cửa sổ dài 60 giây, bắt đầu trên lưới 5 giây, tỷ lệ nhãn >=90% và window_id dựa trên chỉ số 700 Hz gốc.
- Số 395 là quan sát của lượt S2 này, không được mã hóa thành điều kiện bắt buộc để che lỗi phương pháp.

Một báo cáo passed chỉ chứng minh gói đơn hoặc cặp gói đã được kiểm tra. Công cụ luôn giữ `four_machine_acceptance: false` và từ chối so sánh một thư mục với chính nó. Không có cơ chế xác thực danh tính máy; cần các thành viên xác nhận nguồn gói để nghiệm thu nhóm.

## Trạng thái nghiệm thu

| Máy | Trạng thái có bằng chứng |
| --- | --- |
| Như | Gói S2 Python 3.12.14 đã kiểm tra đạt |
| Thái | Có bản bàn giao trước đó, chưa nhận gói chạy cùng source G1 hiện tại để so sánh trực tiếp |
| Mai | Đã có schema/chốt cơ sở thiết kế, chưa nhận gói S2 cùng source để so sánh |
| Nam | Chưa nhận gói S2 cùng source để so sánh |

Phần chuẩn bị G1 của Như đã có đầu ra. Nghiệm thu bốn máy còn chờ gói và xác nhận của các thành viên; không thể thay bằng chạy nhiều môi trường trên một máy. Chưa triển khai G2.
