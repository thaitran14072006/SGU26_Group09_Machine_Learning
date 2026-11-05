# Bàn giao Mai Giai đoạn 1 ngày 03/11/2026

Phạm vi theo G1 của PHAN_CONG_MAY_HOC.docx, người phụ trách Nguyễn Thị Trúc Mai.

## Đầu ra

- `configs/feature_schema.json`: schema nháp có metadata, danh mục/thứ tự feature, kiểu dữ liệu, đơn vị, ba nhóm cảm biến và trạng thái các điểm cần chốt.
- `docs/G1_DANH_MUC_FEATURE.csv`: danh mục cùng tên/thứ tự với schema.
- `docs/G1_HOP_DONG_FEATURE.md`: định nghĩa feature, nhãn nhị phân, cách ghi thiếu, chất lượng và giao diện phối hợp.

Schema nháp gồm **8 metadata + 68 feature dự kiến** (37 ngực, 31 cổ tay). Nhóm kết hợp giữ thứ tự ngực rồi cổ tay. Đây là danh mục G1 cần đối chiếu trước khi triển khai, không phải 175 cột của bộ thử trước đó.

Đã chuẩn bị phần G1 của Mai. Trạng thái schema là **draft_pending_feature_details_and_s2_acceptance**: đã chốt dùng thiết kế tiền xử lý G1 của Như làm đầu vào theo yêu cầu người dùng; còn xác minh scale/đơn vị raw, dải tần/detector và ngưỡng QA sinh lý chưa được quy định đầy đủ; vùng biên, QA cửa sổ và dung sai raw G1 đã lấy theo bản Như mới nhất. Không đánh dấu nghiệm thu chung G1 hoặc S2 bốn máy khi chưa có bằng chứng.

## Đúng tên file theo phân công

Schema trong Git là `configs/feature_schema.json`; schema của từng gói feature G2 sẽ là `data/features/<feature_version>/feature_schema.json`. Hai file có mục đích khác nhau: bản G1 là hợp đồng dự kiến, bản G2 xác nhận bảng thực tế của một phiên bản đã khóa.

Module phụ trách theo phân công: `features.py` (G2), `baselines.py` và `metrics.py` (G3). G1 không tạo mô hình hoặc chạy lại 15 người. Các thử nghiệm sinh lý trước khi đọc phân công là bản thử riêng, không thay đầu ra G1 này.

## Kiểm tra đã thực hiện

Đọc được JSON/CSV; tên cột duy nhất, metadata tách khỏi feature; prefix và nhóm ngực/cổ tay/kết hợp thống nhất; mỗi feature có đơn vị/định nghĩa; thứ tự CSV trùng schema; có đầy đủ các nhóm yêu cầu và trạng thái pending.

## Bàn giao tiếp

Schema sử dụng metadata, khóa cửa sổ, nhãn và giao diện processor của hạ tầng Thái đã có. Tiền xử lý lấy theo thiết kế G1 Như đã bàn giao tại `G1_NHU_THIET_KE_TIEN_XU_LY.md`: đồng bộ tần số gốc, cửa sổ 60/5, EDA/RESP theo bộ lọc đề xuất của Như, ACC/EMG 12 đoạn 5 giây rồi gộp. Mai làm feature/schema theo đầu vào đó, không lập nhánh tiền xử lý riêng. Các thông số Như chưa quy định vẫn để mở trước khi khóa G2; bổ sung bất thường từ khảo sát Nam nếu ảnh hưởng feature/QA.

Chưa gửi tin nhắn, chưa tạo PR hay ghi nhận review thay thành viên.

## Thông số thiết kế G1 đã chọn theo Như

- Cửa sổ 60 giây, bước 5 giây, majority ≥90%; giữ tần số gốc từng kênh và cùng mốc PKL.
- EDA ngực low-pass 5 Hz; RESP ngực band-pass 0,1–0,35 Hz; EDA cổ tay low-pass 1 Hz.
- Butterworth bậc thiết kế 4, SOS, lọc hai chiều offline trên đoạn hữu hạn liên tục trước khi cắt cửa sổ.
- ACC giữ trọng lực, TEMP giữ tín hiệu gốc; không chuyển scale khi chưa có bằng chứng. ACC/EMG chia 12 đoạn 5 giây rồi lấy trung bình feature.
- Nhận revision `nhu_g1_design_v2` trong `configs/preprocessing_g1_design.yaml`: biên EDA 10 giây, RESP 30 giây, odd padding; loại cửa sổ thiếu mẫu/NaN/Inf ở kênh yêu cầu hoặc chồng biên, ghi lý do và giữ chung khóa cho các so sánh cảm biến.
- Gói raw G1 dùng Python 3.12.14, thư viện khóa, rtol=1e-7 và atol=1e-9. Đã kiểm tra gói S2 máy Mai bằng source Git d85d9e7 và môi trường khóa (kết quả bên dưới); không áp dung sai raw cho 68 feature sinh lý.

Đây là chốt lựa chọn thiết kế G1 theo bản Như đã đưa vào repo, chưa phải nghiệm thu bộ lọc trên tín hiệu thật. Các thông số chưa có trong bản Như (detector, PSD, scale, ngưỡng QA sinh lý) vẫn được ghi rõ còn mở để hoàn thiện trước lượt chính. Chưa chạy G2.

## Cập nhật ngày 05/11/2026 theo README và EDA mới

Đã đọc README hiện tại, bản phân công và proposal v3 được cung cấp trên máy; giữ nguyên các tài liệu Word. Phần Mai tiếp tục dùng thiết kế `nhu_g1_design_v2` và nhận kết quả EDA mới của Nam, không còn ghi chờ Nam khảo sát G1. Bằng chứng và giới hạn EDA đã được bổ sung vào schema và hợp đồng feature.

Đã kiểm tra trực tiếp JSON/CSV: đủ 68 feature duy nhất, 8 metadata, 37 feature ngực + 31 cổ tay; tên/thứ tự/kiểu/đơn vị/định nghĩa/cách gộp khớp danh mục; nhóm combined và ánh xạ nhị phân nhất quán. Đây là phần danh mục và schema nháp của Mai theo G1.

README hiện chạy gói `chest_raw_stats_majority_v1` gồm 21 feature; schema thực tế của gói phải lấy từ thư mục output. Không thay bằng schema nháp 68 feature trong `configs/` và không ép số cửa sổ theo bảng EDA.

Đã tạo `.venv-g1` Python 3.12.14, kiểm tra đủ 16 thư viện khớp `requirements.lock.txt`, cài dự án editable và kiểm tra tương thích bằng uv đạt. Mẫu S2 và 25/25 test đã đạt như phần bằng chứng bên dưới. Chưa có gói S2 từ các máy khác để đối chiếu, nên chưa đánh dấu toàn nhóm hoàn tất G1.

## Bằng chứng mẫu S2 trên máy Mai — Python 3.12.14

- Môi trường: `.venv-g1`; đủ 16 phiên bản khớp file khóa, kiểm tra tương thích bằng `uv pip check` đạt.
- Chạy bản mã được xuất từ Git commit `d85d9e7d766cc164b4047ac2be939deaff9e8a5a` tại `outputs/g1/source_d85d9e7/wesad-stress-detection` để không lẫn mã thử nghiệm chưa commit trong workspace.
- Bộ test G1/hạ tầng của bản mã đó: 25/25 đạt. Log: `outputs/g1/tests_py31214.txt`; môi trường: `outputs/g1/environment_py31214.json`.
- Gói: `data/features/g1_s2_raw_stats_py31214_d85d9e7/`; 395 cửa sổ, 8 metadata + 21 thống kê raw; Baseline 219, Stress 113, Amusement 63.
- Báo cáo: `outputs/g1/S2_acceptance_py31214.json`, `status: passed`, `scope: g1_raw_stats_single_package`, `four_machine_acceptance: false`.
- Raw SHA-256: `36ef5e8afc0f91998eefba7c12fc9fa97b7b07198cbec0126917d7abb436ca23`; code SHA-256: `44cf7eef4d541670b04e5f571f9df94009451a1b808174e549d6b27bbaa0f425`.

Dùng môi trường mới cho dự án bằng `.\.venv-g1\Scripts\Activate.ps1`, rồi `python --version` phải hiện 3.12.14. Python Anaconda cũ vẫn được giữ. Python nền nằm trong `outputs/g1/python/`; cần giữ thư mục đó để `.venv-g1` hoạt động. Raw, môi trường và output không commit lên Git.

Lệnh chạy lại đúng bản mã đã kiểm tra (các đường dẫn local riêng đã nằm trong config local bị Git bỏ qua):

```powershell
.\.venv-g1\Scripts\python.exe outputs/g1/source_d85d9e7/wesad-stress-detection/scripts/extract_features.py --config configs/features_smoke.yaml --local configs/local.yaml --subjects S2 --output-dir D:/mayhoc/Machine-learning/wesad-stress-detection/data/features/g1_s2_raw_stats_py31214_d85d9e7 --resume
.\.venv-g1\Scripts\python.exe outputs/g1/source_d85d9e7/wesad-stress-detection/scripts/check_g1_sample.py --features-dir D:/mayhoc/Machine-learning/wesad-stress-detection/data/features/g1_s2_raw_stats_py31214_d85d9e7 --output D:/mayhoc/Machine-learning/wesad-stress-detection/outputs/g1/S2_acceptance_py31214.json
```

Bàn giao cả thư mục gói S2 ngoài Git để các thành viên đối chiếu. Cần cùng raw/code/Python/thư viện/config; số cửa sổ và phân bố lớp bằng nhau chưa đủ xác nhận các giá trị feature. Chưa nhận gói khác để chạy so sánh; không lấy bản sao gói của chính máy này làm nghiệm thu nhiều máy.
