# Bàn giao G1 phần Mai

Cập nhật ngày 05/11/2026. Phần việc theo PHAN_CONG_MAY_HOC.docx: chốt danh mục feature, đơn vị, nhãn nhị phân, cách đánh dấu thiếu và schema đầu ra. Chỉ hoàn thiện thiết kế G1 và gói S2 cục bộ; chưa triển khai extractor sinh lý hay chạy lô G2.

## Năm file bàn giao trên Git

| File | Nội dung |
| --- | --- |
| `README.md` | Mục bàn giao Mai và liên kết tới đầu ra G1 |
| `configs/feature_schema.json` | Hợp đồng sinh lý 68 feature, metadata, công thức/thông số, đơn vị và bằng chứng đơn vị gộp trong unit_policy.unit_evidence |
| `docs/G1_DANH_MUC_FEATURE.csv` | Danh mục 68 feature, đúng tên/thứ tự/đơn vị/phương pháp như JSON |
| `docs/G1_HOP_DONG_FEATURE.md` | Quy ước nhãn, NaN, cửa sổ, feature và phối hợp |
| `docs/BAN_GIAO.md` | Trạng thái, kết quả tự kiểm tra S2 và hướng dẫn gửi gói cho Như |

Thiết kế 68 feature gồm 37 ngực, 31 cổ tay và 8 metadata, trạng thái design_locked_implementation_not_validated. Đây là thiết kế phục vụ proposal để triển khai tiếp; processor raw hiện tại của nhóm tạo 112 thống kê (63 ngực, 49 cổ tay), không áp detector/PSD sinh lý trong hợp đồng này. Schema raw thực tế lấy từ file sinh cùng gói CSV ngoài Git.

Nhãn nhị phân 2→1, 1/3→0; loại các nhãn khác. Feature sinh lý thiếu dùng NaN/ô CSV trống kèm lý do, không dùng số 0 giả; metadata không thiếu, Inf là lỗi. Đơn vị và quy tắc chuyển đổi ghi trong schema/hợp đồng; bằng chứng được kiểm tra với README WESAD gốc và mẫu S2 đã căn thời gian, không suy rộng thành kiểm tra toàn bộ 15 người.

## Kiểm tra S2 và người tổng hợp

Phân công yêu cầu cả bốn chạy mẫu S2. Mai đã tự kiểm tra gói máy mình để phát hiện lỗi cột/nhãn/checksum trước khi bàn giao. Theo cách nhóm người dùng đã xác định, Như nhận ZIP và tổng hợp đối chiếu bốn máy. Mai không thay Như ghi nhận nghiệm thu chung.

- Gói máy Mai: `data/features/g1_s2_raw112_py31214_fa7b694/`.
- Python 3.12.14; đủ 16 thư viện khớp file khóa.
- Source commit: `fa7b6942d69717cf6ea7f48ca364f576486de08f`.
- Gói cục bộ passed: 395 cửa sổ, 8 metadata + 112 feature; Baseline 219, Stress 113, Amusement 63.
- Trong 1.204 cửa sổ ứng viên: 809 loại theo nhãn, 0 loại vì thiếu mẫu/nonfinite; giữ 395.
- Đã kiểm tra cột, khóa, nhãn, thời gian, feature hữu hạn, môi trường, chữ ký mã/config, hash raw/CSV/schema, bảng ghép, số lượng QA và đẳng thức thống kê.
- Báo cáo chỉ có phạm vi g1_raw112_single_package; four_machine_acceptance=false, physiological_acceptance=false.

### Bằng chứng kết quả máy Mai

Kết quả: **passed**, phạm vi `g1_raw112_single_package`, phiên bản `all_sensors_raw_stats_v1`. Đây là tự kiểm tra một gói trước khi bàn giao cho Như; chưa thực hiện so sánh gói của các máy khác.

| Thông tin | Giá trị |
| --- | --- |
| Raw S2 SHA-256 | `36ef5e8afc0f91998eefba7c12fc9fa97b7b07198cbec0126917d7abb436ca23` |
| Code SHA-256 | `b0d02f7a9f80e754342c8fa73f7b6d2e57a2b772a5e01882517b7ef6bc739a96` |
| S2.csv SHA-256 | `14bd24e6a471f58693e7822c03cc7b492eeba314dab290bc10a8281285d8384a` |
| Dung sai dùng kiểm tra đẳng thức thống kê raw | rtol=1e-07, atol=1e-09 |
| Số feature ngực / cổ tay / kết hợp | 63 / 49 / 112 |
| Nghiệm thu bốn máy | Chưa; `four_machine_acceptance=false` |

Các thư viện khoa học ghi trong gói: numpy 2.5.4, pandas 3.1.0, PyYAML 6.0.3, scipy 1.18.1, scikit-learn 1.9.1, imbalanced-learn 0.14.2. Toàn bộ 16 phiên bản đã được kiểm tra khớp file khóa. Dung sai trên dùng trong kiểm tra gói cục bộ, chưa phải kết quả so giá trị giữa các máy.

Gửi Như nguyên thư mục gói ở trên dưới dạng ZIP ngoài Git, giữ đủ CSV/schema/config/manifest/checksum. Không gửi .venv-g1 hoặc bản Python đã cài. Không dùng gói 21 feature cũ để đối chiếu với gói 112.

Repo đã pull tới 9fcbbf7; gói này được tạo bằng fa7b694 trước lần pull đó. Như cần kiểm tra cùng code/source hash trước khi so các gói; nếu khác source thì thống nhất phiên bản chạy lại. Không sửa manifest/chữ ký để ép khớp. Chưa có kết quả đối chiếu bốn máy trong phần bàn giao này.

## Kiểm tra thiết kế đã thực hiện

JSON/CSV khớp 68 feature; 76 tên cột duy nhất gồm metadata; các nhóm 37/31 và tham chiếu phương pháp hợp lệ. Chốt công thức detector/ngưỡng/PSD, NaN không SCR, ACC/EMG 12 đoạn 5 giây và đơn vị. Extractor sinh lý đúng hợp đồng chưa triển khai/kiểm chứng, không coi bộ thử 175 cột là sản phẩm G1 này.
