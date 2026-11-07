# Kiểm tra S2 sinh lý trên máy Mai — 07/11/2026

## Kết luận

Đã chạy lại S2 qua `wesad.preprocessing_physiological:prepare_subject` và extractor hiện tại. Kiểm tra tích hợp/cấu trúc đạt; checksum CSV trùng giá trị Như công bố. Chưa nghiệm thu chất lượng sinh lý hoặc cho phép ghép các gói khác chữ ký. Không thay bộ raw112, phương pháp, ngưỡng hay phiên bản feature trong lượt này.

## 1. Lượt chạy và đối chiếu

- Cấu hình: `configs/features_g2.yaml`, đường dẫn raw từ `configs/local.yaml`; Python 3.12.14.
- Đầu ra ngoài Git: `data/features/g2_physiological68_S2_mai_review_20261107/`.
- 395 cửa sổ, 68 feature + 8 metadata; nhóm ngực 37, cổ tay 31.
- Nhãn 1/2/3: 219/113/63. Có 1.204 ứng viên, loại 809 do nhãn; không loại thêm do thiếu mẫu, nonfinite hoặc biên lọc.
- Schema/thứ tự cột, khóa, lưới 60/5, độ thuần nhãn, nhãn nhị phân, cờ lý do thiếu, manifest và checksum được kiểm tra bằng `package_report`. Không có khóa trùng hoặc Inf; 3.046 giá trị NaN có lý do.
- Bảy metadata đầu khớp bảng S2 raw112 trên máy Mai.
- SHA-256 CSV: `aaec0dea4f42114b6574c845b76319237fef8ac96273ad92c196608c9e48eba9`, trùng checksum trong `reports/g2_preprocessing_nhu/S2_provider_basic_verification.json`.
- Raw hash, cấu hình xử lý, số thiếu từng feature, cờ chất lượng và QA đều khớp báo cáo Như.
- Không có gói CSV sinh lý S2 nguyên bản của Như tại máy Mai; đối chiếu dựa trên báo cáo/checksum đã commit, chưa kiểm tra độc lập toàn bộ gói gốc.
- Toàn bộ checkout hiện tại: **72/72 test đạt**; log tại `outputs/g2_s2_mai_review/tests.txt`.

Mã nguồn và môi trường chưa có cùng chữ ký: code hash máy Mai `469b9ef1575aba4881425ee4982caa9f8f2f72030b7c1b351360b9beb5bcabae`, Như `abd964e88fa0e5922c8bfc2cb71e1f4ad46ebb1d381e3f6bf82571817e1b452b`. Checkout Mai còn thay đổi trước lượt này trong `src/wesad/extraction.py`: giữ hợp đồng trong manifest/schema và phục hồi khi resume, đọc CSV round-trip. Python cùng bản 3.12.14 và sáu phiên bản thư viện ghi nhận khớp nhau, nhưng bản build standalone của Mai khác conda-forge của Như. Git metadata trong gói Mai là null vì tiến trình chạy trong sandbox không đọc được Git root ở thư mục cha; không sửa manifest để giả thông tin nguồn chạy. HEAD quan sát riêng bằng Git là `7720b01`, checkout có thay đổi cục bộ. CSV khớp không đồng nghĩa đạt tái lập nghiêm ngặt cùng nguồn/môi trường.

## 2. Rà BVP

Đã kiểm tra BVP raw S2, lọc 0.5–5 Hz và phát hiện nhịp bằng đúng helper/ngưỡng hiện tại. BVP có 389.056 mẫu, không có NaN/Inf ở đầu vào; điều này chưa chứng minh chất lượng sinh lý.

Sáu feature nhịp/IBI BVP cùng thiếu ở **336/395 cửa sổ (85,1%)**. Các feature năng lượng dải tần BVP vẫn có giá trị. Kết quả thiếu được tái tạo đúng từ các peak phát hiện, không phải lỗi nối provider hoặc ghi CSV.

| Điều kiện không đạt đồng thời | Số cửa sổ |
| --- | ---: |
| Chỉ tỷ lệ khoảng nhịp hợp lệ dưới 80% | 231 |
| Tỷ lệ dưới 80%, dưới 20 khoảng hợp lệ, dưới 10 cặp kề hợp lệ | 67 |
| Tỷ lệ dưới 80% và dưới 10 cặp kề hợp lệ | 37 |
| Tỷ lệ dưới 80% và dưới 20 khoảng hợp lệ | 1 |
| Đạt tất cả điều kiện | 59 |

Khoảng hợp lệ phải nằm trong 0.3–2 giây và lệch không quá 20% trung vị khoảng trong miền đó. Đây là heuristic hiện tại; không có peak chuẩn độc lập để xác định các khoảng bị loại là sai thật hay do detector/quy tắc quá chặt.

| Nhãn | Tổng cửa sổ | Thiếu sáu feature nhịp/IBI BVP |
| --- | ---: | ---: |
| Baseline (1) | 219 | 160 |
| Stress (2) | 113 | 113 |
| Amusement (3) | 63 | 63 |

Thiếu phụ thuộc mạnh vào nhãn trong S2; không thể coi đây là thiếu rải rác rồi tuyên bố bộ BVP đã sẵn sàng cho so sánh cảm biến.

Detector dùng prominence bằng 0.5 độ lệch chuẩn trên cả đoạn hữu hạn đã lọc. S2 có độ lệch chuẩn toàn đoạn 71.0108, ngưỡng 35.5054; tỷ lệ độ lệch chuẩn toàn đoạn/cửa sổ có trung vị 1.5653, lớn nhất 5.0829. Ngưỡng chung vì vậy có thể ảnh hưởng khác nhau giữa các cửa sổ có biên độ khác nhau, nhưng chưa chứng minh đây là nguyên nhân duy nhất.

Thử nhạy cảm ngoài pipeline bằng phát hiện riêng trong mỗi cửa sổ với prominence cục bộ: đạt 72 thay vì 59 cửa sổ; cứu được 19 nhưng mất 6 cửa sổ từng đạt. Thử này thay cả biên phát hiện lẫn phạm vi tính prominence, chỉ là chẩn đoán; không dùng làm feature mới hoặc phương án sửa đã xác nhận. Dữ liệu theo cửa sổ: `outputs/g2_s2_mai_review/bvp_windows.csv`; tổng hợp: `bvp_diagnosis.json` cùng thư mục.

Hai feature nhiệt độ ngực S2 vẫn thiếu có chủ đích do `suspect_contact`. SCR không có sự kiện hợp lệ thì biên độ thiếu có cờ; không điền 0 thay cho biên độ.

## Bước tiếp theo

1. Mai/Như rà bằng mắt các đoạn BVP có peak được đánh dấu, đối chiếu thời điểm và ECG như nguồn tham khảo; ECG cũng không phải peak chuẩn đã gán nhãn. Phân biệt bỏ sót peak, peak nhiễu và biến thiên khoảng nhịp trước khi sửa detector/quy tắc.
2. Thống nhất giữ heuristic với hạn chế đã công bố hay thay phương pháp. Nếu đổi định nghĩa, cập nhật hợp đồng/phiên bản riêng, chạy lại S2 và các test liên quan; không trộn với gói cũ.
3. Trước khi chia lô sinh lý cho bốn người, dùng cùng nguồn sạch và môi trường thống nhất. Lô Mai là S5/S6/S10. Chưa mở rộng 15 người trong lượt kiểm tra này.

Chẩn đoán này hoàn thành việc chạy/kiểm tra S2 và tìm điều kiện gây thiếu BVP; chưa giải quyết xong xác thực detector sinh lý. Không chạy G3, không thêm phần RAM, không commit/push hoặc đưa dữ liệu lớn lên Git.
