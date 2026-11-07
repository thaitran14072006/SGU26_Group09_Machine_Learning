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

## 3. Rà đồ thị BVP và đối chiếu ECG sau lượt kiểm tra đầu

Đã chạy `scripts/diagnose_s2_bvp.py` bằng Python 3.12.14 với provider hiện tại. Mã kiểm tra tái tạo đúng trạng thái thiếu của cả 395 cửa sổ trong CSV đã kiểm tra. Bổ sung bảng so sánh mọi cửa sổ, năm ví dụ chọn xác định trước và đồ thị raw/filtered BVP, peak, khoảng nhịp nhận/loại và ECG cùng thời gian. Dữ liệu và ảnh nằm ngoài Git tại `outputs/g2_s2_bvp_visual_review/`.

| Ví dụ | Bắt đầu (s) | Peak BVP / ECG trong 60s | Khoảng BVP hợp lệ | Nhận xét khi rà hình |
| --- | ---: | ---: | ---: | --- |
| Baseline đạt | 1095 | 72 / 71 | 85,9% | Đoạn phóng 20–35s có chu kỳ BVP rõ, detector nhận đều; vẫn có nhiễu ở cuối cửa sổ |
| Baseline thiếu, biên độ thấp | 980 | 59 / 70 | 74,1% | Có các đỉnh nhỏ trong đoạn 20–35s bị ngưỡng chung bỏ qua; ngưỡng cục bộ thêm đỉnh nhưng cửa sổ vẫn không đạt |
| Baseline thiếu, biên độ cao | 1395 | 96 / 84 | 27,4% | BVP có nhiều dao động/đỉnh phụ; detector nhận cả các đỉnh gần nhau, khoảng nhịp không ổn định |
| Stress thiếu | 2550 | 46 / 74 | 44,4% | BVP có biến động biên độ mạnh và đoạn ít dao động rõ, tạo khoảng giữa peak lên đến nhiều giây; ECG trong đoạn phóng có peak rõ hơn |
| Amusement thiếu | 4915 | 40 / 64 | 33,3% | Nhiều chu kỳ biên độ nhỏ bị ngưỡng chung bỏ qua; thử cục bộ tăng peak lên 70 nhưng vẫn không đạt điều kiện khoảng nhịp |

Trên toàn bộ S2, ECG vượt các kiểm tra heuristic ở 395/395 cửa sổ. Trung vị số peak BVP/ECG theo nhãn: baseline 71/72, stress 54/76, amusement 56/67. Peak ECG chưa có kiểm chứng độc lập; không coi số peak ECG là số nhịp thật, không ghép từng peak theo chỉ số và không thay feature BVP bằng ECG.

**Kết luận từ bằng chứng hiện có:** detector BVP với ngưỡng prominence toàn đoạn có dấu hiệu bỏ sót chu kỳ nhỏ ở một số đoạn; ở đoạn khác lại nhận các đỉnh phụ/dao động không ổn định. Dạng sóng thô cũng biến dạng ở một số đoạn, nên chưa thể quy toàn bộ vấn đề cho bộ lọc hoặc chỉ sửa một ngưỡng. Cửa sổ đạt cũng có thể chứa đoạn nhiễu; vượt heuristic không tương đương đã xác nhận chất lượng sinh lý.

Đã thử riêng năm ví dụ với **cùng toàn đoạn đầu vào, khoảng cách peak và mask biên**, chỉ thay prominence bằng 0.5 độ lệch chuẩn của cửa sổ đang xét. Bốn ví dụ thiếu vẫn thiếu, ví dụ đạt vẫn đạt. Đây là thử có kiểm soát hơn thử cắt riêng từng cửa sổ ở mục 2; không thay thế kết quả lịch sử và không sửa CSV. Không có bằng chứng đủ để chỉ hạ ngưỡng rồi công nhận bộ BVP hoàn tất.

Ảnh đã xem: `baseline_pass.png`, `baseline_fail_low_amplitude.png`, `baseline_fail_high_amplitude.png`, `stress_fail.png`, `amusement_fail.png`. Các điểm đen là peak hiện tại; vòng cam là thử prominence cục bộ; khoảng đỏ là khoảng bị loại. Panel BVP và ECG phóng cùng 20–35 giây trong cửa sổ. Ảnh được render bằng matplotlib của Python hệ thống từ mảng đã xuất bằng môi trường 3.12.14; Python hệ thống không tính feature/provider.

Bằng chứng số nhỏ được lưu tại `reports/g2_preprocessing_mai/S2_bvp_visual_diagnosis.json`. Lệnh tái tạo dữ liệu chẩn đoán (chọn thư mục output mới):

```powershell
.\.venv-g1\Scripts\python.exe -X utf8 scripts/diagnose_s2_bvp.py --raw data/raw/S2/S2.pkl --package data/features/g2_physiological68_S2_mai_review_20261107 --output outputs/g2_s2_bvp_visual_review_lan2
```

## Bàn giao tiếp theo

1. Phần cần thống nhất về tiền xử lý và detector BVP để Như chạy, chốt và bàn giao một phiên bản dùng chung. Đây là hướng làm việc Mai yêu cầu, chưa phải xác nhận Như đã hoàn thành hoặc nhận việc.
2. Mai lấy đúng mã nguồn, cấu hình, phiên bản/schema và kết quả Như bàn giao làm cơ sở để tiếp tục extractor, schema và hợp nhất. Không yêu cầu Mai chạy lại để đối chiếu nhiều máy như một điều kiện bàn giao. Các lượt chẩn đoán ở trên là bằng chứng lịch sử, không phải quy trình đối chiếu bắt buộc tiếp theo.
3. Khi nhận gói, Mai kiểm tra schema, manifest/checksum, subject, nhãn, khóa trùng và NaN/Inf có lý do. Nếu Như đổi định nghĩa feature, cập nhật hợp đồng/phiên bản tương ứng trước khi dùng; không ghép lẫn gói cũ. Chưa coi BVP đã được xử lý xong chỉ vì đã chọn một nguồn dùng chung.
4. Nếu nhóm tiếp tục chia lô, Mai chạy S5/S6/S10 bằng đúng phiên bản Như bàn giao; nếu nhận kết quả đã đủ subject thì kiểm tra và dùng trực tiếp, không chạy lặp để đối chiếu. Chưa mở rộng 15 người trong lượt kiểm tra này.

Chẩn đoán này hoàn thành việc chạy/kiểm tra S2 và tìm điều kiện gây thiếu BVP; chưa giải quyết xong xác thực detector sinh lý. Không chạy G3, không thêm phần RAM, không commit/push hoặc đưa dữ liệu lớn lên Git.
