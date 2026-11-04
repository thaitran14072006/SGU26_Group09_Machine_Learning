# Thiết kế tiền xử lý G1 của Như

Ngày lập: 03/11/2026. Người phụ trách: Phạm Quỳnh Như.

Mục tiêu G1 là chuẩn bị môi trường, kiểm tra quy tắc cửa sổ và xác định hợp đồng tiền xử lý để đối chiếu với Mai. Phạm vi bàn giao của Như trên main chỉ gồm thiết kế và cấu hình chạy thử G1. Chưa triển khai bộ lọc, cửa sổ đa tần số hoặc helper ACC/EMG của G2.

Cập nhật theo commit Mai `b5a0cb8`: Mai đã chọn thiết kế này làm cơ sở cho schema G1. Bản hợp đồng có cấu trúc nằm ở `configs/preprocessing_g1_design.yaml`, chỉ dùng để bàn giao thiết kế (`runtime_ready: false`), không truyền cho extractor. Phần bổ sung về biên/QA dưới đây là quyết định thiết kế Như gửi tiếp; chưa ghi nhận Mai đã review bổ sung hoặc nghiệm thu tín hiệu.

## 1. Quy tắc cửa sổ đã có trong code

- Giữ trục thời gian gốc, bắt đầu từ 0 giây.
- Cửa sổ 60 giây, bước 5 giây; bỏ phần cuối không đủ 60 giây.
- Tính nhãn trên toàn cửa sổ nhãn gốc ở 700 Hz: 42.000 mẫu.
- Giữ nhãn đa số thuộc 1, 2, 3 và có tỷ lệ >= 0,90. Nhãn 0, 4 và nhãn không dùng vẫn nằm trong mẫu số; không nối các đoạn sau khi bỏ nhãn.
- Nhị phân: Stress (2) -> 1; Baseline (1), Amusement (3) -> 0.
- Khóa cửa sổ dựa trên subject và mốc bắt đầu gốc, không đánh số lại theo số cửa sổ còn lại của từng thiết bị.

`preprocessing.iter_windows` hiện yêu cầu tín hiệu đã cùng độ dài và tần số với nhãn. Giữ hàm này cho smoke test ngực. G2 cần thêm cách cắt theo mốc thời gian cho các kênh khác tần số, không truyền trực tiếp tín hiệu cổ tay vào hàm hiện tại.

## 2. Thiết kế đồng bộ cần chốt trong G1

| Thiết bị | Kênh | Tần số theo proposal |
| --- | --- | ---: |
| Ngực | ECG, EDA, EMG, Resp, Temp, ACC | 700 Hz |
| Cổ tay | BVP | 64 Hz |
| Cổ tay | EDA, TEMP | 4 Hz |
| Cổ tay | ACC | 32 Hz |

Đọc tín hiệu đã đồng bộ trong PKL WESAD. Xác minh khóa, shape, thời lượng và mốc chung trên S2 trước khi chấp nhận thiết kế. Không tự thêm offset thời gian khi chưa có bằng chứng.

Tạo danh sách cửa sổ và nhãn một lần từ nhãn gốc. Với mỗi khoảng nửa mở `[start_s, end_s)`, cắt từng kênh ở chỉ số `start_s * fs` đến trước `end_s * fs`. Trên lưới 60/5 giây, các chỉ số ở những tần số trên đều nguyên. Kiểm tra số mẫu, không cắt ngắn âm thầm khi thiếu dữ liệu.

Không cần nâng tất cả kênh lên 700 Hz. Mỗi kênh giữ tần số gốc; chỉ thời gian bắt đầu/kết thúc và khóa cửa sổ cần thống nhất. Đối chiếu thời lượng thực tế giữa các kênh; nếu lệch thì ghi báo cáo và xử lý rõ trước khi ghép.

## 3. Lựa chọn bộ lọc G1 và trạng thái xác nhận

| Kênh | Cấu hình thiết kế | Căn cứ và trạng thái |
| --- | --- | --- |
| EDA ngực 700 Hz | Low-pass 5 Hz | Bài báo gốc, mục 4.1, trang PDF 4 |
| Resp ngực 700 Hz | Band-pass 0,1–0,35 Hz | Bài báo gốc, mục 4.1, trang PDF 4 |
| EDA cổ tay 4 Hz | Low-pass 1 Hz | Mai đã nhận làm lựa chọn G1 ở `b5a0cb8`; chưa nghiệm thu tín hiệu, không phải cutoff được xác minh từ bài báo |
| ECG/BVP, EMG | Chốt cùng thuật toán feature của Mai | Chưa đủ căn cứ để khóa tham số trong G1 |
| ACC/TEMP | Giữ tín hiệu gốc cho thống kê ban đầu | Mai đã nhận làm lựa chọn G1; ACC giữ trọng lực, chưa tự đổi scale |

EDA cổ tay có Nyquist 2 Hz; không áp dụng trực tiếp cutoff 5 Hz lên kênh 4 Hz và không coi nội suy là cách khôi phục thông tin đã mất.

Lựa chọn thiết kế G1 đã được Mai đưa vào schema: Butterworth bậc thiết kế 4, dạng SOS, lọc hai chiều cho phân tích offline. Không khẳng định đây là bậc lọc của bài báo. Chưa triển khai bộ lọc; kiểm tra đáp ứng tần số và tín hiệu thật thuộc bước triển khai sau G1.

Thiết kế biên/QA bổ sung của Như: lọc liên tục từng kênh trước khi cắt cửa sổ, odd padding; không nối các đoạn rời nhau theo nhãn hoặc lọc xuyên NaN/Inf. Bỏ 10 giây mỗi đầu đoạn hữu hạn với EDA ngực/cổ tay, 30 giây với Resp. Đoạn quá ngắn để padding hoặc bỏ biên được đánh dấu không dùng được. Đây là giá trị thiết kế để bàn giao, chưa chứng minh hết quá độ; phải kiểm tra trong G2 trước khi áp dụng lượt chính.

Một cửa sổ thiếu mẫu, có NaN/Inf hoặc giao vùng biên ở bất cứ kênh được yêu cầu nào sẽ bị loại và ghi lý do; dùng chung tập khóa cho so sánh cảm biến. Không tự điền 0 hoặc nội suy. Tín hiệu phẳng/tiếp xúc nghi ngờ được gắn cờ để xem xét, không tự chứng nhận chất lượng sinh lý chỉ vì hữu hạn. Giữ ghi chú nhiệt độ ngực S2 có vấn đề tiếp xúc. Ngưỡng QA sinh lý và kiểm chứng đơn vị tiếp tục phối hợp với Mai trước khi triển khai feature; G1 chưa thực hiện các thuật toán đó.

## 4. Điểm bàn giao với Mai

- Như cung cấp các đoạn tín hiệu đã xử lý theo cùng khóa, nhãn và thời gian; Mai tính feature và schema.
- ACC/EMG: mỗi cửa sổ 60 giây chia thành 12 đoạn 5 giây không chồng lấp; tính feature từng đoạn rồi lấy trung bình theo phân công. Feature EMG chi tiết và đơn vị do Như/Mai chốt trước G2.
- Những feature sinh lý khác tính trên cửa sổ 60 giây theo định nghĩa đã thống nhất.
- Cột có tiền tố thiết bị/cảm biến, ví dụ `chest_eda_*`, `wrist_eda_*` để chọn ngực/cổ tay/kết hợp.
- Metadata: `subject_id`, `window_id`, `start_s`, `end_s`, `label_original`, `label_binary`, `label_fraction`, `quality_flag`.
- Thống nhất một tập cửa sổ dùng chung cho các so sánh cảm biến. Ghi lý do và số cửa sổ bị loại, không để từng mô hình tự chọn mẫu khác nhau.
- Không đưa metadata vào X. Imputer, scaler và chọn feature chỉ fit trong train của từng fold. NaN feature có lý do được ghi nhận; Inf phải điều tra.
- Mai tích hợp processor qua chữ ký `process_subject(path, config) -> DataFrame`. Đầu ra lọc/feature mới phải dùng `feature_version` và `output_dir` mới, không ghi đè thống kê raw.

## 5. Chạy kiểm tra trên máy Như

Mở PowerShell tại thư mục `wesad-stress-detection`. Không cần activate môi trường.

```powershell
# Thư mục tạm riêng, tránh lỗi quyền thư mục tạm khi chạy trong sandbox Windows.
New-Item -ItemType Directory -Force .venv/task-temp | Out-Null
$env:TEMP = (Resolve-Path .venv/task-temp).Path
$env:TMP = $env:TEMP

.\.venv-g1\Scripts\python.exe -m pip check
.\.venv-g1\Scripts\python.exe -m unittest discover -s tests -v
```

Môi trường G1 riêng `.venv-g1` dùng Python 3.12.14, khớp phiên bản trong bàn giao Thái; giữ nguyên `.venv` cũ 3.13.9. File `.python-version` ghi phiên bản mục tiêu cho G1. Cách tạo lại môi trường bằng Python 3.12.14 đã cài trên máy:

```powershell
python -m venv .venv-g1
.\.venv-g1\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv-g1\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
```

Lệnh `python` trong bước tạo phải là Python 3.12.14; kiểm tra `python --version` trước. Trên máy Như đã tạo sẵn, không cần chạy lại. Test hiện có dùng dữ liệu giả: đúng ngưỡng 90%, dưới ngưỡng, nhãn không dùng, không nối đoạn, bỏ phần dư cuối; thêm kiểm tra công cụ đối chiếu G1. Kết quả test không thay cho nghiệm thu raw S2 trên bốn máy.

## 6. Chạy mẫu S2 để kiểm tra G1

1. Đã có raw S2 từ bộ WESAD tải từ nguồn tác giả. CSV đặc trưng không thay thế raw để kiểm tra tín hiệu. Giữ raw ngoài repository.
2. Sao chép `configs/local.example.yaml` thành `configs/local.yaml`; thay `data_root` bằng thư mục thật chứa trực tiếp `S2/`. File local chỉ chứa `data_root` và được Git bỏ qua.
3. Chạy config riêng của Như:

```powershell
.\.venv-g1\Scripts\python.exe scripts/extract_features.py --config configs/features_nhu_smoke.yaml --local configs/local.yaml --subjects S2 --resume
```

Đây là lượt thống kê raw để kiểm tra môi trường và cửa sổ G1, chưa chạy bộ lọc trong bản thiết kế này. Đầu ra: `data/features/nhu_g1_S2_py312_raw_stats_majority_v1/`. Dùng thư mục mới sau khi đổi source để không trộn mã băm với các lượt chạy trước.

4. Kiểm tra `run_status.json` complete cho S2, số cửa sổ theo nhãn, 60 giây/cửa sổ, khóa không trùng và checksum. Complete ở đây chỉ áp dụng subject đã chọn, không chứng minh đủ 15 người.
5. Đối chiếu với Mai/Thái: cùng raw/source, cấu hình xử lý, phiên bản Python 3.12.14 và thư viện đã khóa; tên/thứ tự cột, khóa, nhãn, thời gian phải khớp. Với 21 thống kê raw G1, Như bàn giao dung sai `rtol=1e-7`, `atol=1e-9`; không tự áp dụng dung sai này cho 68 feature sinh lý. Công cụ và cách nhận gói đối chiếu ghi trong [bàn giao G1](G1_NHU_BAN_GIAO.md).
6. G1 dừng ở thiết kế và chạy mẫu. Việc triển khai lọc, cửa sổ đa tần số, helper ACC/EMG và chạy lô `S4 S9 S11 S17` chỉ bắt đầu khi người dùng yêu cầu G2.

## 7. Điều kiện hoàn tất G1 của Như

Kết quả kiểm tra thực tế trên máy Như ngày 03/11/2026:

| Hạng mục | Kết quả |
| --- | --- |
| Môi trường G1 `.venv-g1` | Python 3.12.14; `.venv` 3.13.9 cũ vẫn giữ nguyên |
| Thư viện | Cài đúng `requirements.lock.txt`, cài package editable thành công |
| `pip check` | `No broken requirements found.` |
| Bộ kiểm tra hiện có | 19/19 đạt: 14 hạ tầng gốc và 5 kiểm tra nghiệm thu G1 |
| Config `features_nhu_smoke.yaml` | Đọc bằng `read_config` thành công; xác minh S2, 60/5 giây và ngưỡng 0,90 |
| Dữ liệu S2 thật | Đã có; chỉ chạy extractor thống kê raw để kiểm tra G1 |
| Bộ lọc và đồng bộ đa tần số | Chỉ ở mức thiết kế G1, chưa có triển khai trong phần bàn giao của Như trên main |

Giữ nguyên source hạ tầng gốc, phiên bản khóa thư viện và kết quả của Thái. Các phiên chạy dùng dữ liệu giả của bộ test không phải kết quả thực nghiệm WESAD. Kết quả mẫu S2 trên máy Như được ghi riêng trong [biên bản G1](G1_NHU_KIEM_TRA_S2_2026-11-03.md).

- Có môi trường chạy và kết quả bộ kiểm tra hiện có.
- Mai đã chọn thiết kế đồng bộ/bộ lọc ban đầu tại `b5a0cb8`; phần biên/QA bổ sung của Như có trong config thiết kế để đối chiếu tiếp.
- Có raw S2 và bằng chứng đối chiếu mẫu giữa các máy.

Phần thiết kế, môi trường và chạy mẫu G1 trên máy Như đã có đầu ra. Mẫu S2 trên một máy không thay thế nghiệm thu bốn máy. Chưa đánh dấu nghiệm thu chung khi chưa nhận gói từ các máy khác. File này không gửi thông điệp hay ghi nhận review thay nhóm.

## Nguồn đối chiếu

- `PHAN_CONG_MAY_HOC.docx`, các mục G1/G2 và hợp đồng dữ liệu, file người dùng cung cấp.
- `Nhom09_NhanDienStress_WESAD_Proposal_v2 (3).docx`, file người dùng có trên máy: tần số kênh và phạm vi tiền xử lý.
- Schmidt và cộng sự, *Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection*, mục 4.1, trang PDF 4, file PDF người dùng cung cấp: EDA ngực, RESP và các cách tính feature. Bước trượt chính của nhóm là 5 giây theo proposal/phân công, không lấy bước 0,25 giây của bài báo.
- Code hiện tại: `src/wesad/preprocessing.py`, `src/wesad/extraction.py`, `tests/test_infrastructure.py`.
