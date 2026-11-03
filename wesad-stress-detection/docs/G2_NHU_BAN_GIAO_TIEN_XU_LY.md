# Bàn giao tiền xử lý G2 của Như

Ngày 03/11/2026. Đối chiếu bản nháp G1 của Mai tại commit `e12c577` và schema `g1_mai_draft_v1` gồm 8 metadata, 68 feature dự kiến. Phần Như triển khai ở đây là lọc, đồng bộ cửa sổ và các phép tính ACC/EMG để Mai gọi trong extractor. Chưa có raw S2 trên máy; chưa nghiệm thu tín hiệu, đơn vị hoặc bộ đặc trưng cuối.

## Phần đã thực hiện

| File | Giao diện bàn giao |
| --- | --- |
| `src/wesad/preprocessing.py` | `filter_signal`, `prepare_signals`, `five_second_features`; giữ hàm `iter_windows` cũ |
| `src/wesad/windowing.py` | `SampledSignal`, `AlignedWindow`, `WindowAudit`, `iter_multirate_windows` |
| `configs/preprocessing_nhu_draft.yaml` | Tần số, shape, bộ lọc và vùng bỏ biên cho lượt chẩn đoán |
| `scripts/check_preprocessing.py` | Đọc từng PKL, lọc, kiểm tra cửa sổ và ghi metadata/báo cáo chất lượng |
| `tests/test_preprocessing.py` | 14 test bổ sung, gồm gọi CLI thật trên PKL giả |

Schema và danh mục feature của Mai giữ nguyên trạng thái nháp. Chưa tạo `features.py`, chưa gán giá trị giả cho các feature sinh lý còn thiếu.

## Quyết định triển khai của Như

- Giữ tần số gốc: ngực 700 Hz, BVP cổ tay 64 Hz, ACC cổ tay 32 Hz, EDA/TEMP cổ tay 4 Hz. Các kênh phải có chung mốc bắt đầu từ PKL đồng bộ; thời lượng chênh lệch được báo cáo, không tự sửa offset.
- Nhãn dùng trục 700 Hz, cửa sổ 60 giây, bước 5 giây, đa số ít nhất 90% trên toàn cửa sổ gốc. Nhãn 0/4/khác vẫn tính trong mẫu số, không thành lớp đích.
- Dùng khoảng `[start_s, end_s)` cho mỗi kênh. Không nội suy, không nối đoạn và không cắt ngắn tín hiệu cho vừa cửa sổ.
- Chọn chung cửa sổ đầy đủ, hữu hạn ở tất cả kênh được yêu cầu trong config. Thiếu một kênh/mẫu thì không giữ cửa sổ đó. Các tập cảm biến cần dùng chính danh sách khóa này khi so sánh; không chạy lại QA riêng từng nhóm để đổi mẫu đánh giá.
- Lọc Butterworth dạng SOS, `sosfiltfilt`, axis thời gian = 0, odd padding. Bậc thiết kế 4; band-pass có bậc một chiều 8, lọc hai chiều tiếp tục làm thay đổi đáp ứng biên độ. Đây là lựa chọn triển khai của Như, không khẳng định đúng bậc lọc bài báo.
- Lọc riêng từng đoạn hữu hạn liên tục. Không lọc xuyên NaN/Inf. Đoạn không đủ padding hoặc vùng nội thất sau khi bỏ biên trở thành NaN, không thay bằng 0.
- Lọc trước cửa sổ trong mỗi người, không nối giữa người. Đây là phân tích offline dùng mẫu tương lai trong cùng bản ghi, không mô tả như xử lý realtime. Không có thao tác học imputer/scaler trên toàn bộ dữ liệu.
- ACC giữ trọng lực, không tự trừ mean hoặc chuyển scale. Tính magnitude `sqrt(x*x+y*y+z*z)`, không dùng tổng trị tuyệt đối các trục làm magnitude.
- ACC/EMG có đúng 12 đoạn 5 giây không chồng lấp trong 60 giây. Tính feature mỗi đoạn rồi lấy mean 12 giá trị; không bỏ một đoạn lỗi rồi lấy mean các đoạn còn lại.
- EMG dùng RMS, MAV, waveform length/s và tích phân tuyệt đối theo danh mục Mai. Tên helper khớp schema sau khi thêm prefix thiết bị. EMG hiện chưa lọc high-pass; cần khóa cùng Mai trước lượt chính.

## Thông số kiểm tra đang để nháp

| Kênh | Bộ lọc | Bỏ biên mỗi đầu đoạn hữu hạn | Trạng thái |
| --- | --- | ---: | --- |
| EDA ngực | Low-pass 5 Hz | 10 giây | Cutoff theo bài báo; vùng biên là đề xuất |
| Resp ngực | Band-pass 0,1–0,35 Hz | 30 giây | Cutoff theo bài báo; vùng biên là đề xuất |
| EDA cổ tay | Low-pass 1 Hz | 10 giây | Cutoff và vùng biên là đề xuất của Như |
| ECG/BVP/EMG/ACC/TEMP | Chưa áp dụng bộ lọc mới | 0 | Cần kiểm tra cùng detector/feature của Mai |

Các vùng biên trên chưa chứng minh loại hết quá độ và có thể làm giảm số cửa sổ đáng kể. Cần kiểm tra đáp ứng bộ lọc, tín hiệu thật và số cửa sổ bị loại trước khi khóa phiên bản. Với EDA cổ tay 4 Hz, cutoff phải nhỏ hơn Nyquist 2 Hz; hàm sẽ từ chối cutoff 5 Hz.

## Mai tích hợp như thế nào

```python
from wesad.preprocessing import prepare_signals, five_second_features
from wesad.windowing import WindowAudit, iter_multirate_windows

# data là một PKL đã đọc bởi load_subject; config là YAML tiền xử lý.
signals = prepare_signals(data, config)
audit = WindowAudit()
for window in iter_multirate_windows(signals, data["label"],
                                    audit=audit, **config["window"]):
    row = window.metadata(subject_id)
    # row có đủ 8 metadata, khóa <subject>_<start_index_700hz>.
    # Ví dụ helper; ĐƠN VỊ VẪN LÀ ĐƠN VỊ ĐẦU VÀO, phải xác minh trước
    # khi xuất g/mV theo schema. Không tự giả định hệ số từ giá trị tín hiệu.
    acc = window.signals["chest_acc"]
    features = five_second_features(acc.values, acc.sampling_rate_hz, sensor="acc")
    row.update({"chest_" + key: value for key, value in features.items()})
    # Mai bổ sung các feature khác trong features.process_subject.
    # Không xuất row đang thiếu 56 feature này như một bảng hoàn chỉnh.
```

`quality_flag` hiện là `finite_alignment_only_not_physiological_QA`; không ghi `ok` vì chưa có QA sinh lý. Helper trả feature cùng đơn vị đầu vào: mean/std/RMS/MAV theo đơn vị tín hiệu, tích phân theo đơn vị tín hiệu × giây, waveform length theo đơn vị tín hiệu/giây. Không có bước chuyển đổi sang g/mV trong code này.

Việc kiểm tra hữu hạn có tính bảo thủ: NaN/Inf hoặc vùng lọc bị đánh dấu ở bất cứ kênh yêu cầu nào đều loại toàn bộ cửa sổ. Những lỗi sinh lý khác như nhịp không tin cậy sau này vẫn có thể tạo NaN feature có lý do theo hợp đồng Mai. Báo cáo flat chỉ là dấu hiệu toàn kênh hằng, chưa phải detector nhiễu sinh lý hay chính sách loại đã nghiệm thu.

## Chạy ngay khi có raw

Tạo `configs/local.yaml` chỉ chứa đường dẫn `data_root` thật. Chạy tại thư mục `wesad-stress-detection`:

```powershell
.\.venv\Scripts\python.exe scripts/check_preprocessing.py --config configs/preprocessing_nhu_draft.yaml --local configs/local.yaml --subjects S2 --output outputs/preprocessing/nhu_S2_draft_01
```

Mỗi lượt dùng thư mục output mới để không ghi đè bằng chứng. Gói đầu ra gồm config không chứa đường dẫn raw riêng, phiên bản môi trường/source, checksum raw, metadata cửa sổ, chất lượng từng kênh, lý do loại và thời gian. Không lưu lại toàn bộ tín hiệu hay tạo bảng 68 feature. Run complete chỉ có nghĩa công cụ chẩn đoán đã chạy xong trên danh sách được chọn; `ready_for_training` luôn false.

Sau khi S2 được kiểm tra và nhóm chốt tham số, đổi phiên bản config rồi mới chạy lô Như `--subjects S4 S9 S11 S17`. Không dùng kết quả chẩn đoán này làm dữ liệu huấn luyện chính.

## Kết quả kiểm tra và phần còn mở

Ngày 03/11/2026: chạy `python -m unittest discover -s tests -v` trong `.venv`, **28/28 test đạt** (14 cũ + 14 mới). Dữ liệu kiểm tra là dữ liệu giả. Kiểm tra lọc có đo sai số so với thành phần tần số đã biết; kiểm tra ACC/EMG dùng nghiệm số tính được trước; kiểm tra CLI đọc PKL giả và xác nhận không xuất kết quả training.

Các điểm cần xử lý khi có S2 và Mai đối chiếu:

1. Xác minh mốc đồng bộ, shape thực tế, scale và đơn vị từng kênh; chưa tự chia ACC cổ tay cho 64.
2. Vẽ tín hiệu trước/sau lọc; đo thời gian, RAM đỉnh và ảnh hưởng vùng biên. Chưa có số liệu RAM thật.
3. Mai/Như khóa detector ECG/BVP, xử lý nhịp lỗi và số khoảng tối thiểu; decomposition EDA/SCR, onset/ngưỡng và chính sách không có SCR.
4. Khóa PSD và ba dải ECG/BVP; không đặt phổ tín hiệu raw thành LF/HF-HRV.
5. Chốt lọc EMG, QA sinh lý và dung sai so sánh từng nhóm feature. Mai cập nhật schema sau khi thống nhất, chưa ghi nhận Mai đã đồng ý.
6. Tích hợp `features.process_subject`, trích S2 và đối chiếu bốn máy, sau đó mới chạy đủ 15 người. RF/MLP thuộc G3, chưa dùng bảng feature nháp để báo điểm.

Source mới thay đổi mã băm hạ tầng: mọi lượt extractor/experiment bằng source mới cần thư mục output mới; không tái sử dụng thư mục có signature source cũ.

## Tài liệu kỹ thuật đối chiếu

- Hợp đồng Mai: [G1_HOP_DONG_FEATURE.md](G1_HOP_DONG_FEATURE.md), [schema nháp](../configs/feature_schema.json).
- Bài báo WESAD người dùng cung cấp, mục 4.1 trang PDF 4: cutoff EDA/Resp. Thiết kế bản nháp riêng Như: [G1_NHU_THIET_KE_TIEN_XU_LY.md](G1_NHU_THIET_KE_TIEN_XU_LY.md).
- SciPy [butter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html) và [sosfiltfilt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html): thiết kế SOS, bậc band-pass, trục lọc và padding.
