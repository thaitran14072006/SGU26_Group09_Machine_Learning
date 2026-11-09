# G4 phần Như: tuning RF và MLP

Ngày bắt đầu: 09/11/2026. Người dùng xác nhận Như làm cả RF và MLP, phụ trách outer test **S4, S9, S11, S17**, cho ba lớp và nhị phân. Phạm vi bản này là tuning cơ bản trên bộ kết hợp sinh lý68 đã nghiệm thu. Không sửa kết quả mặc định G3.

## Thiết kế đã khóa trước khi chạy

- Cùng CSV 6.189 cửa sổ, 15 người, 68 feature; SHA-256 `2f2fddd678f333a800f2869d2e78b407934b2c3492190ed70e28d8a9107fd8ba`.
- Outer LOSO: mỗi lượt vẫn dùng đủ 14 người train, một người test; chỉ nhận bốn outer test được giao. Không thu nhỏ dữ liệu train còn bốn người. Inner GroupKFold 3 lượt chỉ trên 14 người train; chọn theo Macro-F1. Nhãn ba lớp `[1,2,3]`, nhị phân `[0,1]`, `zero_division=0`.
- Bốn tổ hợp cố định cho mỗi mô hình tại `configs/g4_nhu_tuning.yaml`, lấy từ lưới triển khai đã có trong dự án và miền tham số ở phân công. Đây là tìm kiếm giới hạn, không phải thử hết tích Descartes.
- RF: thay số cây 100/200, độ sâu 10/None, `max_features` sqrt/0.5. MLP: thay hidden sizes (32,)/(64,32), alpha 1e-4/1e-3, learning rate 1e-3/1e-2. MLP giữ `max_iter=200`; không tự tăng sau khi xem điểm test.
- Median imputer nằm trong Pipeline; MLP thêm StandardScaler, RF không cần scaler. Toàn Pipeline được fit lại trong từng inner train rồi refit trên outer train. Giữ NaN và không dùng metadata làm X.
- Seed 42, chạy nối tiếp, không selection hoặc cân bằng lớp trong lượt tuning này. Không chọn lại grid theo outer test.
- Môi trường lượt chính: `.venv-g1`, Python 3.12.14 và thư viện theo `requirements.lock.txt`. Lượt quick ban đầu dùng `.venv` Python 3.13.9 được giữ riêng, không trộn vào main.

## Chạy và tiếp tục

Chạy trong thư mục `wesad-stress-detection`. Cần giải nén gói G2 đúng đường dẫn trong config. Không chạy đồng thời hai tiến trình vào cùng output.

```powershell
# Quick một outer test; vẫn có đủ 15 người đầu vào
.\.venv-g1\Scripts\python.exe scripts/run_nhu_g4.py --test-subject S2 --output outputs/g4_nhu_rf_mlp_quick312 --resume

# Lượt theo phân công: 4 subject x 2 mô hình x 2 bài toán = 16 outer folds
.\.venv-g1\Scripts\python.exe scripts/run_nhu_g4.py --resume

# Kiểm tra độc lập sau khi hoàn thành
.\.venv-g1\Scripts\python.exe scripts/run_nhu_g4.py --verify-only

# Vẽ loss từ JSON đã lưu, trong môi trường có matplotlib (không fit lại)
python scripts/plot_nhu_g4.py

# So sánh từng người với G3 và tạo ZIP bàn giao, sau khi vẽ hình
.\.venv-g1\Scripts\python.exe scripts/report_nhu_g4.py
```

`--resume` kiểm tra signature và checksum từng checkpoint trước khi bỏ qua fold. Đổi dữ liệu, nguồn chạy hoặc cấu hình cần output mới; không dùng lại thư mục cũ để ghi đè một thử nghiệm khác. Lô này có 16 outer folds, 192 inner fits và 16 refits. Core runner vẫn ghi `status: partial` vì nhóm chưa đủ 15 người; verifier riêng ghi `scope: assigned_outer4` khi hoàn thành đúng phần được giao.

## Đầu ra

- `outputs/g4_nhu_rf_mlp_outer4/`: config, môi trường, log, source snapshot, dự đoán, splits, best_params, điểm theo người, summary và verification.
- `folds/<task>/<model>/<subject>/inner_cv.csv`: điểm của bốn ứng viên; `details.json`: splits, best_params, confusion matrix và cảnh báo, kể cả chưa hội tụ.
- `details.json/mlp_optimization` lưu đường train loss, số epoch, loss cuối và có chạm max_iter hay không của ứng viên tốt nhất khi refit outer train. Không gọi đây là validation loss. Cảnh báo được bắt từ toàn bộ inner search và refit, chưa gán riêng từng candidate; không xem việc không chạm max_iter là bằng chứng tổng quát hóa tốt.
- Sau lệnh báo cáo: `reports/g4_nhu/` chứa bảng G3–G4 của cùng bốn người, điểm ghép, tham số, `mlp_convergence.csv`, hình `mlp_refit_loss.png` và provenance; ZIP ngoài Git là `outputs/g4_nhu_rf_mlp_outer4_handoff.zip`.
- Chỉ coi phần tuning được giao hoàn tất khi đủ 16 folds và `verification.json` đạt. Bảng mean/SD bốn người không thay bảng 15 người; còn cần Thái/Mai ghép với các lô khác.
- Lượt chạy rộng trước khi xác nhận lô đã được dừng ở `outputs/g4_nhu_rf_mlp_main/`. Không dùng các checkpoint đó vào bàn giao lô bốn người; giữ để truy vết.

## Phần tiếp theo của G4

Sau lượt tuning, còn phối hợp Nam/Mai làm so sánh RF cố định giữa ngực/cổ tay/kết hợp, đóng góp nhóm feature và cân bằng lớp theo phân công. Thiết kế các nhánh này phải chốt trước khi chạy; không lấy tham số thắng outer test làm cấu hình cố định. Chuẩn hóa theo subject do Nam/Thái chủ trì. Bản này chưa thực hiện các nhánh đó hoặc kiểm định G5, không chốt G4 cả nhóm hoàn thành.

Giữ kết quả tuning thấp hơn G3 nếu có; không sửa grid để ép điểm tăng. Các kết luận hiện tại chỉ mô tả, chưa có kiểm định ý nghĩa thống kê. Người dùng tự commit/push và tự gửi ZIP cho Thái/Mai.

## Kết quả lượt 09/11/2026

Đã hoàn thành **16/16 outer folds**, **6.632 dự đoán ngoài mẫu**, verifier đạt. Đã chạy lại `--resume`, toàn bộ 16 checkpoint đều được kiểm tra và bỏ qua, không fit lại. Bảng điểm tại [kết quả lô bốn người](../reports/g4_nhu/KET_QUA.md). Lô Như hoàn thành phần tuning được giao, còn cần nhóm ghép các outer subject khác.

Hình loss chỉ đọc JSON có sẵn, được vẽ bằng Python hệ thống `C:\Anaconda3\python.exe` có matplotlib 3.10.6, giống cách tách môi trường vẽ ở G3. Môi trường train `.venv-g1` không cài thêm thư viện. `plot_environment.json` ghi môi trường vẽ và checksum các JSON nguồn.

MLP không phát sinh cảnh báo hội tụ trong lượt này. Tám lượt refit dừng sau 15–65 epoch, không lượt nào chạm max_iter=200; đường train loss giảm về mức nhỏ. Đây là quan sát trên train, không kết luận mô hình không overfit hoặc đã tổng quát hóa tốt.

Kiểm tra: 4 unit tests G4 và test nested/checkpoint/resume hiện có đều đạt. Kết quả thật cũng đã được kiểm tra splits, checksum, lưới và best_params, loss trace, khóa/nhãn dự đoán, metric và summary tính lại.
