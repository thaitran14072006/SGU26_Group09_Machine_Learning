# exp010 — tuning mlp

Như, chạy mới ngày 15/11/2026. Phần đồ án WESAD, không liên quan challenge.

- Notebook nhật ký: `notebooks/experiments/EXP010_MLP_tuning.ipynb` ở gốc project.
- Notebook trong thư mục này là bản tổng hợp đã lưu output.
- `results/`: 30 outer folds, inner scores, best_params, dự đoán, loss và 4 hình.
- `results/config.yaml`: cấu hình thực tế; `source_snapshot.zip`: nguồn lúc train.
- `results/conclusion.md`: nhận xét từ kết quả thực tế, kể cả điểm giảm.

Dữ liệu physiological68 giữ ngoài Git theo đường dẫn trong config; checksum
`2f2fddd678f333a800f2869d2e78b407934b2c3492190ed70e28d8a9107fd8ba`.
Không đưa Id/metadata vào X. Baseline so sánh là exp006 đã chạy trên cùng 15 người.

## xem và chạy lại

Mở notebook với môi trường có pandas, numpy, matplotlib. Các ô mặc định chỉ đọc
kết quả đã lưu. Nếu máy không có output gốc, notebook tự đọc `results/` tại đây.
Không cần train lại chỉ để xem bảng và hình.

Từ thư mục gốc `wesad-stress-detection`, dùng Python theo môi trường train:

```powershell
# Lượt mới, tạo output riêng; 30 outer folds và 390 lần fit
& '.\.venv-g1\Scripts\python.exe' -u scripts/exp010_train.py

# Kiểm tra gói đã lưu, không fit
& '.\.venv-g1\Scripts\python.exe' scripts/exp010_train.py --verify-only experiments/02_tuning/EXP-010/results
```

Nếu một lượt mới bị ngắt, dùng `--resume-run outputs/<tên lượt đó>` để kiểm tra
checkpoint rồi tiếp tục. Giữ nguyên data/config/source/environment để resume.
Không dùng `--resume-run` trỏ tới bản tổng hợp này: config vẫn ghi output gốc.

Chỉ chọn cấu hình bằng inner CV trong 14 người train, không chỉnh theo outer test.
Đường loss là train loss khi refit ứng viên thắng, không phải validation loss.
Baseline và tuning cùng Python 3.12.14/sklearn 1.9.1, nhưng pandas khác phiên bản;
chưa đo độ nhạy với khác biệt đó. Chưa thực hiện kiểm định thống kê.

Best params được giữ theo từng task/subject để làm đối chứng cho exp015.
Exp020 cần lưới khác; không sửa kết quả của exp010 để ép điểm tăng.
