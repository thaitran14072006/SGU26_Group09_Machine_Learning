# G4 RF/MLP: mặc định so với tuning

Chỉ lô S4/S9/S11/S17, hai bài toán. Mean ± SD mẫu của 4 người; Δ = G4 − G3. Không thay bảng 15 người của nhóm.

| Bài toán | Mô hình | Macro-F1 G3 | Macro-F1 G4 | Δ | BA G3 | BA G4 |
|---|---|---|---|---|---|---|
| binary | MLP | 0.8540 ± 0.1597 | 0.8594 ± 0.1327 | +0.0054 | 0.8602 ± 0.1645 | 0.8572 ± 0.1475 |
| binary | RF | 0.9382 ± 0.0637 | 0.9077 ± 0.1026 | -0.0305 | 0.9264 ± 0.0819 | 0.8965 ± 0.1167 |
| multiclass | MLP | 0.6254 ± 0.0809 | 0.6744 ± 0.1582 | +0.0489 | 0.6452 ± 0.0950 | 0.6791 ± 0.1503 |
| multiclass | RF | 0.6550 ± 0.1324 | 0.6502 ± 0.1390 | -0.0048 | 0.6674 ± 0.1133 | 0.6667 ± 0.1183 |

Tuning có thể thấp hơn mặc định. Không chọn lại lưới sau khi xem outer test. Chưa kiểm định G5.
Cảnh báo hội tụ nằm trong details.json từng fold; số fold có cảnh báo: 0.

MLP refit: 8 lượt, dừng sau 15–65 epoch; 0 lượt chạm max_iter=200. Loss train giảm về mức nhỏ trong hình đã lưu. Đây là kiểm tra tối ưu trên train, không phải validation loss hoặc bằng chứng không overfit.
