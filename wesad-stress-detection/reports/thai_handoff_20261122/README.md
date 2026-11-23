# Bàn giao phần thực nghiệm của Thái — 22/11/2026

Phạm vi phụ trách: EXP-001 Dummy, EXP-002 LR baseline, EXP-011 LR balancing,
EXP-016 LR feature selection và bổ sung EXP-008 SVM tuning theo yêu cầu ngày 22/11.
LR Regularization là phụ lục, không đổi mã hay số lượng 20 hướng của nhóm.
Các mốc hoàn tất dùng thời gian thực trong acceptance/metadata và Git;
không ghi hoàn thành trước hạn 10:00 của kế hoạch ban đầu.

## Kết quả và nguồn minh chứng

| Hạng mục | Hồ sơ |
|---|---|
| 3 selection + 3 balancing + 5 tuning LR | [Báo cáo LR](../lr_workflow_20261121/README.md) |
| 6 cấu hình SVM EXP-008 bổ sung | [Báo cáo SVM](../exp008_thai_20261122/README.md) |
| Pipeline LR binary và multiclass | [Bundle model](../../artifacts/models/lr_candidate_20261122/metadata.json) |
| Cách chọn cấu hình khi fit toàn bộ dữ liệu | [Bằng chứng inner CV](../../artifacts/models/lr_candidate_20261122/selection_evidence.json) |
| Schema đầu vào đủ 68 cột | [Feature schema](../../artifacts/models/lr_candidate_20261122/feature_schema.json) |
| CSV mẫu 30 cửa sổ | [Đầu vào](../../app/examples/features.csv) |
| Dự đoán mẫu | [Binary](../../app/examples/expected_binary.csv), [Multiclass](../../app/examples/expected_multiclass.csv) |
| Nhật ký duy nhất của nhóm | [Excel](../../experiments/experiments_logs.xlsx) |

Ảnh cho báo cáo/thuyết trình lấy từ `lr_workflow_20261121/` và
`exp008_thai_20261122/`: comparison/workflow, confusion matrix từng task.
Mỗi ảnh có CSV nguồn cạnh file; không dùng ảnh hay điểm từ mô hình khác để
minh chứng LR/SVM. Các hồ sơ khác của nhóm giữ nguyên.

## Thiết kế và cách đọc kết quả

Đầu vào là physiological68: 6.189 cửa sổ của 15 subject. Multiclass dùng
1=Baseline, 2=Stress, 3=Amusement; binary dùng 0=Non-stress, 1=Stress.
LOSO giữ riêng một subject để test. Inner GroupKFold(3) chia 14 subject train
để chọn cấu hình. Imputer/scaler/selector/resampling hoặc class weights chỉ
học trong train của mỗi lần fit. Không dùng outer score để chọn cấu hình.

Mỗi run đánh giá hai task: 30 outer fold và 12.378 dự đoán OOF.
Macro-F1 và Balanced Accuracy được tổng hợp mean ± SD mẫu qua 15 subject;
các cửa sổ chồng lấp không phải các quan sát độc lập cho kiểm định thống kê.
Confusion matrix gộp cửa sổ dùng để tìm kiểu lỗi, không thay thế đánh giá
theo subject. Không khẳng định khác biệt có ý nghĩa thống kê từ mean đơn thuần.

LR đa lớp sau quy trình đạt Macro-F1 0.6287 ± 0.1908, BA 0.6801 ± 0.1537.
LR binary đạt Macro-F1 0.8767 ± 0.1591, BA 0.8872 ± 0.1241. Binary sau tuning
thấp hơn gate balancing (F1 0.8931); báo cáo giữ nguyên mức giảm này.
Lớp Amusement khó, S13 có lỗi lớn; xem đủ cặp subject trong các CSV báo cáo.

Pipeline triển khai fit trên toàn bộ 15 người sau khi thực hiện lại quy trình
lựa chọn tuần tự bằng inner GroupKFold(3) trên dữ liệu triển khai. Cả hai task
chọn ANOVA k=16, C=1; multiclass dùng balanced, binary dùng sqrt_balanced.
Đây là LR ứng viên, chưa tuyên bố mô hình thắng cả nhóm. Điểm inner full-data
là tiêu chí lựa chọn, không phải test score mới. Dữ liệu demo là cửa sổ đã
dùng huấn luyện, không thể dùng làm bằng chứng tổng quát hóa.

## Chạy demo CSV

Tại thư mục `wesad-stress-detection`, dùng môi trường theo requirements lock:

```powershell
.\.venv\Scripts\python.exe app/server.py
```

Mở `http://127.0.0.1:8080`, tải CSV mẫu trên giao diện, chọn bài toán,
upload file và bấm **Dự đoán**. Giao diện có bảng nhãn, biểu đồ số lượng nhãn
và nút tải CSV kết quả. Dừng server bằng Ctrl+C.
CSV phải có đủ 68 tên cột; thứ tự cột có thể thay đổi vì pipeline sắp lại theo
schema. Cho phép metadata bổ sung nhưng chỉ dùng 68 đặc trưng để dự đoán.
Giá trị thiếu một phần được median-impute; thiếu tên cột, trùng tên cột,
file rỗng, giá trị phi số/Inf hoặc một hàng thiếu toàn bộ bị từ chối.

Chạy suy luận không cần giao diện:

```powershell
.\.venv\Scripts\python.exe -m wesad.inference app/examples/features.csv --task binary --output outputs/demo_binary.csv
.\.venv\Scripts\python.exe -m wesad.inference app/examples/features.csv --task multiclass --output outputs/demo_multiclass.csv
```

Model/schema đã commit nên demo không cần raw WESAD hay train lại. Nếu cần
tái tạo model từ đầu, dùng checkout sạch với đủ bảng feature và chạy
`scripts/export_lr_candidate.py`; script từ chối ghi đè bundle đã xuất.
Model joblib cần cùng môi trường đã ghi trong metadata; không nạp model lạ.

## Kiểm tra bàn giao

[Biên bản nghiệm thu](validation.json): 6 run SVM, 180 fold, 74.268 dự đoán
OOF, đủ 12 dòng nhật ký SVM; 11 kiểm thử LR/inference đều đạt. Model và
hai CSV kết quả CLI khớp checksum/kết quả mẫu đã lưu.

```powershell
.\.venv\Scripts\python.exe scripts/continue_lr_experiments.py --status
.\.venv\Scripts\python.exe scripts/report_exp008_thai.py
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_lr_workflow.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_inference.py -v
```

`export_lr_candidate.py` đã kiểm tra save/load cho dự đoán và xác suất trên
toàn bộ 6.189 hàng của mỗi task. Các kiểm thử suy luận so kết quả CLI/core
với CSV mẫu, kiểm tra dữ liệu lỗi và API upload hai task. Kiểm thử HTTP không
thay thế kiểm tra thao tác trình duyệt thủ công.

## Câu hỏi vấn đáp

1. **Vì sao chia theo người?** Để đo khả năng dự đoán trên người chưa từng
   xuất hiện trong train, tránh cửa sổ cùng subject lẫn train/test.
2. **Vì sao cần inner CV?** Để chọn tham số mà không nhìn outer test.
3. **Một run có bao nhiêu fold?** 15 fold mỗi task, tổng 30 fold cho hai task.
4. **Vì sao 12.378 dự đoán?** 6.189 cửa sổ được dự đoán OOF một lần mỗi task.
5. **Macro-F1 khác accuracy thế nào?** Tính F1 từng lớp rồi trung bình đều,
   nhấn mạnh cả lớp ít mẫu; accuracy tính tỷ lệ cửa sổ đúng tổng thể.
6. **Vì sao tuning có thể giảm điểm?** Inner/outer khác subject; chọn theo
   inner CV không bảo đảm tăng điểm trên mọi outer subject.
7. **C và gamma của SVM là gì?** C điều chỉnh mức phạt sai lệch; gamma điều
   chỉnh phạm vi ảnh hưởng của điểm trong kernel RBF.
8. **Tại sao model triển khai khác cấu hình từng LOSO fold?** Model triển khai
   được chọn lại bằng CV trên toàn bộ dữ liệu sẵn có rồi fit để suy luận;
   LOSO đánh giá quy trình, không cung cấp một bộ tham số cố định duy nhất.
9. **CSV mẫu có phải test mới không?** Không. Chỉ kiểm tra luồng suy luận và
   tính nhất quán; chất lượng ngoài mẫu lấy từ LOSO.
10. **Đã hoàn tất nhóm chưa?** Hồ sơ này bàn giao phần Thái. Việc chọn model
    cuối và ghép kết quả KNN/SVM/RF/MLP cần đối chiếu hồ sơ cả nhóm; không
    dùng hoàn tất LR để xác nhận hoàn tất phần người khác.

Benchmark chỉ có 15 người trong phòng thí nghiệm, đã được sử dụng nhiều lần,
chưa có external holdout. Kết luận nghiên cứu còn tính thăm dò.
