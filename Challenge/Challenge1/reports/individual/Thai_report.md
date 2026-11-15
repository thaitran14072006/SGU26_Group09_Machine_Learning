# Thực nghiệm của Thái — House Prices

## Phương pháp

Ridge và ElasticNet. Chia development/holdout 80/20, seed=7; 5-fold CV trên development.
Điền thiếu, one-hot và chuẩn hóa fit riêng trong từng fold bằng pipeline.
Nhãn huấn luyện là log(SalePrice), chuyển dự đoán về giá bằng exp().
Id chỉ dùng để đối chiếu và xuất dự đoán. Không loại ngoại lệ dựa trên holdout.

## Kết quả

- Cấu hình chọn: Thai_EXP-010 — ElasticNet cơ sở.
- CV RMSE log: 0.12900 ± 0.03147 (ĐLC mẫu của 5 fold).
- Holdout RMSE log: 0.11359.
- Holdout MAE giá: 15,072.
- Dự đoán cuối: results/submissions/submission_Thai.csv, 1459 dòng.

## Phân tích sai số

5 nhà có sai số tuyệt đối lớn nhất trên holdout:

```
  Id  actual     predicted  residual_log  absolute_error
 692  755000 591247.139371      0.244484   163752.860629
 186  475000 331032.759257      0.361097   143967.240743
1325  147000 289840.039194     -0.678897   142840.039194
 826  385000 280278.286638      0.317460   104721.713362
 474  440000 353569.797985      0.218694    86430.202015
```

Biểu đồ: members/Thai/outputs/figures/cv_and_residuals.png.
Không chỉnh cấu hình theo holdout. Một lần chia chưa đủ chứng minh ưu thế thống kê.
Các biến log chọn trước, chưa tối ưu ngoại lệ. NA của garage cần xem thêm năm xây dựng và
thuộc tính liên quan; phiên bản hiện tại dùng quy tắc đơn giản và giữ cảnh báo trong JSON.

## Ngày và minh chứng

Ngày 01–13/11/2026 là mốc nhật ký người dùng quy định. Thời điểm chạy máy lưu riêng trong JSON.
Nhật ký: reports/individual/nhat_ky_thuc_nghiem.csv và .xlsx.
Mỗi dòng có nguồn notebook, JSON kết quả, checksum, điểm từng fold và thời điểm chạy máy.
Kết quả CV là số đo chạy thực tế. test.csv không có nhãn, chưa có điểm Kaggle.
