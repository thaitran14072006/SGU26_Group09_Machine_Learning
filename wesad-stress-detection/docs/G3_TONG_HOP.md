# Tổng hợp dự đoán G3 ngày 08/11/2026

**Đã hoàn thành phần tổng hợp G3 của Mai:** Dummy/LR của Mai, KNN/SVM của Nam, RF/MLP của Như. Đủ sáu mô hình mặc định, hai bài toán, 15 subject: **180 fold LOSO, 74.268 dự đoán ngoài mẫu** trên cùng 6.189 cửa sổ sinh lý68. Không huấn luyện/tuning thêm trong lượt hợp nhất.

Kết luận hoàn thành là về kiểm tra/hợp nhất kết quả. **Không tuyên bố mọi producer dùng cùng một bản mã nguồn runtime:** Nam dùng runner cũ/factory riêng đã rà từ source cung cấp, `git_dirty=true`, không có snapshot runtime khớp hash kết quả. Giữ giới hạn này và nguồn producer thật; không sửa chữ ký/config để ép qua merger nghiêm ngặt.

## Gói nguồn và kiểm tra

| Nguồn | Nội dung | Kết luận |
| --- | --- | --- |
| `g3_baselines_physiological68_v1.zip` | Dummy/LR, 60 fold, 24.756 dự đoán | Đã kiểm tra và hợp nhất |
| `g3_nhu_rf_mlp_phys68_main_20261108.zip` | RF/MLP, 60 fold, 24.756 dự đoán | Đã kiểm tra; 261 checksum bàn giao khớp |
| `g3_knn_svm_physiological68_defaults.rar` | KNN/SVM, 60 fold, 24.756 dự đoán | OOF đạt kiểm tra, source cung cấp đã review; giữ giới hạn runtime |
| `WESAD_G3_KNN_SVM_68.zip` | Chỉ code/config | Bằng chứng review nguồn, không dùng thay dự đoán |

- Cùng checksum bảng G2 `2f2fddd678f333a800f2869d2e78b407934b2c3492190ed70e28d8a9107fd8ba`, đúng 68 feature, không metadata trong X. Cùng Python patch 3.12.14 và sáu phiên bản thư viện ghi nhận; khác build được lưu provenance.
- Mỗi model/task đủ 15 subject và đúng 6.189 khóa. Không trùng khóa model/task/sensor/normalization/subject/window, không sai nhãn tham chiếu hoặc lẫn quick.
- Kiểm tra chữ ký run/checksum fold, đúng 14 train/1 test, không giao subject. Tính lại Macro-F1, Balanced Accuracy, confusion matrix và mean/std summary từ dự đoán.
- Dummy/LR/RF/MLP có source hash chuẩn hóa LF `744ea523aa54da3ee680f33717ee3bd24adcdd053522a35f1050b845056e25a9` và snapshot runtime đã kiểm tra.
- Source KNN/SVM cung cấp có AST phần preflight, fit/predict, nhãn và metric khớp runner chung; khác biệt runner ở metadata nguồn. Factory được review và khóa hash `2794d76159e9cd051d935b45f18c7ff7e3468ed1a1cc0a53d3ed800a354a00f1`: median imputer `keep_empty_features=True`, StandardScaler, KNeighborsClassifier/SVC mặc định. Không override tham số, tuning, balancing hay selection. Không thực thi code nhận được.
- Producer Nam ghi nguồn `ef0d828`, dirty, code hash `8ebf5a0954fcd46cd026dec082303ba5f38ef9d2d5174faec2f94794b7b8cc25`. Review source bundle không chứng minh toàn bộ source runtime dirty giống bundle; `strict_same_runtime_source_acceptance=false` được giữ trong summary. Nếu cần tái lập nghiêm ngặt cùng nguồn thì còn phải có bằng chứng runtime hoặc lượt chạy thống nhất riêng.
- Không thay dữ liệu G2 hoặc detector BVP. NaN được xử lý trong train fold theo pipeline đã rà, không fit imputer/scaler trên toàn bảng.

## Kết quả ngoài mẫu

Trung bình qua 15 subject; độ lệch chuẩn mẫu `ddof=1` và confusion matrix nằm trong JSON. Không dùng điểm fit trên toàn dữ liệu hoặc kết quả tuning.

| Bài toán | Mô hình | Macro-F1 | Balanced Accuracy |
| --- | --- | ---: | ---: |
| Ba lớp | Dummy | 0.2350 | 0.3333 |
| Ba lớp | LR | 0.5894 | 0.6315 |
| Ba lớp | KNN | 0.5555 | 0.5753 |
| Ba lớp | SVM | 0.6177 | 0.6595 |
| Ba lớp | RF | 0.6480 | 0.6700 |
| Ba lớp | MLP | 0.6462 | 0.6815 |
| Nhị phân | Dummy | 0.4123 | 0.5000 |
| Nhị phân | LR | 0.8103 | 0.8304 |
| Nhị phân | KNN | 0.7258 | 0.7541 |
| Nhị phân | SVM | 0.8424 | 0.8663 |
| Nhị phân | RF | 0.8735 | 0.8865 |
| Nhị phân | MLP | 0.7986 | 0.8225 |

Đủ 12 tổ hợp model/task, không thiếu mô hình. Không suy ra ý nghĩa thống kê chỉ từ thứ hạng trung bình. Phần chạy Dummy/LR và kiểm tra/hợp nhất của Mai đã hoàn thành; không xác nhận nguồn runtime đồng nhất hoặc thực hiện G4 trong lượt này.

## Bàn giao

- Gói cuối: `outputs/g3_group_predictions_six_models_20261108/` và ZIP cùng tên, ngoài Git. Có predictions, metrics, summary/confusion matrix, 180 splits, config/environment, verification, log, hình minh họa và checksum. Không lấy gói bốn mô hình cũ làm bộ cuối.
- Các gói producer nguyên bản kèm trong `producer_archives/`. ZIP source cũ không còn ở Downloads khi đóng gói cuối; các file đã giải nén được đóng lại thành `reviewed_KNN_SVM_source_repacked.zip`, lưu hash riêng, không coi giống byte ZIP gốc.
- Báo cáo nhỏ: [summary.json](../reports/g3_group/summary.json), [metrics_by_subject.csv](../reports/g3_group/metrics_by_subject.csv) (180 dòng), [received_packages.json](../reports/g3_group/received_packages.json).
- **95/95 test đạt**; log `outputs/g3_group_all_tests.txt`. Test kiểm tra thay nhãn, leakage dù cập nhật checksum, factory/param chưa review và không sửa producer identity.

Merger chuẩn `scripts/merge_g3_predictions.py` vẫn giữ điều kiện cùng code. Runner cũ dùng luồng review riêng `scripts/merge_g3_received.py` với `--standard-runs`, `--knn-svm-run`, `--reviewed-source`, `--output`; chạy `--help` để xem. Chọn output mới và ba lô model gốc, không nối thêm bảng bốn mô hình cũ vì sẽ trùng dự đoán. Không giảm điều kiện merger chuẩn.

Các hình được render từ JSON bằng Python dùng để vẽ, không dùng môi trường đó để huấn luyện. Chưa commit/push cập nhật hợp nhất; ZIP và dự đoán chi tiết giữ ngoài Git.
