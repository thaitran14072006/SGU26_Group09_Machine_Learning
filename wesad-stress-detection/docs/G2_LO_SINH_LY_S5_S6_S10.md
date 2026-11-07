# Bàn giao lô sinh lý S5, S6, S10 — 07/11/2026

Đã chạy lô S5/S6/S10 bằng bản nguồn `7720b01` và cấu hình/hợp đồng của gói `g2_nhu_physiological68_20261107.zip`. Phiên bản `physiological_68_v1`, provider `wesad.preprocessing_physiological:prepare_subject`; giữ nguyên phương pháp và ngưỡng BVP. Không chạy lại lô Như để đối chiếu nhiều máy.

## Kết quả

| Subject | Ứng viên | Loại do nhãn | Giữ | Baseline | Stress | Amusement | Cửa sổ thiếu nhóm nhịp BVP | Ô feature thiếu |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S5 | 1240 | 826 | 414 | 230 | 119 | 65 | 392 | 2472 |
| S6 | 1403 | 991 | 412 | 226 | 121 | 65 | 412 | 2664 |
| S10 | 1088 | 662 | 426 | 226 | 135 | 65 | 326 | 2286 |
| Tổng | 3731 | 2479 | 1252 | 682 | 375 | 195 | 1130 | 7422 |

Mỗi CSV có 68 feature (37 ngực, 31 cổ tay) và 8 metadata. Không loại thêm cửa sổ do thiếu mẫu, nonfinite hoặc biên lọc. Sáu feature nhịp/IBI BVP cùng thiếu khi không đủ nhịp đáng tin; feature PSD BVP vẫn có thể có giá trị. NaN giữ nguyên với cờ lý do, chưa impute hoặc chuẩn hóa.

## Kiểm tra bàn giao

- Gói Như: ZIP SHA-256 `909a4b2e153e152474f2dd5797946968cb21a1437a6f051c4f517baa8fcef28f`; checksum gốc và checksum bàn giao bổ sung đều đúng, hợp đồng kèm gói giống từng byte hợp đồng trong checkout. Kiểm tra trực tiếp gói Như đạt 4 subject, 1658 cửa sổ.
- Lô vừa chạy: thứ tự 76 cột đúng hợp đồng, subject/nhãn nhị phân/độ thuần nhãn và lưới thời gian 60/5 hợp lệ; không trùng khóa hoặc Inf. NaN có lý do cảm biến tương ứng.
- Manifest, số cửa sổ, checksum CSV và raw hash đối chiếu PKL local đều đúng. Số ứng viên/giữ/loại và số ô thiếu khớp QA. Nối mỗi CSV từng subject một lần khớp `all_subjects.csv`.
- ZIP lô mới đã đọc lại để kiểm tra tính toàn vẹn và hash các tệp bàn giao.

## Nguồn chạy

Để không dùng các thay đổi cục bộ chưa commit trong checkout, đã xuất bản nguồn `7720b01` vào thư mục riêng trong `outputs/`. Khôi phục kiểu xuống dòng của các tệp từ bản Git để hash toàn bộ mã chạy khớp chính xác gói Như: `abd964e88fa0e5922c8bfc2cb71e1f4ad46ebb1d381e3f6bf82571817e1b452b`. Bằng chứng tệp/kiểu xuống dòng nằm trong `SOURCE_RECONSTRUCTION.json` kèm gói. Không chỉnh mã tính feature hoặc giả hash/manifest.

Cùng Python 3.12.14 và sáu phiên bản thư viện ghi nhận như Như; build Python của máy chạy là standalone, khác conda-forge của Như. Vì vậy chữ ký môi trường toàn phần khác nhau; giữ nguyên metadata thật và chưa ghép hai gói thành một gói nghiệm thu chung. Git metadata có thể null khi sandbox không đọc được Git root; commit nguồn xuất được ghi riêng trong bằng chứng nguồn chạy.

## Tệp bàn giao ngoài Git

- Thư mục: `data/features/g2_physiological68_S5_S6_S10_20261107/`.
- ZIP gửi nhóm: `data/features/g2_physiological68_S5_S6_S10_20261107.zip`.
- ZIP SHA-256: `dc0de86176b6fa53b58a6e9f0db5acb4aa471a0645861eb08341040383b386d4`.
- Gồm CSV từng subject, bảng tổng, manifest, config, schema gốc sinh cùng CSV, hợp đồng đầy đủ riêng, QA, checksum, log, bằng chứng nguồn chạy và hướng dẫn bàn giao. Không sửa schema/checksum gốc để ép khớp công cụ kiểm tra.
- Báo cáo số nhỏ trong Git: [S5_S6_S10_verification.json](../reports/g2_physiological68/S5_S6_S10_verification.json).

Lô ba subject đã hoàn thành chạy và kiểm tra gói. BVP còn thiếu nhiều, đặc biệt S6 thiếu toàn bộ sáu feature nhịp/IBI; đây chưa phải nghiệm thu chất lượng sinh lý hoặc đủ 15 người. Chưa ghép với raw112, các gói sinh lý tham khảo cũ hoặc bảng chẩn đoán 72 cột. Chưa chạy G3, không thêm phần RAM và chưa commit/push.
