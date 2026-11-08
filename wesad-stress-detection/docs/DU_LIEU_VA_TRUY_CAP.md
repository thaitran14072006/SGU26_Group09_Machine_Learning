# Nguồn dữ liệu và hướng dẫn truy cập

## Bảng đặc trưng dùng cho G3

G3 dùng gói `g2_physiological68_group15_20261107.zip` do nhóm tạo từ WESAD: **15 subject, 6.189 cửa sổ, 68 feature sinh lý + 8 metadata**, phiên bản `physiological_68_v1`. Đây là dữ liệu đã xử lý của nhóm, không phải file raw tải trực tiếp từ WESAD. Không dùng bộ raw112 thay cho bộ này.

**Hiện chưa có link tải công khai.** Thành viên/người đánh giá nhận ZIP trực tiếp từ Mai hoặc đầu mối lưu dữ liệu của nhóm. Repository không tự cung cấp ZIP và chưa bảo đảm người ngoài nhóm tải được chỉ bằng một URL; khi có Drive/OneDrive, bổ sung link và quyền truy cập tại mục này. Không đưa raw hoặc CSV feature lớn lên Git.

Trên máy đang bàn giao, gói nằm tại:

```text
D:\mayhoc\Machine-learning\wesad-stress-detection\data\features\g2_physiological68_group15_20261107.zip
```

Thư mục đã giải nén trên máy Mai, có thể dùng trực tiếp mà không giải nén lại:

```text
D:\mayhoc\Machine-learning\wesad-stress-detection\data\features\g2_physiological68_group15_20261107
```

Trong thư mục này, `all_subjects.csv` là bảng đầu vào G3, `feature_schema.json` xác định các cột feature, `checksums.json` dùng kiểm tra toàn vẹn. Cấu hình `configs/defaults.yaml` đã trỏ tới thư mục tương ứng bằng đường dẫn tương đối từ project.

Các đường dẫn trên là vị trí cục bộ, không phải link tải cho máy khác. Thành viên khác cần nhận ZIP từ nhóm và đặt đúng cấu trúc thư mục. Sau khi nhận ZIP và kiểm tra hash, giải nén tại thư mục project:

```powershell
Expand-Archive -LiteralPath 'C:\Users\Mai\Downloads\g2_physiological68_group15_20261107.zip' -DestinationPath 'data/features'
```

Thay đường dẫn Downloads theo máy nhận; không ghi đè thư mục đã có dữ liệu khác. Kết quả cần có:

```text
data/features/g2_physiological68_group15_20261107/
    all_subjects.csv
    feature_schema.json
    checksums.json
    handoff_checksums.json
    config.json
    quality_report.csv
    common_windows.csv
    S2.csv, S2.json, ...
```

G3 chỉ cần đọc bảng tổng và schema bằng cấu hình `configs/defaults.yaml`; không cần raw PKL để chạy mô hình từ bảng đã nhận. Không nối bảng tổng với CSV từng subject vì sẽ lặp dữ liệu. NaN có lý do được giữ nguyên và xử lý trong train fold, không điền trước trên toàn bảng.

## Kiểm tra SHA-256

Hash được tính lại trên các file bàn giao ngày 08/11/2026. So sánh không phân biệt chữ hoa/thường.

| File | SHA-256 |
| --- | --- |
| ZIP G2 | `68f10ae940e4c1b3d269cf5836803800f3460a92755a5494cd8468d479075c50` |
| all_subjects.csv | `2f2fddd678f333a800f2869d2e78b407934b2c3492190ed70e28d8a9107fd8ba` |
| feature_schema.json | `8730ca9189547bd54e0ec2cec5f574a5e2a989b64f6ea6b0a8dd4a186669f998` |
| checksums.json | `e90d3c495b35a837d9349dd489d0cba0a9433b7f89b5b357e177211dca9a88e4` |

Trước khi giải nén:

```powershell
Get-FileHash -LiteralPath 'C:\Users\Mai\Downloads\g2_physiological68_group15_20261107.zip' -Algorithm SHA256
```

Sau giải nén, tại project:

```powershell
$g2Folder = Resolve-Path 'data/features/g2_physiological68_group15_20261107'
Get-FileHash -LiteralPath "$g2Folder/all_subjects.csv", "$g2Folder/feature_schema.json", "$g2Folder/checksums.json" -Algorithm SHA256
$g2Checks = Get-Content -LiteralPath "$g2Folder/checksums.json" -Raw | ConvertFrom-Json
foreach ($entry in $g2Checks.PSObject.Properties) {
    $actualHash = (Get-FileHash -LiteralPath (Join-Path $g2Folder $entry.Name) -Algorithm SHA256).Hash
    if ($actualHash -ne $entry.Value) { throw "Checksum sai: $($entry.Name)" }
}
'Các checksum gốc đều khớp'
```

Chỉ chạy vòng kiểm tra trên sau khi `checksums.json` đã khớp hash chuẩn trong bảng. Hash sai thì nhận lại đúng gói hoặc xác minh phiên bản, không sửa checksum để ép khớp. `checksums.json` kiểm tra CSV/schema gốc; `handoff_checksums.json` kiểm tra thêm các tệp bàn giao. Không chỉnh sửa trực tiếp CSV bằng Excel rồi coi đó là cùng bản dữ liệu.

## Nguồn WESAD gốc

- **Link tải raw WESAD nhóm sử dụng:** [WESAD trên Sciebo của Đại học Siegen](https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx?opendetails=). Đây là link người dùng cung cấp cho dữ liệu gốc; không phải link tải ZIP 68 feature do nhóm xử lý.
- [Trang WESAD của nhóm tác giả](https://kristofvl.github.io/usi/data_wesad.html): vào mục **Download**, chọn liên kết **ICMI'18 dataset** để nhận bản ghi gốc và tài liệu đi kèm.
- [WESAD trên UCI](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection): thông tin và trích dẫn dataset, DOI [10.24432/C57K5T](https://doi.org/10.24432/C57K5T).
- [Bài báo WESAD, ICMI 2018](https://doi.org/10.1145/3242969.3242985): Schmidt và cộng sự. Đọc điều kiện sử dụng và ghi nguồn theo trang tác giả; dữ liệu gốc phục vụ nghiên cứu phi thương mại theo điều kiện được công bố tại đó.

Trang tác giả/UCI đã được tra cứu ngày 08/11/2026; link Sciebo được cập nhật theo đường dẫn người dùng cung cấp. WESAD gốc không chứa sẵn bảng 68 feature của nhóm hoặc checksum ZIP G2 trên; không dùng checksum của gói nhóm cho ZIP raw tải từ tác giả.

## Tái tạo từ raw khi không nhận gói G2

Giải nén WESAD và đặt `data_root` trong `configs/local.yaml` sao cho có các đường dẫn `data_root/S2/S2.pkl`, ... `data_root/S17/S17.pkl` (không có S12). Cài Python 3.12.14 và thư viện theo `requirements.lock.txt`, đọc [bàn giao G2](G2_BAN_GIAO.md) và [biên bản nghiệm thu](G2_NGHIEM_THU.md) trước khi chạy.

```powershell
.\.venv-g1\Scripts\python.exe -X utf8 scripts/extract_features.py --config configs/features_g2.yaml --local configs/local.yaml --subjects S2 S3 S4 S5 S6 S7 S8 S9 S10 S11 S13 S14 S15 S16 S17 --output-dir data/features/g2_physiological68_regenerated
.\.venv-g1\Scripts\python.exe -X utf8 scripts/finalize_g2_physiological.py --source data/features/g2_physiological68_regenerated --output data/features/g2_physiological68_regenerated_delivery --raw-root data/raw --report outputs/g2_regenerated_verification.json
```

Lệnh minh họa dùng `data/raw` làm raw root; nếu raw ở vị trí khác, thay `--raw-root` bằng đúng đường dẫn trong local config. Giữ bản tái tạo ở thư mục riêng. Cùng định nghĩa chưa bảo đảm ZIP giống byte khi khác source/môi trường/log; không gán hash chuẩn ở trên cho gói mới nếu chưa xác minh. Lượt chạy gói chính thức và bằng chứng nguồn được ghi trong bàn giao G2. Phần G3 đang khóa hash bảng chính thức trong `configs/defaults.yaml`.
