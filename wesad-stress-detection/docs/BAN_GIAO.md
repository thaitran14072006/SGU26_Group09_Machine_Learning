# Tổng hợp và chốt G1 — nhóm 09

Ngày tổng hợp: **06/11/2026**. Mai tổng hợp theo yêu cầu người dùng, từ các đầu ra đã có trong repo và biên bản đối chiếu của Như tại commit `7cba519`. Báo cáo đối chiếu kỹ thuật của Như là [G1_DOI_CHIEU_S2_BON_THANH_VIEN.md](G1_DOI_CHIEU_S2_BON_THANH_VIEN.md); phần dưới là tổng hợp để chốt G1, không thay tên người thực hiện các lần chạy hoặc ghi nhận review thay thành viên khác.

## Kết luận chốt

**Chốt hoàn thành phạm vi G1 theo tiêu chí kết quả đầu ra được nhóm chấp nhận trong biên bản ngày 06/11/2026.** Đã có đầu ra môi trường/hạ tầng, khảo sát dữ liệu, thiết kế tiền xử lý/cửa sổ, hợp đồng feature/schema và đối chiếu đủ bốn CSV S2. Người tổng hợp chốt: Mai theo phân công tổ chức nhóm người dùng đã xác định.

Tiêu chí nghiệm thu S2 của nhóm là bốn đầu ra giống nhau. Điều kiện nghiêm ngặt cùng mã nguồn và môi trường **chưa đạt**; đây là ngoại lệ được ghi rõ trong biên bản, không phải chứng nhận tái lập hoàn toàn theo điều kiện môi trường ban đầu. Không đổi manifest, không sửa JSON kiểm tra cũ và không diễn giải cờ `four_machine_acceptance=false` của công cụ thành true.

## Đầu ra G1 của từng thành viên

| Thành viên | Phạm vi G1 | Bằng chứng trong repo | Kết luận trong phạm vi G1 |
| --- | --- | --- | --- |
| Thái | Repo, môi trường/thư viện khóa, CLI/logging, loader PKL và khung LOSO | `pyproject.toml`, `requirements.lock.txt`, `src/wesad/data.py`, `src/wesad/evaluation.py`, scripts và [bàn giao Thái](BAN_GIAO_CONG_VIEC_THAI_2026-11-02.md) | Có đủ đầu ra hạ tầng; bàn giao Thái ghi 14 test hạ tầng đạt. Không lấy bộ raw 21 cũ làm bộ nghiệm thu 112 hiện tại |
| Nam | Khảo sát nhãn, mẫu tín hiệu, thiếu/bất thường | `notebooks/01_eda.ipynb`, `notebooks/reports/tables/eda_step5/eda_summary.json` và các bảng/hình EDA | Có đầu ra EDA 15 người, 6.196 cửa sổ, 112 thống kê raw; không coi là nghiệm thu feature sinh lý hoặc kết quả mô hình |
| Như | Đồng bộ, thiết kế bộ lọc/cửa sổ 60/5 majority >=90%, test biên và tổng hợp đối chiếu S2 | [thiết kế Như](G1_NHU_THIET_KE_TIEN_XU_LY.md), [bàn giao Như](G1_NHU_BAN_GIAO.md), [kiểm tra S2](G1_NHU_KIEM_TRA_S2.md), [biên bản bốn gói](G1_DOI_CHIEU_S2_BON_THANH_VIEN.md) | Theo bàn giao Như: 35/35 test đạt; nhận đủ bốn gói và đối chiếu đầu ra đạt theo quyết định nhóm |
| Mai | Danh mục feature, đơn vị, nhãn nhị phân, cách đánh dấu thiếu, schema và bàn giao S2 | `configs/feature_schema.json`, `docs/G1_DANH_MUC_FEATURE.csv`, [hợp đồng feature](G1_HOP_DONG_FEATURE.md), phần bàn giao máy Mai bên dưới | Chốt hợp đồng thiết kế 68 feature + 8 metadata; JSON/CSV đồng nhất; gói raw112 máy Mai nằm trong bốn gói đã đối chiếu |

Đây là tổng hợp nội dung và bằng chứng hiện có, không phải xác nhận đã diễn ra đầy đủ các lượt review cá nhân theo vòng trong bản phân công. Thành viên vẫn xử lý góp ý nếu có; không tạo chữ ký hoặc ghi nhận đồng ý thay người khác.

## Kết quả đối chiếu S2 do Như tổng hợp

Theo biên bản Như ngày 06/11/2026, đã nhận `Thai_G1_S2.zip`, `Mai_test.zip`, `Nam_test.zip` và gói tham chiếu Như. Phạm vi nhận diện là các gói mang tên thành viên do người dùng cung cấp, không tự xác thực tính độc lập của các máy chạy.

- Bộ được nghiệm thu G1: `all_sensors_raw_stats_v1`, 112 thống kê raw, 63 ngực + 49 cổ tay; thêm 8 metadata thành 120 cột.
- Cả bốn CSV có **395 cửa sổ**, nhãn **219 Baseline / 113 Stress / 63 Amusement**.
- Khóa cửa sổ, nhãn, mốc thời gian, quality flag và toàn bộ giá trị feature khớp; sai lệch tuyệt đối lớn nhất **0**.
- Cùng SHA-256 CSV: `14bd24e6a471f58693e7822c03cc7b492eeba314dab290bc10a8281285d8384a`.
- Cùng raw S2, cấu hình 60 giây/bước 5 giây, majority >=90%; không chỉ so số dòng.
- Kiểm tra bổ sung trên máy Mai khi tổng hợp: hash S2.csv hiện có khớp hash chung trong biên bản. Không chạy lại các gói trên máy Như hoặc tự nhận đã trực tiếp kiểm tra ba gói còn lại.

## Ngoại lệ được giữ nguyên

| Gói | Khác biệt so với gói Như được ghi trong biên bản |
| --- | --- |
| Thái | Code hash khác, git_dirty=true |
| Mai | Code hash khác, snapshot gói không ghi nhận Git commit; source xuất từ fa7b694 được truy vết riêng trong bàn giao Mai |
| Nam | Python 3.13.5, numpy 2.5.3, pandas 3.0.6; imbalanced-learn không ghi nhận; code hash khác, git_dirty=true; kiểm tra gói đơn chưa đạt phiên bản môi trường |

Kết quả nghiêm ngặt giữa Như và từng gói chưa đạt vì các khác biệt trên. Nhóm đã quyết định không yêu cầu chạy lại để chốt đối chiếu G1 theo đầu ra. Khi thực hiện lượt chung tiếp theo cần khóa cùng source sạch, Python 3.12.14 và thư viện; không bỏ điều kiện hoặc sửa chữ ký để ép resume.

## Phạm vi bàn giao sau G1

Bộ raw112 dùng kiểm tra G1; bộ sinh lý68 là hợp đồng thiết kế để triển khai, chưa phải extractor đã chạy đạt. HRV/PRV, tonic/phasic/SCR, bộ lọc sinh lý và ACC/EMG 12 đoạn 5 giây vẫn thuộc triển khai G2. Chốt G1 không có nghĩa chốt G2/G3 hoặc số liệu mô hình. Raw/gói feature/ZIP tiếp tục chia sẻ ngoài Git; không tạo thêm bản báo cáo JSON riêng trong docs.

## Bàn giao máy Mai và bằng chứng gói đơn trước khi tổng hợp

Phần sau giữ kết quả cục bộ ngày 05/11/2026 để truy vết. Các câu “chưa đối chiếu” trong kết quả gói đơn là trạng thái tại lần chạy đó; trạng thái chung mới nhất nằm ở phần tổng hợp phía trên.

# Bàn giao G1 phần Mai

Cập nhật ngày 05/11/2026. Phần việc theo PHAN_CONG_MAY_HOC.docx: chốt danh mục feature, đơn vị, nhãn nhị phân, cách đánh dấu thiếu và schema đầu ra. Chỉ hoàn thiện thiết kế G1 và gói S2 cục bộ; chưa triển khai extractor sinh lý hay chạy lô G2.

## Năm file bàn giao trên Git

| File | Nội dung |
| --- | --- |
| `README.md` | Mục bàn giao Mai và liên kết tới đầu ra G1 |
| `configs/feature_schema.json` | Hợp đồng sinh lý 68 feature, metadata, công thức/thông số, đơn vị và bằng chứng đơn vị gộp trong unit_policy.unit_evidence |
| `docs/G1_DANH_MUC_FEATURE.csv` | Danh mục 68 feature, đúng tên/thứ tự/đơn vị/phương pháp như JSON |
| `docs/G1_HOP_DONG_FEATURE.md` | Quy ước nhãn, NaN, cửa sổ, feature và phối hợp |
| `docs/BAN_GIAO.md` | Trạng thái, kết quả tự kiểm tra S2 và hướng dẫn gửi gói cho Như |

Thiết kế 68 feature gồm 37 ngực, 31 cổ tay và 8 metadata, trạng thái design_locked_implementation_not_validated. Đây là thiết kế phục vụ proposal để triển khai tiếp; processor raw hiện tại của nhóm tạo 112 thống kê (63 ngực, 49 cổ tay), không áp detector/PSD sinh lý trong hợp đồng này. Schema raw thực tế lấy từ file sinh cùng gói CSV ngoài Git.

Nhãn nhị phân 2→1, 1/3→0; loại các nhãn khác. Feature sinh lý thiếu dùng NaN/ô CSV trống kèm lý do, không dùng số 0 giả; metadata không thiếu, Inf là lỗi. Đơn vị và quy tắc chuyển đổi ghi trong schema/hợp đồng; bằng chứng được kiểm tra với README WESAD gốc và mẫu S2 đã căn thời gian, không suy rộng thành kiểm tra toàn bộ 15 người.

## Kiểm tra S2 và người tổng hợp

Phân công yêu cầu cả bốn chạy mẫu S2. Mai đã tự kiểm tra gói máy mình để phát hiện lỗi cột/nhãn/checksum trước khi bàn giao. Theo cách nhóm người dùng đã xác định, Như nhận ZIP và tổng hợp đối chiếu bốn máy. Mai không thay Như ghi nhận nghiệm thu chung.

- Gói máy Mai: `data/features/g1_s2_raw112_py31214_fa7b694/`.
- Python 3.12.14; đủ 16 thư viện khớp file khóa.
- Source commit: `fa7b6942d69717cf6ea7f48ca364f576486de08f`.
- Gói cục bộ passed: 395 cửa sổ, 8 metadata + 112 feature; Baseline 219, Stress 113, Amusement 63.
- Trong 1.204 cửa sổ ứng viên: 809 loại theo nhãn, 0 loại vì thiếu mẫu/nonfinite; giữ 395.
- Đã kiểm tra cột, khóa, nhãn, thời gian, feature hữu hạn, môi trường, chữ ký mã/config, hash raw/CSV/schema, bảng ghép, số lượng QA và đẳng thức thống kê.
- Báo cáo chỉ có phạm vi g1_raw112_single_package; four_machine_acceptance=false, physiological_acceptance=false.

### Bằng chứng kết quả máy Mai

Kết quả: **passed**, phạm vi `g1_raw112_single_package`, phiên bản `all_sensors_raw_stats_v1`. Đây là tự kiểm tra một gói trước khi bàn giao cho Như; chưa thực hiện so sánh gói của các máy khác.

| Thông tin | Giá trị |
| --- | --- |
| Raw S2 SHA-256 | `36ef5e8afc0f91998eefba7c12fc9fa97b7b07198cbec0126917d7abb436ca23` |
| Code SHA-256 | `b0d02f7a9f80e754342c8fa73f7b6d2e57a2b772a5e01882517b7ef6bc739a96` |
| S2.csv SHA-256 | `14bd24e6a471f58693e7822c03cc7b492eeba314dab290bc10a8281285d8384a` |
| Dung sai dùng kiểm tra đẳng thức thống kê raw | rtol=1e-07, atol=1e-09 |
| Số feature ngực / cổ tay / kết hợp | 63 / 49 / 112 |
| Nghiệm thu bốn máy | Chưa; `four_machine_acceptance=false` |

Các thư viện khoa học ghi trong gói: numpy 2.5.4, pandas 3.1.0, PyYAML 6.0.3, scipy 1.18.1, scikit-learn 1.9.1, imbalanced-learn 0.14.2. Toàn bộ 16 phiên bản đã được kiểm tra khớp file khóa. Dung sai trên dùng trong kiểm tra gói cục bộ, chưa phải kết quả so giá trị giữa các máy.

Gửi Như nguyên thư mục gói ở trên dưới dạng ZIP ngoài Git, giữ đủ CSV/schema/config/manifest/checksum. Không gửi .venv-g1 hoặc bản Python đã cài. Không dùng gói 21 feature cũ để đối chiếu với gói 112.

Repo đã pull tới 9fcbbf7; gói này được tạo bằng fa7b694 trước lần pull đó. Như cần kiểm tra cùng code/source hash trước khi so các gói; nếu khác source thì thống nhất phiên bản chạy lại. Không sửa manifest/chữ ký để ép khớp. Chưa có kết quả đối chiếu bốn máy trong phần bàn giao này.

## Kiểm tra thiết kế đã thực hiện

JSON/CSV khớp 68 feature; 76 tên cột duy nhất gồm metadata; các nhóm 37/31 và tham chiếu phương pháp hợp lệ. Chốt công thức detector/ngưỡng/PSD, NaN không SCR, ACC/EMG 12 đoạn 5 giây và đơn vị. Extractor sinh lý đúng hợp đồng chưa triển khai/kiểm chứng, không coi bộ thử 175 cột là sản phẩm G1 này.
