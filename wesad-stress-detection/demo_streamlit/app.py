"""WESAD demo: Streamlit upload -> process -> two classifications, plus subject/LOSO explorer."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import hashlib
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from common import DEFAULT_DATA, LABELS, SUBJECTS, feature_columns, get_target, root_dir, to_absolute

st.set_page_config(page_title="WESAD | Stress Detection", page_icon="", layout="wide", initial_sidebar_state="expanded")
ROOT = root_dir()

# Giữ giao diện sáng, giống bản Streamlit gốc.
# Chỉ định kiểu cho panel thống kê và nút chính; không ép màu nền tối.
st.markdown("""<style>
[data-testid="stMetric"] {
    background: #f7f9fc;
    padding: 12px 16px;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
}
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: #172033; }
h1, h2, h3 { letter-spacing: -0.025em; }
div.stButton > button[kind="primary"] { font-weight: 650; }
.small-note { color: #64748b; font-size: .86rem; }
</style>""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def read_local_csv(path: str, stamp: int) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"subject_id": str, "window_id": str})

@st.cache_resource(show_spinner=False)
def read_model(path: str, stamp: int):
    return joblib.load(path)  # Always a trusted local file; never accept joblib uploads.


def supported_labels(frame: pd.DataFrame, task: str):
    try:
        y = get_target(frame, task)
        valid = {0, 1} if task == "binary" else {1, 2, 3}
        if not set(y.dropna().unique()).issubset(valid):
            return None
        return y.astype(int)
    except (KeyError, ValueError, TypeError):
        return None


def stats_table(pred: pd.Series, task: str) -> pd.DataFrame:
    counts = pred.value_counts().reindex(list(LABELS[task]), fill_value=0)
    total = counts.sum()
    return pd.DataFrame({"Trạng thái": [LABELS[task][k] for k in counts.index],
                         "Số cửa sổ": counts.to_numpy(dtype=int),
                         "Tỷ lệ (%)": (counts.to_numpy() * 100 / total).round(2) if total else 0})


def file_fingerprint(files):
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode("utf-8")); h.update(f.getvalue())
    return h.hexdigest()


def upload_tables(files):
    frames = []
    for f in files:
        t = pd.read_csv(BytesIO(f.getvalue()), dtype={"subject_id": str, "window_id": str})
        if t.empty:
            raise ValueError(f"{f.name}: CSV rỗng")
        if t.columns.duplicated().any():
            raise ValueError(f"{f.name}: cột trùng tên")
        if "subject_id" not in t:
            t["subject_id"] = "CSV mới"
        t["_source_file"] = f.name
        frames.append(t)
    if len({tuple(x.columns) for x in frames}) > 1:
        raise ValueError("Nhiều CSV có các cột khác nhau. Hãy chuẩn hóa về cùng schema 68 đặc trưng.")
    return pd.concat(frames, ignore_index=True)


def fmt_count(n):
    return f"{n:,}".replace(",", ".")

st.title(" WESAD Stress Detection")
st.caption("PHYSIOLOGICAL SIGNALS  ·  68 FEATURES  ·  BINARY + MULTICLASS")
st.markdown("Dữ liệu đầu vào → **Process Data** → hai kết quả phân loại song song, tương tự luồng thao tác trong video mẫu.")

with st.sidebar:
    st.header(" Điều khiển")
    source = st.radio("Nguồn dữ liệu", ["Dữ liệu WESAD trong dự án", "Tải CSV mới"], index=0)
    raw_path = st.text_input("CSV nguồn", DEFAULT_DATA, disabled=(source != "Dữ liệu WESAD trong dự án"))
    uploaded = st.file_uploader("Chọn một hoặc nhiều CSV", type=["csv"], accept_multiple_files=True,
                                disabled=(source != "Tải CSV mới"), help="CSV có 68 cột đặc trưng cùng schema huấn luyện; có thể kèm subject_id, label_original.")
    model_folder = st.text_input("Thư mục mô hình", "demo_streamlit/models")
    st.caption("Chỉ nạp joblib do nhóm tự huấn luyện, không nạp file từ người lạ.")
    pred_path = st.text_input("LOSO predictions.csv", "experiments/04_additional/EXP-017_KNN/predictions.csv")

try:
    if source == "Dữ liệu WESAD trong dự án":
        csv_path = to_absolute(ROOT, raw_path)
        if not csv_path.is_file():
            st.error(f"Không tìm thấy dữ liệu: {csv_path}")
            st.stop()
        data = read_local_csv(str(csv_path), csv_path.stat().st_mtime_ns).copy()
        fingerprint = (str(csv_path), csv_path.stat().st_mtime_ns)
    else:
        if not uploaded:
            st.info("Hãy tải lên một hoặc nhiều CSV để bắt đầu. Bạn vẫn có thể dùng dữ liệu G2 trong dự án từ sidebar.")
            st.stop()
        if sum(len(f.getvalue()) for f in uploaded) > 60 * 1024 * 1024:
            st.error("Tổng file tải lên vượt 60 MB; hãy chia thành các đợt nhỏ.")
            st.stop()
        data = upload_tables(uploaded)
        fingerprint = ("uploaded", file_fingerprint(uploaded))
    if "subject_id" not in data:
        raise ValueError("Thiếu subject_id")
    data["subject_id"] = data["subject_id"].astype(str)
except Exception as exc:
    st.error(f"Không đọc được CSV: {exc}")
    st.stop()

subjects = sorted(data["subject_id"].dropna().unique(), key=lambda s: int(s[1:]) if str(s).startswith("S") and str(s)[1:].isdigit() else 9999)
with st.sidebar:
    chosen = st.selectbox("Người tham gia", ["Tất cả người"] + subjects)
    st.caption(f"Đã đọc {fmt_count(len(data))} cửa sổ · {len(subjects)} người/nhóm dữ liệu")

view = data if chosen == "Tất cả người" else data.loc[data["subject_id"] == chosen]
if view.empty:
    st.warning("Không có dữ liệu theo bộ lọc.")
    st.stop()

folder = to_absolute(ROOT, model_folder)
model_paths = {task: folder / f"knn_top30_{task}.joblib" for task in ("binary", "multiclass")}
models_ready = all(p.is_file() for p in model_paths.values())

tab_predict, tab_data, tab_loso, tab_export = st.tabs([
    " Dự đoán", " Khám phá 15 người", " Đánh giá LOSO", " Xuất kết quả & hướng dẫn"
])

with tab_predict:
    st.subheader("01 / Nạp dữ liệu và xử lý")
    c1, c2, c3 = st.columns(3)
    c1.metric("Cửa sổ đang chọn", fmt_count(len(view)))
    c2.metric("Số người", view["subject_id"].nunique())
    c3.metric("Pipeline", "2 / 2" if models_ready else "Chưa sẵn sàng")
    if not models_ready:
        st.info("Chưa tìm thấy hai pipeline đã fit. Tại thư mục gốc dự án, chạy `python demo_streamlit/train_models.py` một lần, sau đó tải lại giao diện.")
        st.code("python demo_streamlit/train_models.py", language="powershell")
    else:
        window_count = st.slider("Số cửa sổ cần xử lý", min_value=1, max_value=len(view), value=len(view), step=1,
                                 key=f"window_count_{chosen}_{len(view)}")
        sample = view.iloc[:window_count].copy()
        signature = (fingerprint, chosen, window_count, *( (str(p), p.stat().st_mtime_ns) for p in model_paths.values()))
        if st.button("▶ Process Data — Dự đoán cả hai", type="primary", use_container_width=True):
            try:
                with st.status("Đang xử lý dữ liệu sinh lý...", expanded=True) as s:
                    st.write("Kiểm tra schema 68 đặc trưng và thứ tự cột")
                    bundle_b = read_model(str(model_paths["binary"]), model_paths["binary"].stat().st_mtime_ns)
                    bundle_m = read_model(str(model_paths["multiclass"]), model_paths["multiclass"].stat().st_mtime_ns)
                    if not all(isinstance(b, dict) and "pipeline" in b and "feature_columns" in b for b in (bundle_b, bundle_m)):
                        raise ValueError("Joblib không đúng định dạng demo_streamlit; cần pipeline + feature_columns.")
                    columns_b = bundle_b["feature_columns"]
                    columns_m = bundle_m["feature_columns"]
                    if columns_b != columns_m or len(columns_b) != 68:
                        raise ValueError("Hai mô hình cần cùng schema 68 đặc trưng. Không thể dự đoán an toàn.")
                    missing = [col for col in columns_b if col not in sample.columns]
                    if missing:
                        raise ValueError(f"Thiếu {len(missing)} cột đặc trưng: {missing[:12]}")
                    st.write("Áp dụng pipeline Imputer → StandardScaler → SelectKBest(30) → KNN")
                    X = sample.loc[:, columns_b].apply(pd.to_numeric, errors="raise")
                    X = X.replace([np.inf, -np.inf], np.nan)
                    pred_b = bundle_b["pipeline"].predict(X)
                    pred_m = bundle_m["pipeline"].predict(X)
                    if not set(pred_b).issubset(LABELS["binary"]) or not set(pred_m).issubset(LABELS["multiclass"]):
                        raise ValueError("Mô hình trả về nhãn không hợp lệ.")
                    result = pd.DataFrame({"subject_id": sample["subject_id"].to_numpy(),
                                           "binary_prediction": pred_b, "multiclass_prediction": pred_m})
                    if "window_id" in sample:
                        result.insert(1, "window_id", sample["window_id"].to_numpy())
                    result["binary_label"] = result["binary_prediction"].map(LABELS["binary"])
                    result["multiclass_label"] = result["multiclass_prediction"].map(LABELS["multiclass"])
                    for task in ("binary", "multiclass"):
                        y_true = supported_labels(sample, task)
                        if y_true is not None:
                            result[f"{task}_true"] = y_true.map(LABELS[task]).to_numpy()
                    st.write(f"Đã suy luận {fmt_count(len(result))} cửa sổ")
                    s.update(label="Xử lý hoàn tất", state="complete", expanded=False)
                st.session_state["prediction_batch"] = {"signature": signature, "data": result}
            except Exception as exc:
                st.error(f"Dự đoán thất bại: {exc}")
                st.session_state.pop("prediction_batch", None)

        stored = st.session_state.get("prediction_batch")
        if stored is not None and stored["signature"] == signature:
            result = stored["data"]
            st.divider()
            st.subheader("02 / Kết quả dự đoán")
            st.caption("Hai bảng dưới đây là **phân bố nhãn dự đoán**, không phải accuracy hay xác suất mô hình.")
            left, right = st.columns(2, gap="large")
            for panel, task, title in [(left, "binary", "Binary · Stress / Non-Stress"),
                                       (right, "multiclass", "Multiclass · 3 trạng thái")]:
                with panel:
                    st.markdown(f"#### {title}")
                    table = stats_table(result[f"{task}_prediction"], task)
                    st.dataframe(table, hide_index=True, use_container_width=True,
                                 column_config={"Tỷ lệ (%)": st.column_config.NumberColumn(format="%.2f%%")})
                    st.bar_chart(table.set_index("Trạng thái")["Số cửa sổ"], height=235)
            st.download_button("⬇ Tải CSV dự đoán chi tiết", result.to_csv(index=False).encode("utf-8-sig"),
                               "wesad_predictions_binary_multiclass.csv", "text/csv", key="download_pred")
            st.caption("CSV chứa dự đoán của từng cửa sổ để đối soát; giao diện không hiển thị bảng từng cửa sổ theo yêu cầu.")
            if source == "Dữ liệu WESAD trong dự án":
                st.info("Đây là dự đoán minh họa nếu mô hình đã huấn luyện trên cả 15 người. Để báo cáo chất lượng trên người chưa thấy, dùng tab Đánh giá LOSO.")

with tab_data:
    st.subheader("Tổng quan dữ liệu WESAD")
    c1, c2, c3 = st.columns(3)
    c1.metric("Cửa sổ", fmt_count(len(view)))
    c2.metric("Số người", view.subject_id.nunique())
    c3.metric("Số đặc trưng số nhận diện", len(feature_columns(view)))
    st.markdown("#### Nhãn thật (nếu CSV có nhãn)")
    a, b = st.columns(2, gap="large")
    for panel, task, name in [(a, "binary", "Binary — 2 lớp"), (b, "multiclass", "Multiclass — 3 lớp")]:
        with panel:
            st.markdown(f"**{name}**")
            y = supported_labels(view, task)
            if y is None:
                st.info("CSV không chứa nhãn thật cho bài toán này.")
            else:
                table = stats_table(y, task)
                st.bar_chart(table.set_index("Trạng thái")["Số cửa sổ"], height=240)
                st.dataframe(table, use_container_width=True, hide_index=True)
    st.markdown("#### Dữ liệu đặc trưng")
    st.dataframe(view.head(1000), use_container_width=True, height=320, hide_index=True)
    st.caption("Bảng xem trước tối đa 1.000 dòng; nút tải xuất toàn bộ dữ liệu đang lọc.")
    if source == "Dữ liệu WESAD trong dự án":
        per_subject = data.groupby("subject_id").size().rename("Số cửa sổ").reset_index()
        st.bar_chart(per_subject.set_index("subject_id"))
    names = [c for c in feature_columns(view) if c in view.columns]
    shown = st.multiselect("Đặc trưng cần xem theo cửa sổ", names, default=names[:2], max_selections=4)
    if shown:
        st.line_chart(view[shown].head(400).reset_index(drop=True))
    st.download_button("⬇ Tải dữ liệu đang chọn", view.to_csv(index=False).encode("utf-8-sig"), "wesad_filtered.csv", "text/csv")

with tab_loso:
    st.subheader("Đánh giá Leave-One-Subject-Out (LOSO)")
    st.caption("Mỗi fold huấn luyện trên 14 người và kiểm tra người còn lại; đây mới là kết quả đánh giá tổng quát hóa.")
    path = to_absolute(ROOT, pred_path)
    if not path.is_file():
        st.info(f"Chưa có `predictions.csv` tại: {path}")
    else:
        try:
            preds = read_local_csv(str(path), path.stat().st_mtime_ns).copy()
            required = {"subject_id", "task", "y_true", "y_pred"}
            if missing := required.difference(preds.columns):
                raise ValueError(f"Thiếu cột: {sorted(missing)}")
            preds["subject_id"] = preds["subject_id"].astype(str)
            if chosen != "Tất cả người":
                preds = preds.loc[preds.subject_id == chosen]
            if "variant" in preds:
                variants = sorted(preds.variant.dropna().astype(str).unique())
                chosen_variant = st.selectbox("Cấu hình (variant)", variants, index=len(variants) - 1)
                preds = preds.loc[preds.variant.astype(str) == chosen_variant]
            pleft, pright = st.columns(2, gap="large")
            for panel, task in ((pleft, "binary"), (pright, "multiclass")):
                with panel:
                    st.markdown(f"#### {task.capitalize()}")
                    part = preds.loc[preds.task == task].copy()
                    if part.empty:
                        st.info("Chưa có dữ liệu LOSO")
                        continue
                    part["y_true"] = pd.to_numeric(part.y_true, errors="raise").astype(int)
                    part["y_pred"] = pd.to_numeric(part.y_pred, errors="raise").astype(int)
                    labels = list(LABELS[task])
                    st.metric("Macro-F1 (gộp các cửa sổ)", f"{f1_score(part.y_true,part.y_pred,labels=labels,average='macro',zero_division=0):.4f}")
                    st.metric("Balanced Accuracy (gộp)", f"{balanced_accuracy_score(part.y_true,part.y_pred):.4f}")
                    mat = confusion_matrix(part.y_true, part.y_pred, labels=labels)
                    names = [LABELS[task][label] for label in labels]
                    st.dataframe(pd.DataFrame(mat, index=["Thật: " + x for x in names],
                                              columns=["Dự đoán: " + x for x in names]), use_container_width=True)
                    st.download_button(f"⬇ LOSO {task}", part.to_csv(index=False).encode("utf-8-sig"),
                                       f"loso_{task}.csv", "text/csv", key=f"loso_dl_{task}")
            st.caption("Macro-F1 gộp theo cửa sổ khác với trung bình Macro-F1 của 15 fold trong summary.csv. Đừng thay thế hai loại chỉ số.")
        except Exception as exc:
            st.error(f"Không thể đọc LOSO: {exc}")

with tab_export:
    st.subheader("Hướng dẫn thao tác nhanh")
    st.markdown("""**1. Chọn dữ liệu** trong sidebar: CSV G2 của dự án hoặc tải nhiều CSV cùng schema 68 đặc trưng.

**2. Chọn người tham gia** (S2–S11, S13–S17) hoặc **Tất cả người**.

**3. Mở Dự đoán**, kéo thanh trượt đến số cửa sổ cần chạy, nhấn **Process Data**.

**4. Xem hai bảng kết quả** Binary và Multiclass đặt cạnh nhau, mỗi bảng có số lượng và tỷ lệ %.

**5. Mở Đánh giá LOSO** để trình bày chỉ số trên người chưa tham gia huấn luyện; tải CSV nếu cần.

Để quay demo: bật màn hình rộng, dùng dữ liệu đã chuẩn bị và không hiển thị dấu lỗi hay file cá nhân.""")
    st.download_button("⬇ Xuất dữ liệu lọc", view.to_csv(index=False).encode("utf-8-sig"), "wesad_export.csv", "text/csv")
    st.caption("Không phải công cụ chẩn đoán y khoa. Các kết quả trên dữ liệu đã dùng huấn luyện chỉ để minh họa quá trình suy luận.")
