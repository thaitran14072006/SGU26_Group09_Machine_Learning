import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score
from common import root_dir, load_table, feature_columns, get_target, LABELS, SUBJECTS, DEFAULT_DATA, to_absolute

st.set_page_config(page_title='WESAD | Stress Detection', page_icon='🫀', layout='wide')
root=root_dir()
st.title(' WESAD — Stress Detection')
st.caption('15 người · 68 đặc trưng sinh lý · Binary và Multiclass · Kết quả LOSO hoặc dự đoán từ mô hình triển khai')
with st.sidebar:
    st.header('Nguồn dữ liệu')
    raw_path=st.text_input('CSV 68 đặc trưng',value=DEFAULT_DATA)
    st.caption('Đường dẫn tương đối tính từ wesad-stress-detection/')
    pred_path=st.text_input('predictions.csv LOSO (tùy chọn)',value='experiments/04_additional/EXP-017_KNN/predictions.csv')
    model_dir=st.text_input('Thư mục model',value='demo_streamlit/models')
    choice=st.selectbox('Người tham gia',['Tất cả 15 người']+SUBJECTS)
    st.caption('CSV và joblib phải đến từ nguồn tin cậy. Không tải file joblib không rõ nguồn gốc.')

@st.cache_data(show_spinner=False)
def read_csv(path,stamp):
    return pd.read_csv(path)

@st.cache_resource(show_spinner=False)
def load_models(binary_path, multi_path, bstamp, mstamp):
    return joblib.load(binary_path),joblib.load(multi_path)

try:
    csv_file=to_absolute(root,raw_path)
    if not csv_file.is_file():
        st.warning(f'Chưa tìm thấy dữ liệu: {csv_file}. Hãy đặt dữ liệu WESAD đúng vị trí hoặc sửa đường dẫn bên trái.')
        st.stop()
    df=read_csv(str(csv_file),csv_file.stat().st_mtime_ns)
    if 'subject_id' not in df.columns: st.error('Thiếu cột subject_id');st.stop()
    df['subject_id']=df['subject_id'].astype(str)
    feats=feature_columns(df)
    if len(feats)!=68: st.warning(f'Nhận diện {len(feats)} đặc trưng số, mong đợi 68. Kiểm tra dữ liệu trước khi dự đoán.')
except Exception as exc:
    st.error(f'Lỗi đọc dữ liệu: {exc}')
    st.stop()

subject_df=df if choice=='Tất cả 15 người' else df[df.subject_id==choice]
if subject_df.empty: st.warning('Không có mẫu cho người đang chọn.');st.stop()

st.subheader('Tổng quan')
col1,col2,col3,col4=st.columns(4)
col1.metric('Cửa sổ đang xem',f'{len(subject_df):,}')
col2.metric('Số người',subject_df.subject_id.nunique())
col3.metric('Số đặc trưng',len(feats))
col4.metric('Toàn bộ cửa sổ',f'{len(df):,}')

st.markdown('### Phân bố nhãn thật của dữ liệu')
try:
    target_binary = get_target(subject_df, 'binary')
    target_multi = get_target(subject_df, 'multiclass')
    binary_counts = (target_binary.map(LABELS['binary']).value_counts()
                     .reindex(['Non-Stress', 'Stress'], fill_value=0))
    multi_counts = (target_multi.map(LABELS['multiclass']).value_counts()
                    .reindex(['Baseline', 'Stress', 'Amusement'], fill_value=0))
    col_binary, col_multi = st.columns(2, gap='large')
    with col_binary:
        st.markdown('#### Binary — Stress / Non-Stress')
        st.bar_chart(binary_counts.rename('Số cửa sổ'), use_container_width=True)
        st.caption('Non-Stress = Baseline + Amusement; Stress = Stress.')
    with col_multi:
        st.markdown('#### Multiclass — 3 trạng thái')
        st.bar_chart(multi_counts.rename('Số cửa sổ'), use_container_width=True)
        st.caption('Baseline / Stress / Amusement.')
except Exception as exc:
    st.info(f'Không đọc được phân bố nhãn: {exc}')

tabs=st.tabs([' Dữ liệu 15 người',' Đánh giá LOSO',' Dự đoán Binary + Multiclass',' Xuất báo cáo'])
with tabs[0]:
    st.subheader('Xem 68 đặc trưng theo người và cửa sổ')
    st.dataframe(subject_df, use_container_width=True, hide_index=True, height=420)
    st.download_button('Tải CSV dữ liệu đang lọc',subject_df.to_csv(index=False).encode('utf-8-sig'),'wesad_filtered.csv','text/csv')
    if len(feats)>0:
        columns=st.multiselect('Chọn đặc trưng để vẽ',feats,default=feats[:2],max_selections=5)
        if columns:
            plot_df=subject_df[columns].head(500).reset_index(drop=True)
            st.caption('Biểu đồ hiển thị tối đa 500 cửa sổ đầu của bộ lọc để đảm bảo tốc độ.')
            st.line_chart(plot_df)
    counts=df.groupby('subject_id').size().rename('Số cửa sổ').reset_index()
    st.dataframe(counts,hide_index=True)

with tabs[1]:
    st.subheader('Kết quả đánh giá LOSO (người kiểm tra không tham gia huấn luyện)')
    pp=to_absolute(root,pred_path)
    if not pp.is_file():
        st.info(f'Chưa có predictions.csv tại {pp}. Hãy chạy thí nghiệm LOSO hoặc chọn đường dẫn CSV có thật.')
    else:
        try:
            preds=read_csv(str(pp),pp.stat().st_mtime_ns).copy()
            required={'subject_id','task','y_true','y_pred'}
            if not required.issubset(preds.columns):
                st.error(f'predictions.csv thiếu cột: {sorted(required-set(preds.columns))}')
            else:
                preds['subject_id']=preds['subject_id'].astype(str)
                if choice!='Tất cả 15 người':preds=preds[preds.subject_id==choice]
                if 'variant' in preds.columns:
                    variants=sorted(preds['variant'].dropna().unique())
                    selection=st.selectbox('Cấu hình đánh giá',variants,index=len(variants)-1)
                    preds=preds[preds.variant==selection]
                for task in ('binary','multiclass'):
                    subset=preds[preds.task==task].copy()
                    st.markdown(f'**{task.upper()}**')
                    if subset.empty: st.info('Không có dữ liệu dự đoán LOSO');continue
                    labels=list(LABELS[task])
                    f1=f1_score(subset.y_true,subset.y_pred,labels=labels,average='macro',zero_division=0)
                    ba=balanced_accuracy_score(subset.y_true,subset.y_pred)
                    a,b=st.columns(2)
                    a.metric('Macro-F1',f'{f1:.4f}')
                    b.metric('Balanced Accuracy',f'{ba:.4f}')
                    matrix=confusion_matrix(subset.y_true,subset.y_pred,labels=labels)
                    cm=pd.DataFrame(matrix,index=[f'Thực: {LABELS[task][i]}' for i in labels],columns=[f'Dự đoán: {LABELS[task][i]}' for i in labels])
                    st.dataframe(cm,use_container_width=True)
                    st.dataframe(subset.head(500),use_container_width=True,hide_index=True)
                    st.download_button(f'Tải dự đoán LOSO {task}',subset.to_csv(index=False).encode('utf-8-sig'),f'loso_{task}.csv','text/csv',key=f'dl_{task}')
                if choice=='Tất cả 15 người':
                    st.caption('Chỉ số ở đây được tính gộp từ các hàng dự đoán; để báo cáo mean ± std theo người, dùng summary.csv của thí nghiệm.')
        except Exception as exc:
            st.error(f'Lỗi phân tích kết quả LOSO: {exc}')

with tabs[2]:
    st.subheader('Dự đoán đồng thời Binary và Multiclass')
    folder = to_absolute(root, model_dir)
    bp = folder / 'knn_top30_binary.joblib'
    mp = folder / 'knn_top30_multiclass.joblib'
    if not bp.is_file() or not mp.is_file():
        st.info('Chưa có đủ hai mô hình joblib. Chạy train_models.py để tạo chúng.')
        st.code('python demo_streamlit/train_models.py', language='bash')
    else:
        try:
            b, m = load_models(str(bp), str(mp), bp.stat().st_mtime_ns, mp.stat().st_mtime_ns)
            if b.get('feature_columns') != m.get('feature_columns'):
                st.error('Hai mô hình sử dụng bộ cột đầu vào khác nhau.')
            else:
                needed = b['feature_columns']
                absent = [col for col in needed if col not in subject_df.columns]
                if absent:
                    st.error(f'Thiếu {len(absent)} đặc trưng: {absent[:10]}')
                else:
                    # Giới hạn thay đổi theo số cửa sổ của bộ lọc hiện tại.
                    max_windows = len(subject_df)
                    st.caption(
                        f'Đang có {max_windows:,} cửa sổ trong bộ lọc. '
                        'Chọn số lượng cần dự đoán; có thể chạy toàn bộ dữ liệu đang chọn.'
                    )
                    length = st.slider(
                        'Số cửa sổ cần dự đoán',
                        min_value=1,
                        max_value=max_windows,
                        value=max_windows,
                        step=1,
                        key=f'demo_window_count_{choice}_{max_windows}',
                        help='Kéo thanh để chọn từ 1 đến toàn bộ số cửa sổ của người tham gia đang lọc.',
                    )
                    subset = subject_df.iloc[:int(length)].copy()
                    # Không hiển thị kết quả cũ khi bộ lọc, dữ liệu hoặc model thay đổi.
                    current_key = (
                        str(csv_file), csv_file.stat().st_mtime_ns,
                        choice, int(length),
                        str(bp), bp.stat().st_mtime_ns,
                        str(mp), mp.stat().st_mtime_ns,
                    )
                    if st.button('Chạy dự đoán cả hai', type='primary'):
                        with st.spinner(f'Đang dự đoán {len(subset):,} cửa sổ...'):
                            X = subset[needed].replace([np.inf, -np.inf], np.nan)
                            results = pd.DataFrame({
                                'subject_id': subset['subject_id'].to_numpy()
                            })
                            if 'window_id' in subset.columns:
                                results['window_id'] = subset['window_id'].to_numpy()
                            results['binary_prediction'] = b['pipeline'].predict(X)
                            results['binary_label'] = results['binary_prediction'].map(LABELS['binary'])
                            results['multiclass_prediction'] = m['pipeline'].predict(X)
                            results['multiclass_label'] = results['multiclass_prediction'].map(LABELS['multiclass'])
                            try:
                                results['binary_true'] = get_target(subset, 'binary').map(LABELS['binary']).to_numpy()
                                results['multiclass_true'] = get_target(subset, 'multiclass').map(LABELS['multiclass']).to_numpy()
                            except (ValueError, TypeError, KeyError):
                                pass  # Dữ liệu mới có thể không có nhãn thật.
                            st.session_state['demo_prediction'] = {
                                'key': current_key, 'results': results,
                            }

                    stored = st.session_state.get('demo_prediction')
                    if stored is not None and stored['key'] == current_key:
                        results = stored['results']
                        st.success(f'Đã dự đoán {len(results):,} cửa sổ với cả hai pipeline.')
                        st.markdown('#### Tổng hợp kết quả dự đoán')
                        col_binary, col_multi = st.columns(2, gap='large')

                        with col_binary:
                            st.markdown('**Binary — Stress / Non-Stress**')
                            count_b = results['binary_label'].value_counts().reindex(
                                ['Non-Stress', 'Stress'], fill_value=0
                            )
                            summary_b = pd.DataFrame({
                                'Trạng thái': count_b.index,
                                'Số cửa sổ': count_b.values,
                                'Tỷ lệ (%)': (count_b.values / len(results) * 100).round(2),
                            })
                            st.dataframe(
                                summary_b,
                                hide_index=True,
                                use_container_width=True,
                                column_config={
                                    'Tỷ lệ (%)': st.column_config.NumberColumn('Tỷ lệ (%)', format='%.2f%%'),
                                },
                            )

                        with col_multi:
                            st.markdown('**Multiclass — 3 trạng thái**')
                            count_m = results['multiclass_label'].value_counts().reindex(
                                ['Baseline', 'Stress', 'Amusement'], fill_value=0
                            )
                            summary_m = pd.DataFrame({
                                'Trạng thái': count_m.index,
                                'Số cửa sổ': count_m.values,
                                'Tỷ lệ (%)': (count_m.values / len(results) * 100).round(2),
                            })
                            st.dataframe(
                                summary_m,
                                hide_index=True,
                                use_container_width=True,
                                column_config={
                                    'Tỷ lệ (%)': st.column_config.NumberColumn('Tỷ lệ (%)', format='%.2f%%'),
                                },
                            )

                        st.caption(
                            'Tỷ lệ phần trăm được tính trên số cửa sổ đã dự đoán, '
                            'không phải độ chính xác hay xác suất mô hình.'
                        )
                        # Không hiển thị lựa chọn hoặc bảng chi tiết từng cửa sổ.
                        # File CSV vẫn giữ dự đoán từng hàng để đối chiếu khi cần.
                        st.download_button(
                            'Tải CSV kết quả Binary + Multiclass',
                            results.to_csv(index=False).encode('utf-8-sig'),
                            'demo_predictions.csv', 'text/csv',
                        )
        except Exception as exc:
            st.error(f'Không thể chạy mô hình: {exc}')

with tabs[3]:
    st.subheader('Tệp kết quả dùng cho đồ án')
    st.markdown('**Đánh giá nghiên cứu:** sử dụng `results.csv`, `summary.csv`, `predictions.csv` từ các EXP đã chạy đầy đủ LOSO. **Mô hình triển khai:** hai file `.joblib` trong `demo_streamlit/models/`.')
    st.download_button('Xuất dữ liệu đang chọn',subject_df.to_csv(index=False).encode('utf-8-sig'),f'{choice.replace(" ","_")}_68features.csv','text/csv')
    st.caption('Không hiển thị số liệu thực nghiệm giả lập; các kết quả chỉ xuất hiện khi có file từ lượt chạy thật.')
