"""Huấn luyện hai pipelines triển khai trên 15 người; KHÔNG dùng train score làm đánh giá LOSO."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from common import root_dir, load_table, feature_columns, get_target, SUBJECTS, DEFAULT_DATA


def make_pipeline(k):
    return Pipeline([
        ('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),
        ('scaler',StandardScaler()),
        ('selection',SelectKBest(score_func=f_classif,k=k)),
        ('knn',KNeighborsClassifier(n_neighbors=5,weights='uniform',metric='minkowski',p=2)),
    ])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default=DEFAULT_DATA)
    parser.add_argument('--out',default='demo_streamlit/models')
    parser.add_argument('--top-k',type=int,default=30)
    args=parser.parse_args()
    root=root_dir()
    data_path=Path(args.data)
    if not data_path.is_absolute(): data_path=root/data_path
    out=Path(args.out)
    if not out.is_absolute(): out=root/out
    df=load_table(data_path)
    if 'subject_id' not in df: raise ValueError('Thiếu subject_id')
    subjects=set(df.subject_id.astype(str).unique())
    if subjects!=set(SUBJECTS): raise ValueError(f'Bộ người tham gia không đúng 15 người yêu cầu: {sorted(subjects)}')
    feats=feature_columns(df)
    if len(feats)!=68: raise ValueError(f'Cần đúng 68 đặc trưng; nhận được {len(feats)}. Kiểm tra cột dữ liệu: {feats}')
    if not 1<=args.top_k<=68: raise ValueError('--top-k phải từ 1 đến 68')
    X=df[feats].replace([np.inf,-np.inf],np.nan)
    out.mkdir(parents=True,exist_ok=True)
    for task in ('binary','multiclass'):
        y=get_target(df,task)
        expected={0,1} if task=='binary' else {1,2,3}
        if set(y.unique())!=expected: raise ValueError(f'{task}: nhãn không hợp lệ {set(y.unique())}')
        pipeline=make_pipeline(args.top_k)
        pipeline.fit(X,y)
        obj={'pipeline':pipeline,'feature_columns':feats,'task':task,'trained_subjects':sorted(subjects),'n_rows':len(df)}
        path=out/f'knn_top{args.top_k}_{task}.joblib'
        joblib.dump(obj,path)
        print(f'Đã lưu {path}')
    (out/'training_info.json').write_text(json.dumps({'data':str(data_path),'subjects':sorted(subjects),'n_rows':len(df),'n_features':len(feats),'top_k':args.top_k,'trained_utc':datetime.now(timezone.utc).isoformat(),'caution':'Models trained on all 15 subjects. Use separate LOSO predictions for unbiased evaluation.'},indent=2,ensure_ascii=False),encoding='utf-8')
    print('Hoàn tất. Không dùng dự đoán trên dữ liệu train để báo cáo LOSO.')

if __name__=='__main__': main()
