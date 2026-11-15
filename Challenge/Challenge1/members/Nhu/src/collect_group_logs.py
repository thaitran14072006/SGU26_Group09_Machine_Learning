"""Gom số đo đã có; giữ nguyên dòng cũ, không chạy mô hình thay thành viên khác."""
from pathlib import Path
import csv
import io
import json
import hashlib
import math
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
LOG=ROOT/'reports/individual/nhat_ky_thuc_nghiem.csv'
MEMBER=ROOT/'members/Nhu'


def collect():
    run=ROOT/(MEMBER/'outputs/latest_run.txt').read_text(encoding='utf8')
    comparison=pd.read_csv(run/'comparison.csv')
    comparison.to_csv(MEMBER/'outputs/experiment_log.csv',index=False)
    original=LOG.read_bytes()
    reader=csv.DictReader(io.StringIO(original.decode('utf-8-sig')))
    fields=reader.fieldnames
    old=list(reader)
    existing={r['Mã thực nghiệm'] for r in old}
    additions=[]
    info=json.loads((run/'run_info.json').read_text(encoding='utf8'))
    for row in comparison.itertuples():
        if row.exp_id in existing:continue
        params=dict(n_estimators=100,max_depth=None if pd.isna(row.max_depth) else int(row.max_depth),
                    min_samples_leaf=row.min_samples_leaf,random_state=7)
        delta=row.cv_rmse_mean-comparison.iloc[0].cv_rmse_mean
        record=dict(zip(fields,['']*len(fields)))
        record.update({'Ngày thực hiện':info['date'],'Mã thực nghiệm':row.exp_id,
            'Thành viên':'như','Mô hình':'random forest','Tiền xử lý':row.mode,
            'Tham số':json.dumps(params),'Giả thuyết':row.hypothesis,
            'CV RMSE log TB':row.cv_rmse_mean,'CV RMSE log ĐLC':row.cv_rmse_std,
            'CV MAE giá':row.cv_mae_price,'Thời gian (giây)':row.seconds,
            'Kết luận':f'chênh lệch baseline {delta:+.5f}; sd mẫu ddof=1; chọn bằng cv',
            'File minh chứng':'members/Nhu/notebooks/15_11_2026_rf.ipynb',
            'File kết quả':(run/f'{row.exp_id}_fold_scores.csv').relative_to(ROOT).as_posix(),
            'SHA256 train':info['train_sha256'],'Trạng thái':'đã chạy và kiểm tra'})
        additions.append(record)
    nam=pd.read_csv(ROOT/'members/Nam/outputs/experiment_log.csv')
    for idx,row in nam.iterrows():
        exp=f'Nam_{row["version"]}_{idx+1:02}'
        if exp in existing:continue
        record=dict(zip(fields,['']*len(fields)))
        record.update({'Ngày thực hiện':pd.Timestamp(row.timestamp).strftime('%d/%m/%Y'),
            'Mã thực nghiệm':exp,'Thành viên':'Nam','Mô hình':'SVR',
            'Tiền xử lý':str(row.variant),'Tham số':str(row.params),
            'Giả thuyết':str(row.notes),'CV RMSE log TB':row.cv_rmse_mean,
            'CV RMSE log ĐLC':row.cv_rmse_std,'Thời gian (giây)':row.seconds,
            'Kết luận':'nhập từ nhật ký Nam; sd nguồn ddof=0; chưa train lại',
            'File minh chứng':'members/Nam/outputs/experiment_log.csv',
            'File kết quả':'members/Nam/outputs/selected_configs.json',
            'Trạng thái':'đã tổng hợp từ nguồn'})
        additions.append(record)
    if additions:
        buffer=io.StringIO(newline='')
        writer=csv.DictWriter(buffer,fieldnames=fields,lineterminator='\n')
        writer.writerows(additions)
        suffix=('' if original.endswith(b'\n') else '\n')+buffer.getvalue()
        with LOG.open('ab') as stream:stream.write(suffix.encode('utf8'))
        assert LOG.read_bytes().startswith(original)
    summary=[]
    best=json.loads((run/'selected_config.json').read_text(encoding='utf8'))
    report=json.loads((run/'holdout_report.json').read_text(encoding='utf8'))
    row=comparison.loc[comparison.exp_id==best['exp_id']].iloc[0]
    summary.append(dict(member='như',model='rf',selected=best['exp_id'],cv_rmse_log=row.cv_rmse_mean,
        cv_sd_sample=row.cv_rmse_std,holdout_rmse_log=report['rmse_log'],holdout_mae=report['mae_price'],
        source=(run/'holdout_report.json').relative_to(ROOT).as_posix(),notes='đã chạy và kiểm tra model nạp lại'))
    thai=json.loads((ROOT/'members/Thai/outputs/experiments/final_summary.json').read_text(encoding='utf8'))
    tc=pd.read_csv(ROOT/'members/Thai/outputs/experiments/comparison.csv')
    t=tc.loc[tc.exp_id==thai['selected_exp']].iloc[0]
    summary.append(dict(member='Thái',model=t.model,selected=t.exp_id,cv_rmse_log=t.cv_rmse_log,
        cv_sd_sample=t.cv_std,holdout_rmse_log=thai['holdout_rmse_log'],holdout_mae=thai['holdout_mae_price'],
        source='members/Thai/outputs/experiments/final_summary.json',
        notes='số đo nguồn; hash byte train trong nhật ký khác bản hiện tại, cần xác minh snapshot'))
    nam_report=json.loads((ROOT/'members/Nam/outputs/v5_holdout_report.json').read_text(encoding='utf8'))
    nc=json.loads((ROOT/'members/Nam/outputs/selected_configs.json').read_text(encoding='utf8'))
    nm=nc[nam_report['selected_version']]['metrics']
    summary.append(dict(member='Nam',model='svr',selected=nam_report['selected_version'],
        cv_rmse_log=nam_report['development_cv_rmse'],cv_sd_sample=nm['cv_rmse_std']*math.sqrt(5/4),
        holdout_rmse_log=nam_report['holdout_rmse_log'],holdout_mae=nam_report['holdout_mae_price'],
        source='members/Nam/outputs/v5_holdout_report.json',notes='số đo nguồn; sd chuyển ddof=0 sang ddof=1'))
    mai=[json.loads(p.read_text(encoding='utf8')) for p in (ROOT/'members/Mai/outputs/experiments').glob('*/result.json')]
    m=min(mai,key=lambda r:r['cv_mean_rmse_log'])
    summary.append(dict(member='Mai',model='gradient boosting',selected=m['run_id'],cv_rmse_log=m['cv_mean_rmse_log'],
        cv_sd_sample=pd.Series(m['cv_fold_rmse_log']).std(ddof=1),holdout_rmse_log=m['holdout_rmse_log'],
        holdout_mae=m['holdout_mae_price'],source=f'members/Mai/outputs/experiments/{m["run_id"]}/result.json',
        notes='số đo nguồn; sd tính lại ddof=1 từ 5 fold'))
    table=pd.DataFrame(summary)
    table.to_csv(ROOT/'results/group_model_comparison.csv',index=False)
    note='''# tổng hợp kết quả challenge — 15/11/2026

Như tổng hợp từ kết quả hiện có, không chạy lại mô hình của các bạn khác.
Bảng số đo ở results/group_model_comparison.csv, mỗi dòng có đường dẫn nguồn.
Điểm càng thấp càng tốt. Độ lệch chuẩn trong bảng so sánh đã quy về mẫu ddof=1;
nhật ký gốc giữ nguyên cách tính và ngày của người tạo.

## kiểm tra và giới hạn

- Dữ liệu train/test hiện tại khớp nội dung CSV trong ZIP người dùng gửi.
- Danh sách development/holdout và cv_fold khớp KFold seed 7 theo đúng thứ tự Id.
- RF của Như đã kiểm tra 30 fold, 1.459 Id và dự đoán từ pipeline nạp lại.
- Đây không phải chứng nhận đã train lại cả bốn mô hình trên cùng môi trường.
- Hash byte train ở nhật ký Thái khác file hiện tại; chưa có snapshot dữ liệu
  của lượt đó để xác nhận khác do xuống dòng hay khác nội dung. Chưa chốt xếp hạng.
- Các phiên bản sklearn và độ lớn không gian tìm kiếm giữa các bạn còn khác nhau.
- Không lấy holdout để chọn trọng số ensemble. Mai dùng OOF development nếu kết hợp.
- Chưa nộp các submission lên Kaggle và chưa có điểm leaderboard.
'''
    (ROOT/'reports/group_comparison_notes.md').write_text(note,encoding='utf8')
    print('added log rows:',len(additions),'| group models:',len(table))
    return additions


if __name__=='__main__':collect()
