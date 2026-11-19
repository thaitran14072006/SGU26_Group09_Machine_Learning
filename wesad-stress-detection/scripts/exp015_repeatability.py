"""Đối chiếu ba lượt train EXP-015 độc lập; cùng seed kiểm tra khả năng tái lập."""
from pathlib import Path
import argparse
import json
import hashlib
import shutil
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'experiments/03_balancing/EXP-015/results'


def read_json(path):return json.loads(Path(path).read_text(encoding='utf8'))


def collect(repeat_runs):
    assert len(repeat_runs)==2,'cần 2 lượt mới, cộng lượt gốc thành 3'
    assert not (PACKAGE/'repeatability').exists(),'không ghi đè lịch sử đối chiếu'
    sources=[PACKAGE]+[p.resolve() for p in repeat_runs]
    assert len(set(sources))==3
    base_config=read_json(PACKAGE/'config.json')
    base_params=read_json(PACKAGE/'best_params.json')
    keys=['task','model','subject_id','window_id','sensor_set','normalization']
    metric_keys=['task','model','subject_id']
    pred0=pd.read_csv(PACKAGE/'predictions.csv',dtype={'window_id':str,'subject_id':str})
    metric0=pd.read_csv(PACKAGE/'metrics_by_subject.csv')
    records=[];scores=[];details=[]
    for index,source in enumerate(sources,1):
        verification=read_json(source/'verification.json')
        assert verification['status']=='passed' and verification['outer_folds']==30
        config=read_json(source/'config.json')
        assert config['data_sha256']==base_config['data_sha256']
        assert config['environment']==base_config['environment']
        assert config['code_sha256']==base_config['code_sha256']
        assert config['settings']['seed']==42
        assert read_json(source/'best_params.json')==base_params
        pred=pd.read_csv(source/'predictions.csv',dtype={'window_id':str,'subject_id':str})
        pair=pred0.merge(pred,on=keys,how='outer',validate='one_to_one',indicator=True,suffixes=('_base','_repeat'))
        assert len(pair)==12378 and pair['_merge'].eq('both').all()
        assert pair.y_true_base.eq(pair.y_true_repeat).all()
        n_diff=int(pair.y_pred_base.ne(pair.y_pred_repeat).sum())
        metric=pd.read_csv(source/'metrics_by_subject.csv')
        merged=metric0.merge(metric,on=metric_keys,validate='one_to_one',suffixes=('_base','_repeat'))
        assert len(merged)==30 and merged.n_test_base.eq(merged.n_test_repeat).all()
        max_delta=max(float(abs(merged[m+'_base']-merged[m+'_repeat']).max()) for m in ['macro_f1','balanced_accuracy'])
        provenance=read_json(source/'notebook_provenance.json')
        run_id=f'EXP-015-R{index:02}'
        records.append(dict(exp_chung='EXP-015',run_id=run_id,member='như',date=provenance['date'],
            seed=42,outer_folds=30,predictions=len(pred),prediction_differences=n_diff,
            max_metric_difference=max_delta,predictions_match=n_diff==0,
            metrics_match=max_delta<1e-12,source_run=source.relative_to(ROOT).as_posix()))
        for task,part in metric.groupby('task'):
            scores.append(dict(exp_chung='EXP-015',run_id=run_id,date=provenance['date'],task=task,
                n_subjects=len(part),macro_f1_mean=float(part.macro_f1.mean()),
                macro_f1_std=float(part.macro_f1.std(ddof=1)),
                balanced_accuracy_mean=float(part.balanced_accuracy.mean()),
                balanced_accuracy_std=float(part.balanced_accuracy.std(ddof=1))))
        merged.insert(0,'run_id',run_id);details.append(merged)
    destination=PACKAGE/'repeatability';destination.mkdir()
    table=pd.DataFrame(records);score_table=pd.DataFrame(scores)
    table.to_csv(destination/'runs.csv',index=False)
    score_table.to_csv(destination/'scores.csv',index=False)
    pd.concat(details,ignore_index=True).to_csv(destination/'metrics_comparison.csv',index=False)
    # Lượt gốc đã nằm trong PACKAGE; hai lượt mới giữ đủ checkpoint để kiểm tra.
    for index,source in enumerate(sources[1:],2):
        target=destination/f'R{index:02}'
        shutil.copytree(source,target)
    verification=dict(status='passed',exp_chung='EXP-015',independent_training_runs=3,
        total_outer_fits=90,predictions_per_run=12378,seed=42,
        all_predictions_match=bool(table.predictions_match.all()),
        all_metrics_match=bool(table.metrics_match.all()),
        interpretation='same-seed reproducibility; not stability across different seeds')
    (destination/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf8')
    note='''# kiểm tra chạy lại exp015 — 18/11/2026

Có ba lượt train độc lập: lượt gốc ngày 16/11 và hai lượt mới ngày 18/11.
Mỗi lượt có 30 outer folds, tổng 90 lần fit MLP. Không dùng resume để tính thành lượt mới.
Cùng dữ liệu, môi trường, best_params từ exp010 và seed 42.

Đối chiếu 12.378 dự đoán và 30 bộ điểm mỗi lượt. Xem runs.csv và scores.csv.
Đây là kiểm tra khả năng chạy lại với cùng seed; chưa kiểm tra biến thiên giữa các seed.
Kết quả khớp không có nghĩa mô hình đã tổng quát hóa tốt hơn hoặc có ý nghĩa thống kê.
Mã EXP chung của cả ba lượt là EXP-015.
'''
    (destination/'README.md').write_text(note,encoding='utf8')
    manifest={p.relative_to(PACKAGE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
              for p in PACKAGE.rglob('*') if p.is_file() and p!=PACKAGE/'artifact_checksums.json'}
    (PACKAGE/'artifact_checksums.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print(table.to_string(index=False))
    print(json.dumps(verification,ensure_ascii=False))
    return table,score_table


if __name__=='__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs=2,type=Path,required=True)
    args=parser.parse_args();collect(args.runs)
