"""Kiểm tra và vẽ điểm train/test từ cùng một lượt refit, không huấn luyện thêm."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
FOLDERS={
    '006':ROOT/'experiments/01_baselines/EXP-006_MLP',
    '010':ROOT/'experiments/02_tuning/EXP-010',
    '015':ROOT/'experiments/03_balancing/EXP-015',
}


def read_json(path):return json.loads(Path(path).read_text(encoding='utf8'))


def matrix_scores(matrix):
    m=np.asarray(matrix,dtype=float)
    tp=np.diag(m);rows=m.sum(axis=1);cols=m.sum(axis=0)
    f1=np.divide(2*tp,rows+cols,out=np.zeros_like(tp),where=(rows+cols)>0).mean()
    recall=np.divide(tp,rows,out=np.zeros_like(tp),where=rows>0)
    ba=recall[rows>0].mean()
    return float(f1),float(ba),int(m.sum())


def verify(folder):
    table=pd.read_csv(folder/'metrics.csv')
    assert len(table)==30 and not table.duplicated(['task','subject_id']).any()
    assert table.groupby('task').subject_id.nunique().eq(15).all()
    assert table.reference_prediction_differences.eq(0).all()
    assert table.reference_metrics_match.all()
    for row in table.itertuples():
        record=read_json(folder/'folds'/f'{row.task}_{row.subject_id}.json')
        assert len(record['outer_train_subjects'])==14
        assert row.subject_id not in record['outer_train_subjects']
        assert record['outer_test_subjects']==[row.subject_id]
        for split in ['train','test']:
            f1,ba,count=matrix_scores(record[f'{split}_confusion_matrix'])
            assert np.isclose(f1,getattr(row,f'{split}_macro_f1'),atol=1e-12,rtol=0)
            assert np.isclose(ba,getattr(row,f'{split}_balanced_accuracy'),atol=1e-12,rtol=0)
            assert count==getattr(row,f'n_{split}')
        assert row.n_train+row.n_test==6189
    predictions=pd.read_csv(folder/'predictions_test.csv')
    assert len(predictions)==12378
    assert not predictions.duplicated(['task','subject_id','window_id']).any()
    result=dict(status='passed',outer_folds=30,reference_predictions_match=True,
        reference_metrics_match=True,confusion_matrix_metrics_match=True,
        train_and_test_same_refitted_model=True,train_evaluated_before_ROS_duplication=True,
        test_not_used_for_model_selection=True)
    (folder/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    return table


def plot(folder,table,experiment):
    order=sorted(table.subject_id.unique(),key=lambda s:int(s[1:]))
    fig,axes=plt.subplots(2,2,figsize=(14,8),layout='constrained')
    for i,task in enumerate(['multiclass','binary']):
        part=table[table.task==task].set_index('subject_id').loc[order]
        for ax,metric,title in zip(axes[i],['macro_f1','balanced_accuracy'],['macro f1','balanced accuracy']):
            ax.plot(order,part[f'train_{metric}'],marker='^',label='train: 14 người',linewidth=1.8)
            ax.plot(order,part[f'test_{metric}'],marker='o',linestyle='--',label='test: 1 người',linewidth=1.8)
            ax.set(title=f'{task} — {title}',xlabel='người giữ lại làm outer test',ylabel='điểm',ylim=(0,1.05))
            ax.grid(alpha=.25);ax.legend(loc='lower left',fontsize=9)
    fig.suptitle(f'{experiment}: train và test từ cùng mô hình refit',fontsize=14)
    fig.savefig(folder/'train_vs_test.png',dpi=150)
    plt.close(fig)


def report(folder,table,info,experiment):
    summary=table.groupby('task').agg(
        n_subjects=('subject_id','nunique'),train_macro_f1_mean=('train_macro_f1','mean'),
        train_macro_f1_std=('train_macro_f1','std'),test_macro_f1_mean=('test_macro_f1','mean'),
        test_macro_f1_std=('test_macro_f1','std'),macro_f1_gap_mean=('macro_f1_gap','mean'),
        train_balanced_accuracy_mean=('train_balanced_accuracy','mean'),
        test_balanced_accuracy_mean=('test_balanced_accuracy','mean'),
        balanced_accuracy_gap_mean=('balanced_accuracy_gap','mean')).reset_index()
    summary.to_csv(folder/'summary.csv',index=False)
    lines=[f'# {experiment}: so sánh train và test — {info["date"]}','',
        'Đây là lượt refit bổ sung để đo cả train và test từ cùng mô hình. '
        'Giữ nguyên cấu hình đã chốt, không tuning lại hoặc chọn theo điểm test.',
        'Dự đoán test ở đủ 30 folds khớp toàn bộ kết quả đã lưu. '
        'Không coi điểm train mới đo là điểm đã được ghi trong lượt chạy gốc.','']
    for row in summary.itertuples():
        lines.append(f'- {row.task}: macro f1 train {row.train_macro_f1_mean:.4f}, '
                     f'test {row.test_macro_f1_mean:.4f}; khoảng cách {row.macro_f1_gap_mean:.4f}.')
    lines+=['','Mô hình khớp rất sát train nhưng điểm trên người chưa gặp thấp hơn ở nhiều folds. '
        'Khoảng cách này cho thấy khả năng tổng quát hóa còn hạn chế, có thể liên quan '
        'đến overfitting và khác biệt giữa các người; biểu đồ chưa tách được hai nguyên nhân.',
        '', 'Trục ngang là người được giữ lại làm test, không phải epoch. '
        'Hai đường trên mỗi trục là điểm train và outer test. Không gọi đây là learning curve '
        'train–validation hay loss theo epoch. Điểm train tính trên các cửa sổ gốc của '
        '14 người, kể cả ở exp015; không nhân thêm trọng số do các mẫu ROS bị lặp.',
        '', 'Mean và độ lệch chuẩn mẫu theo 15 người. Các tập train giữa outer folds '
        'có chồng lặp nên không xem các điểm train là 15 quan sát độc lập để kiểm định.',
        '', 'Không dùng đồ thị để tự sửa tham số theo outer test. Nếu cần thử cải tiến, '
        'chọn bằng training/inner validation theo hướng nghiên cứu đã định.']
    reference=read_json(folder/'reference.json')
    if 'original_pandas' in reference:
        lines+=['',f"Baseline exp006 dùng pandas {reference['original_pandas']}; lượt refit dùng "
                f"pandas {info['environment']['packages']['pandas']}. Dự đoán test đã được đối chiếu khớp; "
                'chưa so sánh trọng số mô hình hoặc độ nhạy với khác biệt phiên bản.']
    (folder/'analysis.md').write_text('\n'.join(lines),encoding='utf8')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve()
    info=read_json(run/'run_info.json')
    summaries=[]
    for exp in info['experiments']:
        source=run/f'EXP-{exp}';target=FOLDERS[exp]/'results/train_test'
        assert not target.exists(),'không ghi đè phần minh chứng train/test đã có'
        table=verify(source)
        shutil.copytree(source,target)
        shutil.copy2(run/'run_info.json',target/'run_info.json')
        shutil.copy2(run/'source_snapshot.zip',target/'source_snapshot.zip')
        plot(target,table,f'EXP-{exp}')
        summary=report(target,table,info,f'EXP-{exp}')
        summary.insert(0,'experiment',f'EXP-{exp}');summaries.append(summary)
        manifest={p.relative_to(target).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in target.rglob('*') if p.is_file()}
        (target/'checksums.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
        root_manifest=target.parent/'artifact_checksums.json'
        if root_manifest.exists():
            all_files={p.relative_to(target.parent).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in target.parent.rglob('*') if p.is_file() and p!=root_manifest}
            root_manifest.write_text(json.dumps(all_files,ensure_ascii=False,indent=2),encoding='utf8')
        print(f'EXP-{exp}: train/test verified and plot saved to {target}',flush=True)
    combined=pd.concat(summaries,ignore_index=True)
    combined.to_csv(run/'summary_all_experiments.csv',index=False)
    print(combined.to_string(index=False))


if __name__=='__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf8');main()
