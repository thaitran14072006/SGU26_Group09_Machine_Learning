"""EXP-017: KNN with fold-local feature selection versus the unchanged 68-feature baseline."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from wesad.data import load_feature_table  # noqa: E402


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def make_pipeline(k=None):
    steps = [('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
             ('scaler', StandardScaler())]
    if k is not None:
        steps.append(('selection', SelectKBest(score_func=f_classif, k=k)))
    steps.append(('model', KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='minkowski', p=2)))
    return Pipeline(steps)


def draw_plots(records, predictions, out, variants):
    images = out / 'figures'
    images.mkdir(parents=True, exist_ok=True)
    for task, labels in [('binary', [0,1]), ('multiclass', [1,2,3])]:
        names = ['Không stress', 'Stress'] if task == 'binary' else ['Baseline','Stress','Amusement']
        for variant in variants:
            df = predictions[(predictions.task == task) & (predictions.variant == variant)]
            cm = confusion_matrix(df.y_true, df.y_pred, labels=labels)
            fig, ax = plt.subplots(figsize=(6,5))
            ax.imshow(cm, cmap='Blues')
            ax.set_xticks(range(len(labels)), names)
            ax.set_yticks(range(len(labels)), names)
            ax.set(xlabel='Dự đoán', ylabel='Thực tế', title=f'{task} — {variant}')
            for (i,j), value in np.ndenumerate(cm):
                ax.text(j,i,str(value),ha='center',va='center')
            fig.tight_layout()
            fig.savefig(images / f'confusion_matrix_{task}_{variant}.png', dpi=160)
            plt.close(fig)
        piv = records[records.task == task].pivot(index='subject_id', columns='variant', values='macro_f1')
        piv = piv.reindex(sorted(piv.index, key=lambda s:int(s[1:])))
        ax=piv.plot.bar(figsize=(11,4.5),rot=0)
        ax.set(ylim=(0,1),xlabel='Người tham gia',ylabel='Macro-F1',title=f'LOSO — {task}')
        ax.figure.tight_layout()
        ax.figure.savefig(images / f'macro_f1_by_subject_{task}.png',dpi=160)
        plt.close(ax.figure)
    scores=records.groupby(['task','variant']).macro_f1.mean().unstack()
    ax=scores.plot.bar(figsize=(7,4.5),rot=0)
    ax.set(ylim=(0,1),ylabel='Macro-F1 trung bình',title='So sánh KNN 68 và chọn đặc trưng')
    ax.figure.tight_layout()
    ax.figure.savefig(images / 'comparison_feature_selection.png',dpi=160)
    plt.close(ax.figure)


def run(config_path, quick_subject=None):
    config_path = Path(config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    data, schema = ROOT / cfg['features'], ROOT / cfg['schema']
    if not data.is_file() or not schema.is_file():
        raise FileNotFoundError(f'Cần tải bộ G2 physiological68 theo README: {data} và {schema}')
    if digest(data) != cfg['feature_sha256']:
        raise ValueError('SHA256 của dữ liệu G2 khác cấu hình nhóm; không chạy trên dữ liệu khác mà không ghi nhận.')
    df, features = load_feature_table(data, schema)
    subjects = set(df.subject_id.astype(str))
    if len(features) != 68 or subjects != set(cfg['expected_subjects']) or len(subjects) != 15:
        raise ValueError('EXP-017 yêu cầu đúng 68 features và đủ 15 người.')
    k = int(cfg['selected_features'])
    if not (1 <= k < 68):
        raise ValueError('selected_features phải từ 1 đến 67.')
    if quick_subject is not None and quick_subject not in subjects:
        raise ValueError(f'Không tìm thấy người thử {quick_subject}.')
    variants = ['knn_68',f'knn_top{k}']
    out = ROOT / cfg['output_dir'] / f'quick_{quick_subject}' if quick_subject else config_path.parent
    out.mkdir(parents=True,exist_ok=True)
    identity = {'data_sha256':digest(data),'schema_sha256':digest(schema),
                'config_sha256':digest(config_path),'runner_sha256':digest(Path(__file__)),
                'scope':quick_subject or 'main','features':68,'selected_features':k,
                'method':'LOSO; selector fitted only to outer-training subjects; no hyperparameter tuning'}
    manifest=out/('provenance.json' if quick_subject else 'result_provenance.json')
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))!=identity:
        raise ValueError('Đầu ra cũ không khớp phiên bản code/config/data. Hãy đổi output_dir thay vì ghi đè.')
    manifest.write_text(json.dumps(identity,indent=2,ensure_ascii=False),encoding='utf-8')
    X=df[features].astype(float); groups=df.subject_id.astype(str)
    records=[]; predictions=[]; selections=[]
    for task,labels in [('binary',[0,1]),('multiclass',[1,2,3])]:
        y=((df.label_original==2).astype(int) if task=='binary' else df.label_original.astype(int))
        for train,test in LeaveOneGroupOut().split(X,y,groups):
            subject=str(groups.iloc[test].iloc[0])
            if quick_subject and subject!=quick_subject: continue
            assert subject not in set(groups.iloc[train])
            for variant,k_choice in [(variants[0],None),(variants[1],k)]:
                estimator=make_pipeline(k_choice)
                estimator.fit(X.iloc[train],y.iloc[train])
                pred=estimator.predict(X.iloc[test])
                metrics={'task':task,'variant':variant,'subject_id':subject,'n_test':len(test),
                         'macro_f1':float(f1_score(y.iloc[test],pred,labels=labels,average='macro',zero_division=0)),
                         'balanced_accuracy':float(balanced_accuracy_score(y.iloc[test],pred))}
                records.append(metrics)
                out_pred=df.iloc[test][['subject_id','window_id']].copy()
                out_pred['task']=task;out_pred['variant']=variant
                out_pred['y_true']=y.iloc[test].to_numpy();out_pred['y_pred']=pred
                predictions.append(out_pred)
                if k_choice:
                    selector=estimator.named_steps['selection']
                    for i in selector.get_support(indices=True):
                        selections.append({'task':task,'subject_id':subject,'feature':features[i],
                                           'feature_index':int(i),'anova_f_score':float(selector.scores_[i])})
                print(f'{task} / {subject} / {variant}: F1={metrics["macro_f1"]:.4f}',flush=True)
    results=pd.DataFrame(records); pred_df=pd.concat(predictions,ignore_index=True)
    if quick_subject is None and (len(results)!=60 or len(pred_df)!=4*len(df)):
        raise AssertionError('Chạy chính phải đủ 60 fold và 4 dự đoán trên mỗi cửa sổ.')
    results.to_csv(out/'results.csv',index=False)
    pred_df.to_csv(out/'predictions.csv',index=False)
    sel_df = pd.DataFrame(selections)
    sel_df.to_csv(out/'selected_features_by_fold.csv',index=False)
    if not sel_df.empty and quick_subject is None:
        freq = sel_df.groupby(['task','feature'],as_index=False).agg(n_folds_selected=('subject_id','nunique'),mean_anova_f_score=('anova_f_score','mean'))
        freq.sort_values(['task','n_folds_selected','feature'],ascending=[True,False,True]).to_csv(out/'feature_selection_frequency.csv',index=False)
    summary=results.groupby(['task','variant'],as_index=False).agg(
        n_subjects=('subject_id','nunique'),n_test=('n_test','sum'),
        macro_f1_mean=('macro_f1','mean'),macro_f1_std=('macro_f1','std'),
        balanced_accuracy_mean=('balanced_accuracy','mean'),
        balanced_accuracy_std=('balanced_accuracy','std'))
    summary.to_csv(out/'summary.csv',index=False)
    if quick_subject is None:
        base=summary[summary.variant==variants[0]].set_index('task')
        selected=summary[summary.variant==variants[1]].set_index('task')
        delta=pd.DataFrame({'task':base.index,
                            'macro_f1_delta':selected.macro_f1_mean-base.macro_f1_mean,
                            'balanced_accuracy_delta':selected.balanced_accuracy_mean-base.balanced_accuracy_mean})
        delta.to_csv(out/'comparison_delta.csv',index=False)
        delta.to_csv(out/'comparison_exp003_internal.csv',index=False)
        draw_plots(results,pred_df,out,variants)
        (out/'RUN_COMPLETE.txt').write_text('Hoàn thành 60 fold: 15 người x 2 bài toán x 2 biến thể\n',encoding='utf-8')
    print('\nĐã ghi kết quả tại',out)
    print(summary.to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='experiments/04_improvements/EXP-017_KNN_Feature_Selection/config.yaml')
    parser.add_argument('--quick-subject',default=None)
    args=parser.parse_args()
    run(args.config,args.quick_subject)
