"""Random Forest House Prices: chạy CV theo split chung, chốt trước holdout."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import time
import platform
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold

MEMBER = Path(__file__).resolve().parents[1]
ROOT = MEMBER.parents[1]
OUT = MEMBER / 'outputs'
SEED = 7
TRIALS = [
    ('Nhu_EXP-001', 'baseline', None, 1, 'mốc rf với điền thiếu trung vị và one-hot'),
    ('Nhu_EXP-002', 'semantic', None, 1, 'phân biệt không có tiện ích với giá trị chưa biết'),
    ('Nhu_EXP-003', 'engineered', None, 1, 'thêm diện tích, tuổi nhà và tiện ích so với exp002'),
    ('Nhu_EXP-004', 'engineered', 12, 1, 'giới hạn độ sâu so với exp003'),
    ('Nhu_EXP-005', 'engineered', None, 3, 'tăng kích thước lá so với exp003'),
    ('Nhu_EXP-006', 'engineered', 12, 3, 'kết hợp độ sâu và lá; so với exp004 và exp005'),
]
ABSENT_COLUMNS = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
    'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType',
    'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data():
    train = pd.read_csv(ROOT / 'data/raw/train.csv')
    test = pd.read_csv(ROOT / 'data/raw/test.csv')
    assert train.shape == (1460, 81) and test.shape == (1459, 80)
    assert train.Id.is_unique and test.Id.is_unique
    assert train.SalePrice.notna().all() and (train.SalePrice > 0).all()
    assert set(train.columns) - {'SalePrice'} == set(test.columns)
    dev_ids = pd.read_csv(ROOT / 'data/splits/development_ids.csv').Id.tolist()
    hold_ids = pd.read_csv(ROOT / 'data/splits/holdout_ids.csv').Id.tolist()
    assert len(dev_ids) == len(set(dev_ids)) == 1168
    assert len(hold_ids) == len(set(hold_ids)) == 292
    assert not set(dev_ids) & set(hold_ids)
    assert set(dev_ids) | set(hold_ids) == set(train.Id)
    dev = train.set_index('Id', drop=False).loc[dev_ids].reset_index(drop=True)
    hold = train.set_index('Id', drop=False).loc[hold_ids].reset_index(drop=True)
    shared = pd.read_csv(ROOT / 'data/splits/split_seed7.csv').set_index('Id')
    folds = list(KFold(5, shuffle=True, random_state=SEED).split(dev))
    for fold, (_, valid) in enumerate(folds):
        assert (shared.loc[dev.iloc[valid].Id, 'cv_fold'] == fold).all()
    return train, test, dev, hold, folds


class HouseFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, mode='baseline'):
        self.mode = mode

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        x = X.copy()
        if self.mode == 'baseline':
            return x
        # NA ở các cột này có nghĩa không có tiện ích theo data_description.
        x[ABSENT_COLUMNS] = x[ABSENT_COLUMNS].fillna('absent')
        # Mã loại nhà là phân loại, không phải đại lượng liên tục.
        x['MSSubClass'] = x['MSSubClass'].astype(str)
        if self.mode == 'engineered':
            # Chỉ tính trên một hàng; các giá trị thiếu còn lại do imputer học từ train.
            x['TotalArea'] = x['TotalBsmtSF'] + x['1stFlrSF'] + x['2ndFlrSF']
            x['HouseAge'] = (x['YrSold'] - x['YearBuilt']).clip(lower=0)
            x['RemodelAge'] = (x['YrSold'] - x['YearRemodAdd']).clip(lower=0)
            x['TotalBath'] = (x['FullBath'] + .5*x['HalfBath']
                              + x['BsmtFullBath'] + .5*x['BsmtHalfBath'])
            x['HasGarage'] = (x['GarageArea'].fillna(0) > 0).astype(int)
            x['HasBasement'] = (x['TotalBsmtSF'].fillna(0) > 0).astype(int)
            x['HasFireplace'] = (x['Fireplaces'].fillna(0) > 0).astype(int)
            porch = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
            x['TotalPorchArea'] = x[porch].sum(axis=1, min_count=len(porch))
        return x


def pipeline(mode, depth=None, leaf=1):
    numeric = Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True))])
    categorical = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocess = ColumnTransformer([
        ('num', numeric, make_column_selector(dtype_include=np.number)),
        ('cat', categorical, make_column_selector(dtype_exclude=np.number))])
    return Pipeline([
        ('features', HouseFeatures(mode)), ('preprocess', preprocess),
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=depth,
            min_samples_leaf=leaf, random_state=SEED, n_jobs=2))])


def feature_importance(fitted):
    prep = fitted.named_steps['preprocess']
    names = []
    for name, transformer, columns in prep.transformers_:
        if name == 'num':
            names.extend(columns)
        elif name == 'cat':
            categories = transformer.named_steps['encoder'].categories_
            for column, values in zip(columns, categories):
                names.extend([column]*len(values))
    values = fitted.named_steps['rf'].feature_importances_
    assert len(names) == len(values)
    return pd.DataFrame({'feature': names, 'importance': values}).groupby('feature').importance.sum()


def setup():
    now = datetime.now(ZoneInfo('Asia/Ho_Chi_Minh'))
    run = OUT / 'experiments' / now.strftime('%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True)
    for sub in ['figures', 'predictions', 'models']:
        (run/sub).mkdir()
    train, test, dev, hold, folds = load_data()
    info = dict(date=now.strftime('%d/%m/%Y'), member='như', seed=SEED,
        python=platform.python_version(), sklearn=sklearn.__version__,
        pandas=pd.__version__, numpy=np.__version__, target='natural log(SalePrice)',
        train_sha256=sha(ROOT/'data/raw/train.csv'),
        test_sha256=sha(ROOT/'data/raw/test.csv'),
        split_sha256=sha(ROOT/'data/splits/split_seed7.csv'),
        source_sha256=sha(__file__), development_ids=dev.Id.tolist(),
        holdout_ids=hold.Id.tolist(), validation='5-fold CV trên development; seed=7',
        trials=[dict(exp_id=i, mode=m, max_depth=d, min_samples_leaf=l, hypothesis=h)
                for i,m,d,l,h in TRIALS])
    write_json(run/'run_info.json', info)
    print('ngày chạy:', info['date'])
    print('train:', train.shape, '| test:', test.shape)
    print('development:', len(dev), '| holdout:', len(hold), '| features gốc: 79')
    print('python:', info['python'], '| sklearn:', info['sklearn'])
    print('kết quả:', run)
    return run, train, test, dev, hold, folds


def eda(dev, run):
    missing = dev.drop(columns='SalePrice').isna().mean().sort_values(ascending=False)
    missing.rename('missing_fraction').to_csv(run/'missingness.csv')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    axes[0,0].hist(dev.SalePrice, bins=35)
    axes[0,0].set(title='development: giá gốc', xlabel='SalePrice')
    axes[0,1].hist(np.log(dev.SalePrice), bins=35)
    axes[0,1].set(title='development: log giá', xlabel='log(SalePrice)')
    missing.head(12).sort_values().plot.barh(ax=axes[1,0])
    axes[1,0].set(title='tỷ lệ thiếu cao nhất', xlabel='tỷ lệ thiếu')
    axes[1,1].scatter(dev.GrLivArea, np.log(dev.SalePrice), alpha=.4, s=12)
    axes[1,1].set(title='diện tích và log giá', xlabel='GrLivArea', ylabel='log(SalePrice)')
    fig.savefig(run/'figures/eda.png', dpi=130)
    plt.show()
    print('chỉ EDA development; giữ nguyên ngoại lệ, không xóa theo điểm holdout')
    return missing.head(12)


def run_trials(dev, folds, run):
    X = dev.drop(columns=['Id', 'SalePrice'])
    y = np.log(dev.SalePrice.to_numpy())
    records = []
    for exp_id, mode, depth, leaf, hypothesis in TRIALS:
        start = time.perf_counter()
        scores, importance, oof = [], [], np.empty(len(dev))
        for number, (tr, va) in enumerate(folds, 1):
            fitted = pipeline(mode, depth, leaf)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                fitted.fit(X.iloc[tr], y[tr])
            predicted = fitted.predict(X.iloc[va])
            train_pred = fitted.predict(X.iloc[tr])
            oof[va] = predicted
            scores.append(dict(fold=number, rmse_log=root_mean_squared_error(y[va], predicted),
                mae_price=mean_absolute_error(np.exp(y[va]), np.exp(predicted)),
                train_rmse_log=root_mean_squared_error(y[tr], train_pred),
                warnings=' | '.join(str(w.message) for w in caught)))
            importance.append(feature_importance(fitted).rename(number))
            print(exp_id, f'fold {number}/5', f'rmse={scores[-1]["rmse_log"]:.5f}', flush=True)
        scores = pd.DataFrame(scores)
        scores.to_csv(run/f'{exp_id}_fold_scores.csv', index=False)
        pd.DataFrame(dict(Id=dev.Id, true_log=y, predicted_log=oof)).to_csv(
            run/'predictions'/f'{exp_id}_oof.csv', index=False)
        pd.concat(importance, axis=1).fillna(0).mean(axis=1).sort_values(ascending=False).rename(
            'mean_impurity_importance').to_csv(run/f'{exp_id}_importance.csv')
        records.append(dict(exp_id=exp_id, date=datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).strftime('%d/%m/%Y'),
            member='như', model='random forest', mode=mode, max_depth=depth,
            min_samples_leaf=leaf, n_estimators=100, seed=SEED, hypothesis=hypothesis,
            cv_rmse_mean=scores.rmse_log.mean(), cv_rmse_std=scores.rmse_log.std(ddof=1),
            train_rmse_mean=scores.train_rmse_log.mean(), cv_mae_price=scores.mae_price.mean(),
            seconds=time.perf_counter()-start, warning_folds=int(scores.warnings.ne('').sum())))
        pd.DataFrame(records).to_csv(run/'comparison.csv', index=False)
    comparison = pd.DataFrame(records)
    # Chọn chỉ theo CV; không đọc nhãn holdout để chọn.
    best = comparison.sort_values(['cv_rmse_mean', 'exp_id']).iloc[0]
    selected = next(t for t in TRIALS if t[0] == best.exp_id)
    lock = dict(exp_id=selected[0], mode=selected[1], max_depth=selected[2],
                min_samples_leaf=selected[3], selection='CV RMSE log trung bình thấp nhất',
                cv_rmse_mean=float(best.cv_rmse_mean))
    write_json(run/'selected_config.json', lock)
    return comparison, lock


def plot_comparison(comparison, run):
    fig, ax = plt.subplots(figsize=(10,4), layout='constrained')
    ax.errorbar(comparison.exp_id, comparison.cv_rmse_mean,
                yerr=comparison.cv_rmse_std, fmt='o-', capsize=4, label='validation ± sd')
    ax.plot(comparison.exp_id, comparison.train_rmse_mean, 'o--', label='train')
    ax.set(xlabel='thí nghiệm', ylabel='rmse log', title='rf: train và 5-fold cv')
    ax.tick_params(axis='x', rotation=15)
    ax.legend(); ax.grid(alpha=.2)
    fig.savefig(run/'figures/cv_comparison.png', dpi=130); plt.show()


def finish(train, test, dev, hold, lock, comparison, run):
    assert json.loads((run/'selected_config.json').read_text(encoding='utf8')) == lock
    assert not (run/'holdout_report.json').exists(), 'holdout đã đánh giá trong lượt này'
    X = dev.drop(columns=['Id', 'SalePrice'])
    y = np.log(dev.SalePrice)
    fitted = pipeline(lock['mode'], lock['max_depth'], lock['min_samples_leaf'])
    fitted.fit(X, y)
    predicted = fitted.predict(hold.drop(columns=['Id', 'SalePrice']))
    truth = np.log(hold.SalePrice.to_numpy())
    report = dict(exp_id=lock['exp_id'], n_holdout=len(hold),
        rmse_log=float(root_mean_squared_error(truth, predicted)),
        mae_price=float(mean_absolute_error(hold.SalePrice, np.exp(predicted))),
        development_train_rmse_log=float(root_mean_squared_error(y, fitted.predict(X))),
        used_for_selection=False)
    write_json(run/'holdout_report.json', report)
    error = pd.DataFrame(dict(Id=hold.Id, actual_price=hold.SalePrice,
        predicted_price=np.exp(predicted), residual_log=truth-predicted,
        GrLivArea=hold.GrLivArea, OverallQual=hold.OverallQual))
    error['absolute_price_error'] = abs(error.actual_price-error.predicted_price)
    error['price_band'] = pd.cut(error.actual_price, [0,150000,250000,np.inf],
                                labels=['<150k','150k–250k','>250k'])
    error.to_csv(run/'predictions/holdout_errors.csv', index=False)
    groups=error.groupby('price_band', observed=True).agg(
        n=('Id','size'), mae_price=('absolute_price_error','mean'),
        mean_residual_log=('residual_log','mean'))
    groups.to_csv(run/'holdout_error_groups.csv')
    fig, axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].scatter(truth, predicted, alpha=.5, s=16)
    bounds=[min(truth.min(),predicted.min()),max(truth.max(),predicted.max())]
    axes[0].plot(bounds,bounds,'k--');axes[0].set(xlabel='log giá thật',ylabel='log dự đoán',title='holdout')
    axes[1].scatter(predicted,truth-predicted,alpha=.5,s=16)
    axes[1].axhline(0,color='black',linestyle='--')
    axes[1].set(xlabel='log dự đoán',ylabel='thật − dự đoán',title='phần dư holdout')
    fig.savefig(run/'figures/holdout_errors.png',dpi=130);plt.show()
    importance=pd.read_csv(run/f'{lock["exp_id"]}_importance.csv').set_index('feature')
    fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
    importance.head(15).iloc[::-1,0].plot.barh(ax=ax)
    ax.set(title='importance trung bình từ 5 RF fit trên train folds',xlabel='impurity importance')
    fig.savefig(run/'figures/feature_importance.png',dpi=130);plt.show()
    final=pipeline(lock['mode'],lock['max_depth'],lock['min_samples_leaf'])
    final.fit(train.drop(columns=['Id','SalePrice']),np.log(train.SalePrice))
    price=np.exp(final.predict(test.drop(columns='Id')))
    submission=pd.DataFrame({'Id':test.Id,'SalePrice':price})
    assert submission.shape==(1459,2) and np.isfinite(price).all() and (price>0).all()
    assert submission.Id.tolist()==pd.read_csv(ROOT/'data/raw/sample_submission.csv').Id.tolist()
    submission.to_csv(run/'predictions/submission_Nhu.csv',index=False)
    joblib.dump(final,run/'models/rf_pipeline.joblib')
    (ROOT/'results/submissions').mkdir(parents=True,exist_ok=True)
    submission.to_csv(ROOT/'results/submissions/submission_Nhu.csv',index=False)
    (OUT/'latest_run.txt').write_text(str(run.relative_to(ROOT)),encoding='utf8')
    write_report(run,comparison,lock,report,groups,importance)
    return report,groups,submission.head()


def write_report(run,comparison,lock,report,groups,importance):
    best=comparison.loc[comparison.exp_id==lock['exp_id']].iloc[0]
    baseline=comparison.iloc[0]
    rows='\n'.join(f'| {r.exp_id} | {r.mode} | {r.cv_rmse_mean:.5f} ± {r.cv_rmse_std:.5f} | {r.cv_mae_price:,.0f} |'
                   for r in comparison.itertuples())
    text=f'''# random forest house prices — như

Ngày thực hiện: {datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')):%d/%m/%Y}.

Dự đoán SalePrice từ 79 thuộc tính gốc; bỏ Id khỏi X. Train có 1.460 nhà,
development 1.168 và holdout 292 nhà theo split seed 7 của nhóm.
Huấn luyện bằng log tự nhiên của giá, chuyển lại bằng exp khi tính MAE và xuất dự đoán.

## giả thuyết và phương pháp

Kiểm tra xử lý NA theo ý nghĩa, đặc trưng diện tích/tuổi nhà/tiện ích và giới hạn
độ sâu hoặc kích thước lá có giảm sai số so với RF cơ sở không.
Mỗi fold tự fit imputer và one-hot. RF không cần chuẩn hóa. Không loại ngoại lệ,
không fit preprocessing trên holdout hoặc test. So sánh sáu phương án đã định trước,
chọn bằng 5-fold CV; độ lệch chuẩn mẫu dùng ddof=1.

| thí nghiệm | đặc trưng | cv rmse log ± sd | cv mae giá |
|---|---|---:|---:|
{rows}

## kết quả

Chọn {lock['exp_id']} bằng CV: {best.cv_rmse_mean:.5f} ± {best.cv_rmse_std:.5f}.
Chênh lệch với baseline: {best.cv_rmse_mean-baseline.cv_rmse_mean:+.5f}; âm là giảm sai số.
Holdout được đánh giá sau khi lưu selected_config.json: RMSE log {report['rmse_log']:.5f},
MAE giá {report['mae_price']:,.0f}. Điểm train development {report['development_train_rmse_log']:.5f}
cần đọc cùng CV/holdout để nhận diện khoảng cách train–validation.
Không đổi cấu hình sau khi xem holdout. Chưa có điểm Kaggle.

## sai số và đóng góp đặc trưng

Nhóm giá có MAE lớn nhất trên holdout: {groups.mae_price.idxmax()}.
Ba feature đứng đầu importance CV: {', '.join(importance.index[:3])}.
Importance đo mức giảm impurity, có thể thiên về biến nhiều mức và chia sẻ đóng góp
giữa biến tương quan; không chứng minh quan hệ nhân quả. Exp003 so với exp002 là
kiểm tra thêm cả nhóm feature, chưa tách ablation từng feature.

## sản phẩm và chạy lại

Notebook: members/Nhu/notebooks/15_11_2026_rf.ipynb.
Kết quả: {run.relative_to(ROOT).as_posix()}.
CSV nộp: results/submissions/submission_Nhu.csv; model đã fit toàn bộ train nằm trong run/models.
Mỗi lần chạy notebook tạo thư mục mới. Chỉ nạp joblib do nhóm tự tạo.
Các cải tiến có thể làm sau: permutation importance trong CV, ablation từng nhóm,
khảo sát thêm max_features và số cây bằng development CV, không tối ưu theo holdout.
'''
    (ROOT/'reports/individual/Nhu_RF_report.md').write_text(text,encoding='utf8')


def verify(run):
    comparison=pd.read_csv(run/'comparison.csv')
    folds=list((run).glob('Nhu_EXP-*_fold_scores.csv'))
    assert len(comparison)==6 and len(folds)==6
    for row in comparison.itertuples():
        scores=pd.read_csv(run/f'{row.exp_id}_fold_scores.csv')
        oof=pd.read_csv(run/'predictions'/f'{row.exp_id}_oof.csv')
        assert len(scores)==5 and len(oof)==1168 and oof.Id.is_unique
        assert np.isclose(scores.rmse_log.mean(),row.cv_rmse_mean)
        # RMSE gộp OOF có thể khác trung bình RMSE của 5 fold.
        assert np.isfinite(root_mean_squared_error(oof.true_log,oof.predicted_log))
    saved=joblib.load(run/'models/rf_pipeline.joblib')
    _,test,_,_,_=load_data()
    submitted=pd.read_csv(run/'predictions/submission_Nhu.csv')
    assert np.allclose(np.exp(saved.predict(test.drop(columns='Id'))),submitted.SalePrice)
    assert submitted.Id.tolist()==test.Id.tolist()
    write_json(run/'verification.json',dict(passed=True,experiments=6,cv_folds=30,
        submission_rows=1459,model_reload_predictions_match=True))
    print('kiểm tra đạt: 6 thí nghiệm, 30 fold, 1.459 dự đoán; nạp model chạy lại khớp')
