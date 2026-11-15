"""House Prices: thí nghiệm Ridge và ElasticNet của Thái."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import time
import hashlib
import warnings
import platform
import subprocess
import tempfile
import joblib
import numpy as np
import pandas as pd
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'data/raw/train.csv').is_file())
OUT = ROOT / 'members/Thai/outputs'
LOG = ROOT / 'reports/individual/nhat_ky_thuc_nghiem.csv'
for sub in ['figures', 'models', 'predictions', 'experiments']:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

# Ngày do người dùng chỉ định cho nhật ký; JSON lưu thêm thời điểm chạy máy.
TRIALS = [
    ('01', 'ridge', 1., None, 'basic', 'Ridge cơ sở', 'Mốc so sánh với điền thiếu, one-hot, chuẩn hóa.'),
    ('02', 'ridge', .1, None, 'basic', 'Giảm alpha còn 0.1', 'Giảm điều chuẩn có thể giảm underfitting.'),
    ('03', 'ridge', 10., None, 'basic', 'Tăng alpha lên 10', 'Điều chuẩn mạnh hơn có thể giảm phương sai.'),
    ('04', 'ridge', 30., None, 'basic', 'Tăng alpha lên 30', 'Kiểm tra thêm mức điều chuẩn cao hơn.'),
    ('05', 'ridge', 100., None, 'basic', 'Tăng alpha lên 100', 'Kiểm tra nguy cơ underfitting khi alpha lớn.'),
    ('06', 'ridge', 30., None, 'semantic', 'Xử lý NA theo ý nghĩa', 'Phân biệt không có tiện ích với dữ liệu chưa biết.'),
    ('07', 'ridge', 30., None, 'log', 'Log các biến diện tích', 'Giảm độ lệch của biến số bằng log1p.'),
    ('08', 'ridge', 30., None, 'engineered', 'Thêm diện tích và tuổi nhà', 'Tổng diện tích và tuổi nhà có thể giải thích giá.'),
    ('09', 'ridge', 10., None, 'engineered', 'Ridge đặc trưng mới alpha 10', 'Kiểm tra điều chuẩn sau khi thêm đặc trưng.'),
    ('10', 'elasticnet', .001, .5, 'engineered', 'ElasticNet cơ sở', 'Kết hợp L1 và L2 có thể giảm đặc trưng ít hữu ích.'),
    ('11', 'elasticnet', .0005, .5, 'engineered', 'ElasticNet alpha 0.0005', 'Giảm độ co của hệ số.'),
    ('12_01', 'elasticnet', .005, .5, 'engineered', 'ElasticNet alpha 0.005', 'Kiểm tra điều chuẩn mạnh hơn.'),
    ('12_02', 'elasticnet', .001, .1, 'engineered', 'ElasticNet l1_ratio 0.1', 'Ưu tiên L2 khi các đặc trưng tương quan.'),
    ('13_01', 'elasticnet', .001, .9, 'engineered', 'ElasticNet l1_ratio 0.9', 'Ưu tiên L1 để chọn bớt đặc trưng.'),
    ('13_02', 'elasticnet', .0005, .1, 'engineered', 'ElasticNet alpha 0.0005 ratio 0.1', 'Kiểm tra kết hợp alpha nhỏ và ưu tiên L2.'),
]

def configs():
    return [dict(exp_id=f'Thai_EXP-{i:03}', date=f'{day[:2]}/11/2026', day=day,
        model=model, alpha=alpha, l1_ratio=ratio, mode=mode, title=title, hypothesis=hypothesis,
        notebook=f'members/Thai/notebooks/{day}_11_2026.ipynb')
        for i, (day, model, alpha, ratio, mode, title, hypothesis) in enumerate(TRIALS, 1)]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_excel_log():
    """Cập nhật Excel từ nhật ký CSV sau mỗi thí nghiệm."""
    exporter = Path(__file__).with_name('export_log.mjs')
    try:
        with tempfile.TemporaryFile() as errors:
            completed = subprocess.run(['node', str(exporter)], stdout=subprocess.DEVNULL,
                                       stderr=errors, timeout=60)
            if completed.returncode:
                errors.seek(0)
                raise RuntimeError(errors.read().decode('utf-8', errors='replace'))
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print('CSV đã lưu. Chưa cập nhật được Excel:', str(error))

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def load_data():
    train = pd.read_csv(ROOT / 'data/raw/train.csv')
    test = pd.read_csv(ROOT / 'data/raw/test.csv')
    assert train.Id.is_unique and test.Id.is_unique
    assert train.SalePrice.notna().all() and (train.SalePrice > 0).all()
    assert set(train.columns) - {'SalePrice'} == set(test.columns)
    return train, test

def get_split(train):
    dev, holdout = train_test_split(np.arange(len(train)), test_size=.2, random_state=7)
    folds = list(KFold(5, shuffle=True, random_state=7).split(dev))
    split = pd.DataFrame({'Id': train.Id, 'split': 'development', 'cv_fold': -1})
    split.loc[holdout, 'split'] = 'holdout'
    for fold, (_, valid) in enumerate(folds):
        split.loc[dev[valid], 'cv_fold'] = fold
    path = ROOT / 'data/splits/split_seed7.csv'
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(path), split)
    else:
        split.to_csv(path, index=False)
    return dev, holdout, folds

class HouseFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, mode='basic'):
        self.mode = mode

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.mode == 'basic':
            return X
        for col in ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                    'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish',
                    'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']:
            X[col] = X[col].fillna('None')
        for col in ['GarageYrBlt', 'GarageCars', 'GarageArea', 'BsmtFinSF1', 'BsmtFinSF2',
                    'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']:
            X[col] = X[col].fillna(0)
        X['MSSubClass'] = X['MSSubClass'].astype(str)
        if self.mode == 'engineered':
            X['TotalArea'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
            X['HouseAge'] = (X['YrSold'] - X['YearBuilt']).clip(lower=0)
            X['RemodelAge'] = (X['YrSold'] - X['YearRemodAdd']).clip(lower=0)
            X['TotalBath'] = X['FullBath'] + .5*X['HalfBath'] + X['BsmtFullBath'] + .5*X['BsmtHalfBath']
            X['QualArea'] = X['OverallQual'] * X['GrLivArea']
        if self.mode in ['log', 'engineered']:
            for col in ['LotArea', 'GrLivArea', '1stFlrSF', '2ndFlrSF', 'TotalBsmtSF',
                        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'WoodDeckSF', 'OpenPorchSF',
                        'EnclosedPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MasVnrArea',
                        'TotalArea', 'QualArea']:
                if col in X:
                    X[col] = np.log1p(X[col].clip(lower=0))
        return X

def make_pipeline(config):
    numeric = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    category = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                         ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True))])
    prep = ColumnTransformer([
        ('numeric', numeric, make_column_selector(dtype_include=np.number)),
        ('category', category, make_column_selector(dtype_exclude=np.number))])
    model = (Ridge(alpha=config['alpha'], solver='lsqr', tol=1e-6) if config['model'] == 'ridge'
        else ElasticNet(alpha=config['alpha'], l1_ratio=config['l1_ratio'], max_iter=30000,
                        tol=1e-5, random_state=7))
    return Pipeline([('features', HouseFeatures(config['mode'])), ('preprocess', prep), ('model', model)])

def eda():
    train, _ = load_data()
    dev, _, _ = get_split(train)
    data = train.iloc[dev]
    missing = data.isna().sum().rename('missing').to_frame()
    missing['fraction'] = missing['missing'] / len(data)
    missing.to_csv(OUT / 'experiments/missing_development.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].hist(data.SalePrice, bins=30)
    axes[0].set_title('SalePrice - development')
    axes[1].hist(np.log(data.SalePrice), bins=30)
    axes[1].set_title('log(SalePrice) - development')
    fig.tight_layout()
    fig.savefig(OUT / 'figures/saleprice_distribution.png', dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, col in zip(axes, ['GrLivArea', 'OverallQual']):
        ax.scatter(data[col], np.log(data.SalePrice), s=12, alpha=.4)
        ax.set(xlabel=col, ylabel='log(SalePrice)')
    fig.tight_layout()
    fig.savefig(OUT / 'figures/feature_relations.png', dpi=130)
    plt.close(fig)
    print(f'EDA chỉ dùng {len(data)} dòng development.')
    print(f'Skew giá: {data.SalePrice.skew():.3f}; log giá: {np.log(data.SalePrice).skew():.3f}')
    print(missing.sort_values('missing', ascending=False).head(10).to_string())

def run_experiment(config):
    train, _ = load_data()
    dev, _, folds = get_split(train)
    X = train.drop(columns=['Id', 'SalePrice']).iloc[dev]
    y = np.log(train.SalePrice.iloc[dev].to_numpy())
    actual_time = datetime.now(timezone(timedelta(hours=7))).isoformat()
    folder = OUT / 'experiments' / config['exp_id'] / datetime.now().strftime('%Y%m%dT%H%M%S%f')
    folder.mkdir(parents=True, exist_ok=False)
    scores, messages = [], []
    oof = np.full(len(y), np.nan)
    start = time.perf_counter()
    for fold, (fit_idx, val_idx) in enumerate(folds):
        model = make_pipeline(config)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            model.fit(X.iloc[fit_idx], y[fit_idx])
        messages.extend(str(w.message) for w in captured)
        prediction = model.predict(X.iloc[val_idx])
        oof[val_idx] = prediction
        scores.append(dict(fold=fold, n_train=len(fit_idx), n_valid=len(val_idx),
            rmse_log=float(np.sqrt(mean_squared_error(y[val_idx], prediction))),
            mae_price=float(mean_absolute_error(np.exp(y[val_idx]), np.exp(prediction))),
            train_rmse_log=float(np.sqrt(mean_squared_error(y[fit_idx], model.predict(X.iloc[fit_idx]))))))
    assert np.isfinite(oof).all()
    pd.DataFrame(scores).to_csv(folder / 'fold_scores.csv', index=False)
    pd.DataFrame({'Id': train.Id.iloc[dev].to_numpy(), 'actual_log': y, 'predicted_log': oof}
                 ).to_csv(folder / 'oof_predictions.csv', index=False)
    mean = float(np.mean([s['rmse_log'] for s in scores]))
    std = float(np.std([s['rmse_log'] for s in scores], ddof=1))
    result = dict(config=config, executed_at=actual_time, python=platform.python_version(),
        sklearn=sklearn.__version__, train_sha256=sha(ROOT / 'data/raw/train.csv'),
        split_sha256=sha(ROOT / 'data/splits/split_seed7.csv'), source_sha256=sha(__file__),
        cv_rmse_log_mean=mean, cv_rmse_log_std=std,
        cv_mae_price=float(np.mean([s['mae_price'] for s in scores])),
        seconds=time.perf_counter()-start, warnings=sorted(set(messages)), folds=scores)
    baseline = sorted((OUT / 'experiments/Thai_EXP-001').glob('*/result.json'))
    conclusion = 'Mốc cơ sở để so sánh.'
    if config['exp_id'] != 'Thai_EXP-001' and baseline:
        ref = json.loads(baseline[-1].read_text(encoding='utf-8'))['cv_rmse_log_mean']
        conclusion = f'CV RMSE log {"giảm" if mean < ref else "tăng"} {abs(mean-ref):.5f} so với Ridge cơ sở.'
    result['conclusion'] = conclusion
    write_json(folder / 'result.json', result)
    row = {'Ngày thực hiện': config['date'], 'Mã thực nghiệm': config['exp_id'], 'Thành viên': 'Thai',
        'Mô hình': config['model'], 'Tiền xử lý': config['mode'],
        'Tham số': f"alpha={config['alpha']}; l1_ratio={config['l1_ratio']}",
        'Giả thuyết': config['hypothesis'], 'CV RMSE log TB': mean, 'CV RMSE log ĐLC': std,
        'CV MAE giá': result['cv_mae_price'], 'Thời gian (giây)': result['seconds'],
        'Kết luận': conclusion, 'File minh chứng': config['notebook'],
        'File kết quả': (folder / 'result.json').relative_to(ROOT).as_posix(),
        'Thời điểm chạy máy': actual_time, 'SHA256 train': result['train_sha256'],
        'Trạng thái': 'Đã chạy' if not messages else 'Đã chạy, xem cảnh báo trong JSON'}
    frame = pd.DataFrame([row])
    if LOG.exists():
        old = pd.read_csv(LOG, encoding='utf-8-sig')
        frame = pd.concat([old, frame], ignore_index=True)
        frame = frame[['Ngày thực hiện'] + [c for c in frame if c != 'Ngày thực hiện']]
    frame.to_csv(LOG, index=False, encoding='utf-8-sig')
    export_excel_log()
    print(config['exp_id'], config['title'])
    print(f'CV RMSE log: {mean:.5f} ± {std:.5f}')
    print(f"MAE giá: {result['cv_mae_price']:,.0f}")
    print(conclusion)
    print('Minh chứng:', row['File kết quả'])
    return result

def finalize():
    train, test = load_data()
    dev, hold, _ = get_split(train)
    results = []
    for config in configs():
        files = sorted((OUT / 'experiments' / config['exp_id']).glob('*/result.json'))
        if not files:
            raise RuntimeError(f"Chưa chạy {config['exp_id']}")
        result = json.loads(files[-1].read_text(encoding='utf-8'))
        assert result['train_sha256'] == sha(ROOT / 'data/raw/train.csv')
        assert result['split_sha256'] == sha(ROOT / 'data/splits/split_seed7.csv')
        results.append(result)
    best = min(results, key=lambda r: r['cv_rmse_log_mean'])
    config = best['config']
    X = train.drop(columns=['Id', 'SalePrice'])
    y = np.log(train.SalePrice)
    model = make_pipeline(config)
    model.fit(X.iloc[dev], y.iloc[dev])
    prediction = model.predict(X.iloc[hold])
    summary = dict(selected_exp=config['exp_id'], selected_config=config,
        selection='Chọn bằng CV trên development trước khi đánh giá holdout',
        holdout_rmse_log=float(np.sqrt(mean_squared_error(y.iloc[hold], prediction))),
        holdout_mae_price=float(mean_absolute_error(train.SalePrice.iloc[hold], np.exp(prediction))))
    errors = pd.DataFrame({'Id': train.Id.iloc[hold].to_numpy(),
        'actual': train.SalePrice.iloc[hold].to_numpy(), 'predicted': np.exp(prediction),
        'residual_log': y.iloc[hold].to_numpy()-prediction})
    errors['absolute_error'] = abs(errors.actual-errors.predicted)
    errors.to_csv(OUT / 'predictions/holdout_errors.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].scatter(prediction, errors.residual_log, s=12, alpha=.5)
    axes[0].axhline(0, color='red', linewidth=1)
    axes[0].set(xlabel='Predicted log price', ylabel='Residual log price')
    axes[1].bar(range(1, 16), [r['cv_rmse_log_mean'] for r in results])
    axes[1].set(xlabel='Experiment', ylabel='CV RMSE log')
    fig.tight_layout()
    fig.savefig(OUT / 'figures/cv_and_residuals.png', dpi=130)
    plt.close(fig)
    model.fit(X, y)
    prices = np.exp(model.predict(test.drop(columns='Id')))
    assert np.isfinite(prices).all() and (prices > 0).all()
    submission = pd.DataFrame({'Id': test.Id, 'SalePrice': prices})
    sample = pd.read_csv(ROOT / 'data/raw/sample_submission.csv')
    assert submission.Id.equals(sample.Id) and list(submission.columns) == list(sample.columns)
    submission.to_csv(OUT / 'predictions/submission_Thai.csv', index=False)
    submission.to_csv(ROOT / 'results/submissions/submission_Thai.csv', index=False)
    joblib.dump(model, OUT / 'models/best_pipeline.joblib')
    write_json(OUT / 'experiments/final_summary.json', summary)
    comparison = pd.DataFrame([dict(exp_id=r['config']['exp_id'], model=r['config']['model'],
        mode=r['config']['mode'], alpha=r['config']['alpha'], l1_ratio=r['config']['l1_ratio'],
        cv_rmse_log=r['cv_rmse_log_mean'], cv_std=r['cv_rmse_log_std']) for r in results])
    comparison.to_csv(OUT / 'experiments/comparison.csv', index=False)
    report = f'''# Thực nghiệm của Thái — House Prices

## Phương pháp

Ridge và ElasticNet. Chia development/holdout 80/20, seed=7; 5-fold CV trên development.
Điền thiếu, one-hot và chuẩn hóa fit riêng trong từng fold bằng pipeline.
Nhãn huấn luyện là log(SalePrice), chuyển dự đoán về giá bằng exp().
Id chỉ dùng để đối chiếu và xuất dự đoán. Không loại ngoại lệ dựa trên holdout.

## Kết quả

- Cấu hình chọn: {config['exp_id']} — {config['title']}.
- CV RMSE log: {best['cv_rmse_log_mean']:.5f} ± {best['cv_rmse_log_std']:.5f} (ĐLC mẫu của 5 fold).
- Holdout RMSE log: {summary['holdout_rmse_log']:.5f}.
- Holdout MAE giá: {summary['holdout_mae_price']:,.0f}.
- Dự đoán cuối: results/submissions/submission_Thai.csv, {len(submission)} dòng.

## Phân tích sai số

5 nhà có sai số tuyệt đối lớn nhất trên holdout:

```
{errors.nlargest(5, 'absolute_error').to_string(index=False)}
```

Biểu đồ: members/Thai/outputs/figures/cv_and_residuals.png.
Không chỉnh cấu hình theo holdout. Một lần chia chưa đủ chứng minh ưu thế thống kê.
Các biến log chọn trước, chưa tối ưu ngoại lệ. NA của garage cần xem thêm năm xây dựng và
thuộc tính liên quan; phiên bản hiện tại dùng quy tắc đơn giản và giữ cảnh báo trong JSON.

## Ngày và minh chứng

Ngày 01–13/11/2026 là mốc nhật ký người dùng quy định. Thời điểm chạy máy lưu riêng trong JSON.
Nhật ký: reports/individual/nhat_ky_thuc_nghiem.csv và .xlsx.
Mỗi dòng có nguồn notebook, JSON kết quả, checksum, điểm từng fold và thời điểm chạy máy.
Kết quả CV là số đo chạy thực tế. test.csv không có nhãn, chưa có điểm Kaggle.
'''
    (ROOT / 'reports/individual/Thai_report.md').write_text(report, encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
