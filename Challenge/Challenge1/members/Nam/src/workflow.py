"""Reproducible SVR experiments for SGU Challenge 1 / member Nam.
Run from any working directory: python path/to/workflow.py --stage V0
"""
from __future__ import annotations
import argparse, copy, json, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, GridSearchCV, cross_validate, train_test_split, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

NAM = Path(__file__).resolve().parents[1]
CHALLENGE = NAM.parents[1]
DATA = CHALLENGE / 'data' / 'raw'
SPLITS = CHALLENGE / 'data' / 'splits'
OUT = NAM / 'outputs'
FIG = OUT / 'figures'
MODELS = OUT / 'models'
PRED = OUT / 'predictions'
LOG = OUT / 'experiment_log.csv'
CONFIG = OUT / 'selected_configs.json'
SEED = 7
CV = KFold(n_splits=5, shuffle=True, random_state=SEED)


def ensure_dirs():
    for p in (SPLITS, FIG, MODELS, PRED, CHALLENGE/'results'/'submissions', CHALLENGE/'reports'/'individual'):
        p.mkdir(parents=True, exist_ok=True)


def read_data():
    if not (DATA/'train.csv').exists():
        raise FileNotFoundError(f'Cannot find {DATA / "train.csv"}. Put the ZIP contents inside Challenge1, keeping data/raw/ from the repository.')
    train = pd.read_csv(DATA/'train.csv')
    if not {'Id','SalePrice'}.issubset(train.columns):
        raise ValueError('train.csv must include Id and SalePrice')
    if (train.SalePrice<=0).any():
        raise ValueError('SalePrice must be strictly positive to use log')
    return train


def split_data(train):
    """Shared split files are authoritative if they exist, otherwise create them. Avoid overriding group splits."""
    ensure_dirs()
    devfile, holdfile = SPLITS/'development_ids.csv', SPLITS/'holdout_ids.csv'
    if devfile.exists() != holdfile.exists():
        raise RuntimeError('Only one shared split file exists; ask the group to complete both files.')
    if devfile.exists():
        dev_ids = pd.read_csv(devfile)['Id'].tolist()
        hold_ids = pd.read_csv(holdfile)['Id'].tolist()
    else:
        dev_ids, hold_ids = train_test_split(train['Id'].tolist(), test_size=.20, random_state=SEED, shuffle=True)
        pd.DataFrame({'Id':dev_ids}).to_csv(devfile,index=False)
        pd.DataFrame({'Id':hold_ids}).to_csv(holdfile,index=False)
    known=set(train.Id)
    if set(dev_ids)&set(hold_ids) or set(dev_ids)|set(hold_ids)!=known or len(dev_ids)+len(hold_ids)!=len(train):
        raise ValueError('Invalid shared Id split: non-disjoint, missing, duplicated or out-of-range Ids')
    return train.set_index('Id',drop=False).loc[dev_ids].reset_index(drop=True), train.set_index('Id',drop=False).loc[hold_ids].reset_index(drop=True)


def xy(frame):
    return frame.drop(columns=['Id','SalePrice'], errors='ignore'), np.log(frame['SalePrice'].to_numpy())


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Deterministic, row-only features: no learned statistics or leakage."""
    def __init__(self, mode='baseline', drop_columns=()):
        self.mode=mode
        self.drop_columns=drop_columns
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        x=X.copy()
        if self.mode in ('preprocess','selected'):
            for c in ('LotArea','GrLivArea','1stFlrSF','TotalBsmtSF'):
                if c in x:
                    x[c] = np.log1p(pd.to_numeric(x[c], errors='coerce').clip(lower=0))
        if self.mode=='selected':
            if {'YrSold','YearBuilt'}.issubset(x):
                x['HouseAgeAtSale']=(pd.to_numeric(x['YrSold'], errors='coerce') - pd.to_numeric(x['YearBuilt'],errors='coerce')).clip(lower=0)
            if {'1stFlrSF','2ndFlrSF'}.issubset(x):
                # the first-floor column was already log-transformed: use GrLivArea as-is instead
                x['SecondFloorPresent']=(pd.to_numeric(x['2ndFlrSF'],errors='coerce').fillna(0)>0).astype(int)
            if {'OverallQual','GrLivArea'}.issubset(x):
                x['QualityLogLivingArea']=pd.to_numeric(x['OverallQual'],errors='coerce')*pd.to_numeric(x['GrLivArea'],errors='coerce')
            x=x.drop(columns=list(self.drop_columns),errors='ignore')
        return x


def encoder():
    try: return OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    except TypeError: return OneHotEncoder(handle_unknown='ignore', sparse=True)


def build_pipeline(X_dev, mode='baseline', params=None):
    """Learn numeric/categorical column membership from development only; folds learn fitted stats."""
    drop=()
    if mode=='selected':
        # Predefined unsupervised missingness rule: derive ONLY on development, not holdout
        drop=tuple(c for c in X_dev if X_dev[c].isna().mean()>=0.65)
    builder=FeatureBuilder(mode=mode,drop_columns=drop)
    example=builder.transform(X_dev.iloc[:min(20,len(X_dev))])
    numerical=example.select_dtypes(include='number').columns.tolist()
    categorical=[c for c in example.columns if c not in numerical]
    prep=ColumnTransformer([
        ('num',Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),('scaler',StandardScaler())]),numerical),
        ('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent',keep_empty_features=True)),('onehot',encoder())]),categorical)
    ],remainder='drop')
    pipe=Pipeline([('features',builder),('preprocess',prep),('svr',SVR(kernel='rbf'))])
    if params: pipe.set_params(**params)
    return pipe,drop


def score_cv(pipe,X,y):
    t=time.time()
    s=cross_validate(pipe,X,y,cv=CV,scoring='neg_root_mean_squared_error',return_train_score=True,n_jobs=1)
    return {'cv_rmse_mean':float(-s['test_score'].mean()),'cv_rmse_std':float(s['test_score'].std()),'train_rmse_mean':float(-s['train_score'].mean()),'seconds':round(time.time()-t,2)}


def log_result(version,variant,metrics,params=None,notes=''):
    row={'timestamp':datetime.now().isoformat(timespec='seconds'),'version':version,'variant':variant,
         'cv_rmse_mean':metrics['cv_rmse_mean'],'cv_rmse_std':metrics['cv_rmse_std'],
         'train_rmse_mean':metrics['train_rmse_mean'],'seconds':metrics['seconds'],
         'params':json.dumps(params or {},ensure_ascii=False),'notes':notes}
    df=pd.DataFrame([row]);df.to_csv(LOG,mode='a',header=not LOG.exists(),index=False)
    print(f"[{version} | {variant}] CV RMSE = {row['cv_rmse_mean']:.5f} ± {row['cv_rmse_std']:.5f}; train = {row['train_rmse_mean']:.5f}; {row['seconds']}s")
    return row


def get_config():
    return json.loads(CONFIG.read_text()) if CONFIG.exists() else {}


def save_config(version,mode,params,metrics):
    x=get_config()
    x[version]={'mode':mode,'params':params,'metrics':metrics}
    CONFIG.write_text(json.dumps(x,indent=2,ensure_ascii=False))


def require(version):
    d=get_config()
    if version not in d: raise RuntimeError(f'Run {version} first to build a comparable experiment history.')
    return d[version]


def eda():
    train=read_data(); dev,_=split_data(train)
    summary=pd.DataFrame({'dtype':dev.dtypes.astype(str),'missing':dev.isna().sum(),'missing_pct':dev.isna().mean().round(3)})
    summary.sort_values('missing_pct',ascending=False).to_csv(OUT/'eda_missingness.csv')
    fig,ax=plt.subplots(figsize=(8,4));ax.hist(np.log(dev.SalePrice),bins=35);ax.set(xlabel='log(SalePrice)',ylabel='Count',title='Development price distribution');fig.tight_layout();fig.savefig(FIG/'eda_log_price.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4));summary.query('missing_pct > 0').head(20)['missing_pct'].plot.bar(ax=ax);ax.set(ylabel='Missing fraction',title='Missingness in development');fig.tight_layout();fig.savefig(FIG/'eda_missingness.png',dpi=150);plt.close(fig)
    print(f'EDA saved: development={len(dev)}, features={len(dev.columns)-2}; see outputs/figures and eda_missingness.csv')
    return summary


def run_stage(stage):
    stage=stage.upper().strip()
    ensure_dirs()
    if stage=='EDA': return eda()
    train=read_data();dev,hold=split_data(train);X,y=xy(dev)
    if stage=='V0':
        pipe,drop=build_pipeline(X,'baseline'); m=score_cv(pipe,X,y)
        log_result(stage,'default RBF SVR',m,notes='Impute, encode, scale inside folds; log target')
        save_config(stage,'baseline',{},m);return m
    if stage=='V1':
        require('V0')
        candidates=[('log_skewed', 'preprocess', {}),('baseline_recheck','baseline',{})]
        results=[]
        for label,mode,p in candidates:
            pipe,_=build_pipeline(X,mode,p);m=score_cv(pipe,X,y);log_result(stage,label,m,p,'Compare feature transformations to original baseline');results.append((m['cv_rmse_mean'],label,mode,p,m))
        best=min(results)
        save_config(stage,best[2],best[3],best[4]);return best
    if stage=='V2':
        require('V1')
        a=require('V1');candidates=[('prior_v1',a['mode'],a['params']),('selected_features','selected',{})]
        results=[]
        for label,mode,p in candidates:
            pipe,drop=build_pipeline(X,mode,p);m=score_cv(pipe,X,y);log_result(stage,label,m,p,'Removed >65% missing and engineered features; dropped='+str(drop));results.append((m['cv_rmse_mean'],label,mode,p,m))
        best=min(results);save_config(stage,best[2],best[3],best[4]);return best
    if stage=='V3':
        a=require('V2');pipe,_=build_pipeline(X,a['mode'])
        grid={'svr__C':[1,10,100], 'svr__epsilon':[0.01,0.05,0.1], 'svr__gamma':['scale',0.001,0.01]}
        search=GridSearchCV(pipe,grid,cv=CV,scoring='neg_root_mean_squared_error',return_train_score=True,n_jobs=1,refit=True)
        start=time.time();search.fit(X,y)
        raw=pd.DataFrame(search.cv_results_)
        raw[['params','mean_test_score','std_test_score','mean_train_score','mean_fit_time']].to_csv(OUT/'v3_grid_search_results.csv',index=False)
        p=search.best_params_;m=score_cv(build_pipeline(X,a['mode'],p)[0],X,y)
        log_result(stage,'best_grid_search',m,p,f"Grid 27 candidates; total search seconds={time.time()-start:.1f}")
        save_config(stage,a['mode'],p,m);return p,m
    if stage=='V4':
        a=require('V3');pipe,_=build_pipeline(X,a['mode'],a['params'])
        pred=cross_val_predict(pipe,X,y,cv=CV,n_jobs=1)
        residual=y-pred
        errors=pd.DataFrame({'Id':dev.Id,'actual_price':np.exp(y),'oof_pred_price':np.exp(pred),'actual_log':y,'oof_pred_log':pred,'residual_log':residual,'abs_error_log':np.abs(residual)})
        errors.to_csv(PRED/'v4_oof_error_analysis.csv',index=False)
        fig,ax=plt.subplots(figsize=(7,5));ax.scatter(pred,residual,s=12,alpha=.5);ax.axhline(0,color='red',linestyle='--');ax.set(xlabel='Out-of-fold predicted log price',ylabel='Actual - prediction (log)',title='SVR out-of-fold residuals');fig.tight_layout();fig.savefig(FIG/'v4_residuals.png',dpi=150);plt.close(fig)
        groups=pd.qcut(errors.actual_price,4,duplicates='drop');res=errors.groupby(groups,observed=True)['abs_error_log'].agg(['mean','count']);res.to_csv(OUT/'v4_price_segment_errors.csv')
        m=score_cv(pipe,X,y);log_result(stage,'OOF error analysis',m,a['params'],'Diagnostics only; no holdout tuning')
        save_config(stage,a['mode'],a['params'],m)
        print('Saved OOF residual plot, error CSV, and price-segment summary')
        return res
    if stage=='V5':
        require('V3')
        configs=get_config()
        candidates=[(k,v) for k,v in configs.items() if k in ('V0','V1','V2','V3')]
        chosen=min(candidates,key=lambda t:t[1]['metrics']['cv_rmse_mean'])
        selected=chosen[1];pipe,_=build_pipeline(X,selected['mode'],selected['params'])
        Xh,yh=xy(hold);pipe.fit(X,y);prediction=pipe.predict(Xh)
        rmse=float(np.sqrt(mean_squared_error(yh,prediction)))
        mae=float(mean_absolute_error(np.exp(yh),np.exp(prediction)))
        report={'selected_version':chosen[0],'mode':selected['mode'],'params':selected['params'],
                'development_cv_rmse':selected['metrics']['cv_rmse_mean'],'holdout_rmse_log':rmse,'holdout_mae_price':mae,
                'development_rows':len(dev),'holdout_rows':len(hold),'random_state':SEED}
        (OUT/'v5_holdout_report.json').write_text(json.dumps(report,indent=2))
        # Refit on ALL labeled data after the once-only holdout evaluation
        Xall,yall=xy(train);final,_=build_pipeline(Xall,selected['mode'],selected['params']);final.fit(Xall,yall)
        dump(final,MODELS/'nam_svr_final.joblib')
        testfile=DATA/'test.csv'
        if not testfile.exists(): raise FileNotFoundError(testfile)
        test=pd.read_csv(testfile)
        if 'Id' not in test: raise ValueError('test.csv needs Id')
        prediction=np.exp(final.predict(test.drop(columns=['Id','SalePrice'],errors='ignore')))
        submission=pd.DataFrame({'Id':test.Id,'SalePrice':prediction})
        path=CHALLENGE/'results'/'submissions'/'submission_Nam_SVR.csv'
        submission.to_csv(path,index=False)
        print('Selected',chosen[0], '| holdout RMSE log',round(rmse,5),'| MAE',round(mae,2),'| output',path)
        return report
    if stage=='ALL':
        for s in ('EDA','V0','V1','V2','V3','V4','V5'):run_stage(s)
        return
    raise ValueError('Stage should be EDA, V0, V1, V2, V3, V4, V5 or ALL')

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Nam SVR experiment runner')
    p.add_argument('--stage',choices=['EDA','V0','V1','V2','V3','V4','V5','ALL'],required=True)
    args=p.parse_args();run_stage(args.stage)
