"""Reject subject leakage and accidental mixing with G3 defaults."""
import copy
import unittest
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
from unittest.mock import patch
import pandas as pd
from sklearn.model_selection import ParameterGrid
from wesad.data import SUBJECTS
from wesad.g4_nhu import check_splits, preflight
from wesad.models import build_pipeline
from wesad.utils import read_config

class G4NhuTests(unittest.TestCase):
    def detail(self):
        train=list(SUBJECTS[1:])
        blocks=[train[::3],train[1::3],train[2::3]]
        return {'outer_train_subjects':train,'outer_test_subjects':['S2'],
                'inner_splits':[{'train_subjects':[s for s in train if s not in b],
                                 'validation_subjects':b} for b in blocks]}

    def test_inner_leakage_and_missing_validation_are_rejected(self):
        check_splits(self.detail(),'S2')
        for mutation in ('outer','overlap','missing'):
            d=copy.deepcopy(self.detail())
            if mutation=='outer': d['inner_splits'][0]['validation_subjects'].append('S2')
            elif mutation=='overlap': d['inner_splits'][0]['train_subjects'].append(d['inner_splits'][0]['validation_subjects'][0])
            else: d['inner_splits'][0]['validation_subjects'].pop()
            with self.assertRaises(ValueError): check_splits(d,'S2')

    def test_grid_candidates_can_be_applied_to_train_fold_pipelines(self):
        config=read_config('configs/g4_nhu_tuning.yaml')
        for model in config['models']:
            candidates=list(ParameterGrid(config['search_spaces'][model]))
            self.assertEqual(len(candidates),4)
            pipe=build_pipeline(model,config)
            self.assertEqual(pipe['imputer'].strategy,'median')
            self.assertTrue(pipe['imputer'].keep_empty_features)
            self.assertEqual('scaler' in pipe.named_steps,model=='mlp')
            for params in candidates: pipe.set_params(**params)

    def test_g3_factory_and_wrong_dataset_checksum_rejected(self):
        config=read_config('configs/g4_nhu_tuning.yaml')
        for key,value in [('model_factories',{'rf':'wesad.models_rf_mlp:build_default_pipeline'}),
                          ('feature_sha256','invalid'),('nested',False)]:
            bad=copy.deepcopy(config);bad[key]=value
            with self.assertRaises(ValueError): preflight(bad)

    def test_mlp_loss_is_recorded_from_best_outer_train_refit(self):
        from wesad.evaluation import run_experiment
        from wesad.utils import PROJECT_ROOT, read_json
        from threadpoolctl import threadpool_limits
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT/'outputs') as tmp:
            root=Path(tmp)
            rows=[{'subject_id':s,'window_id':f'{s}_{i}','start_s':i*5,'end_s':i*5+60,
                   'label_original':i%3+1,'quality_flag':'fixture','x':float(i)}
                  for s in ('S2','S3','S4','S5') for i in range(12)]
            pd.DataFrame(rows).to_csv(root/'table.csv',index=False)
            config={'features':str(root/'table.csv'),'output_dir':str(root/'run'),
                    'models':['mlp'],'tasks':['binary'],'nested':True,'inner_splits':3,
                    'record_mlp_loss':True,'model_params':{'mlp':{'max_iter':3}},
                    'search_spaces':{'mlp':{'model__hidden_layer_sizes':[[4]]}}}
            with patch('wesad.evaluation.provenance',return_value={}), threadpool_limits(1), redirect_stdout(io.StringIO()):
                run_experiment(config,allow_partial=True,outer_subjects=['S2'])
            detail=read_json(root/'run/folds/binary/mlp/S2/details.json')
            opt=detail['mlp_optimization']
            self.assertEqual(opt['n_iter'],3)
            self.assertEqual(len(opt['loss_curve']),3)
            self.assertTrue(opt['reached_max_iter'])
            self.assertTrue(detail['warnings'])
            self.assertNotIn('S2',detail['outer_train_subjects'])

if __name__=='__main__': unittest.main()
