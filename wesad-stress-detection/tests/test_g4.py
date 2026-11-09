"""G4 train-only sampling, weighted baseline, and nested result admission."""
import copy
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import test_g3_baselines as fixtures
from wesad.evaluation import run_experiment
from wesad.g4 import (balanced_config,audit_resampling,candidates,verify_run,merge_tuned)
from wesad.utils import read_config,read_json,write_json,file_hash,resolve_path


class G4Tests(unittest.TestCase):
    def setUp(self):
        helper=fixtures.G3BaselineTests();helper.setUp();self.addCleanup(helper.temp.cleanup)
        self.root=helper.root
        self.config=read_config('configs/g4_balanced_baseline.yaml')
        for key in ('features','schema','output_dir'):self.config[key]=helper.config[key]
        f=pd.read_csv(self.config['features'])
        f['label_original']=[1,1,1,1,2,3]*15;f['label_binary']=f.label_original.eq(2).astype(int)
        f.to_csv(self.config['features'],index=False)
        self.config.update(feature_sha256=file_hash(self.config['features']),run_kind='main')

    def run_fixture(self,nested=False,assigned=False):
        config=copy.deepcopy(self.config)
        if nested:
            config.update(protocol='g4_nested_physiological68_v1',models=['knn'],nested=True,balance=False,run_kind='main_shard',
              search_spaces={'knn':[{'model__n_neighbors':[3],'model__metric':['euclidean'],'model__weights':['uniform']},
                {'model__n_neighbors':[5],'model__metric':['manhattan'],'model__weights':['distance']},
                {'model__n_neighbors':[9],'model__metric':['euclidean'],'model__weights':['distance']},
                {'model__n_neighbors':[5],'model__metric':['euclidean'],'model__weights':['uniform']}]})
            if assigned:
                config.pop('run_kind');config['outer_subjects']=['S2']
        with patch('wesad.evaluation.provenance',return_value={'git_commit':'fixture','git_dirty':True}):
            run_experiment(config,outer_subjects=['S2'] if nested else None)
        out=Path(config['output_dir'])
        with zipfile.ZipFile(out/'source_snapshot.zip','w') as z:
            for name in config['source_paths']:z.write(resolve_path(name),name)
        if assigned:write_json(out/'verification.json',{'status':'passed','scope':'assigned_outer_subjects'})
        return out,config

    def test_balanced_baseline_rejects_tuning_or_silent_parameter_change(self):
        balanced_config(self.config)
        for change in ({'balance':False},{'nested':True},{'model_params':{'lr':{'max_iter':1000}}}):
            with self.subTest(change=change),self.assertRaises(ValueError):balanced_config({**self.config,**change})

    def test_sampler_changes_only_train_for_inner_and_outer(self):
        before=Path(self.config['features']).read_bytes()
        audit=audit_resampling(self.config,['S2'])
        self.assertEqual(len(audit),16)
        self.assertTrue(audit.passed.all())
        self.assertTrue((audit.train_rows_after>audit.train_rows_before).all())
        self.assertTrue((audit.validation_rows_before==audit.validation_rows_after).all())
        self.assertTrue((audit.test_rows_before==audit.test_rows_after).all())
        self.assertFalse(audit.train_subjects.str.split(',').apply(lambda s:'S2' in s).any())
        self.assertEqual(Path(self.config['features']).read_bytes(),before)

    def test_weighted_lr_full_oof_and_wrong_labels(self):
        out,_=self.run_fixture()
        pred,metrics,params,proof=verify_run(out,self.config)
        self.assertEqual((len(pred),len(metrics)),(180,30));self.assertFalse(params)
        path=out/'predictions.csv';f=pd.read_csv(path);f.loc[0,'y_true']=3;f.to_csv(path,index=False)
        with self.assertRaisesRegex(ValueError,'labels'):verify_run(out,self.config)

    def test_nested_shard_has_bestparams_without_claiming_full_group(self):
        out,config=self.run_fixture(nested=True)
        _,_,best,_=verify_run(out,self.config,kind='nested',allow_partial=True)
        self.assertEqual(len(best),2)
        report=merge_tuned([out],self.root/'merged',self.config)
        self.assertEqual(report['status'],'waiting_for_tuned_packages')
        self.assertEqual(report['n_completed_outer_folds'],2)
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            merge_tuned([out,out],self.root/'duplicates',self.config)

    def test_inner_validation_test_leakage_rejected_even_when_rehashed(self):
        out,_=self.run_fixture(nested=True)
        folder=out/'folds/multiclass/knn/S2';details=read_json(folder/'details.json')
        details['inner_splits'][0]['validation_subjects'].append('S2');write_json(folder/'details.json',details)
        done=read_json(folder/'complete.json');done['files']['details.json']=file_hash(folder/'details.json');write_json(folder/'complete.json',done)
        with self.assertRaisesRegex(ValueError,'Inner/test'):
            verify_run(out,self.config,kind='nested',allow_partial=True)

    def test_bestparams_must_follow_inner_scores(self):
        out,_=self.run_fixture(nested=True)
        folder=out/'folds/multiclass/knn/S2';details=read_json(folder/'details.json')
        details['best_params']['model__n_neighbors']=99;write_json(folder/'details.json',details)
        done=read_json(folder/'complete.json');done['files']['details.json']=file_hash(folder/'details.json');write_json(folder/'complete.json',done)
        with self.assertRaisesRegex(ValueError,'Best params'):
            verify_run(out,self.config,kind='nested',allow_partial=True)

    def test_candidate_budget_and_domain_are_enforced(self):
        _,config=self.run_fixture(nested=True)
        self.assertEqual(len(candidates(config,'knn')),4)
        config['search_spaces']['knn'][0]['model__n_neighbors']=[99]
        with self.assertRaisesRegex(ValueError,'domain'):candidates(config,'knn')
        config['search_spaces']['knn']=config['search_spaces']['knn'][:2]
        with self.assertRaisesRegex(ValueError,'four distinct'):candidates(config,'knn')

    def test_no_received_packages_yields_waiting_not_complete(self):
        result=merge_tuned([],self.root/'waiting',self.config)
        self.assertEqual(result['status'],'waiting_for_tuned_packages')
        self.assertEqual(len(result['missing_folds']),120)
        self.assertFalse((self.root/'waiting/predictions.csv').exists())

    def test_declared_assigned_outer_batch_supported_but_quick_rejected(self):
        out,_=self.run_fixture(nested=True,assigned=True)
        _,_,best,proof=verify_run(out,self.config,kind='nested',allow_partial=True)
        self.assertEqual(len(best),2);self.assertTrue(proof['training_core_sha256'])
        write_json(out/'verification.json',{'status':'passed','scope':'quick'})
        with self.assertRaisesRegex(ValueError,'Partial input'):
            verify_run(out,self.config,kind='nested',allow_partial=True)


if __name__=='__main__':unittest.main()
