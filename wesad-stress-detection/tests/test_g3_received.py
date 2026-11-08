"""Legacy admission is explicit and still rejects tampered OOF data and leakage."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from wesad.data import SUBJECTS
from wesad.evaluation import run_experiment
from wesad.models import build_pipeline
from wesad.g3_received import check_legacy_settings,review_source,verify_legacy
from wesad.utils import read_json,write_json,file_hash
import test_g3_baselines as fixtures


class ReceivedG3Tests(unittest.TestCase):
    def setUp(self):
        helper=fixtures.G3BaselineTests();helper.setUp()
        self.addCleanup(helper.temp.cleanup)
        self.root=helper.root;self.reference=helper.config
        self.settings={'expected_subjects':list(SUBJECTS),'models':['knn','svm'],'tasks':['multiclass','binary'],
          'seed':42,'sensor_set':'combined_physiological68','normalization':'global_train','balance':False,'nested':False,
          'variance_threshold':None,'model_factories':{'knn':'wesad.models_neighbors_svm:build_pipeline','svm':'wesad.models_neighbors_svm:build_pipeline'},'model_params':{}}
        self.config={**self.settings,**{k:self.reference[k] for k in ('features','schema','output_dir')}}

    def run_fixture(self):
        # Use installed default estimators, not received source code, to build
        # an old-format producer fixture. The source-review gate is tested separately.
        with patch('wesad.evaluation.provenance',return_value={'git_commit':'fixture','git_dirty':True}),patch('wesad.evaluation.build_pipeline',side_effect=lambda n,c:build_pipeline(n,{'seed':42})):
            run_experiment(self.config)

    def verify(self):
        with patch('wesad.g3_received.review_source',return_value={'runtime_source_snapshot_bound_to_result_hash':False}):
            return verify_legacy(self.root/'run',self.root,self.reference)

    def test_legacy_result_verified_without_rewriting_identity(self):
        self.run_fixture();before=(self.root/'run/config.json').read_bytes()
        p,m,r=self.verify()
        self.assertEqual((len(p),len(m)),(360,60))
        self.assertTrue(r['producer']['git_dirty'])
        self.assertEqual((self.root/'run/config.json').read_bytes(),before)

    def test_wrong_prediction_truth_rejected(self):
        self.run_fixture();path=self.root/'run/predictions.csv'
        f=pd.read_csv(path);f.loc[0,'y_true']=3 if f.loc[0,'y_true']==1 else 1;f.to_csv(path,index=False)
        with self.assertRaisesRegex(ValueError,'truth'):
            self.verify()

    def test_leakage_not_excused_by_updated_checksum(self):
        self.run_fixture();folder=self.root/'run/folds/multiclass/knn/S2'
        detail=read_json(folder/'details.json');detail['outer_train_subjects'][0]='S2';write_json(folder/'details.json',detail)
        done=read_json(folder/'complete.json');done['files']['details.json']=file_hash(folder/'details.json');write_json(folder/'complete.json',done)
        with self.assertRaisesRegex(ValueError,'leakage'):
            self.verify()

    def test_tuning_or_unknown_factory_not_admitted(self):
        for change in ({'nested':True},{'model_params':{'knn':{'n_neighbors':3}}},
                       {'model_factories':{'knn':'arbitrary:factory','svm':'arbitrary:factory'}}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                check_legacy_settings({**self.settings,**change})

    def test_factory_source_is_pinned_not_executed(self):
        target=self.root/'src/wesad/models_neighbors_svm.py';target.parent.mkdir(parents=True)
        target.write_text("raise RuntimeError('Do not execute received source')")
        with self.assertRaisesRegex(ValueError,'Unreviewed factory'):
            review_source(self.root)


if __name__=='__main__':unittest.main()
