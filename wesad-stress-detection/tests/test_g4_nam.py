"""Synthetic integration tests only; never presented as WESAD model results."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
import pandas as pd

from wesad import g4_nam as n
from wesad.g4 import merge_tuned, verify_run
from wesad.utils import read_config, read_json, write_json, file_hash, resolve_path
from wesad.data import SUBJECTS


class NamG4Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = read_config('configs/g4_nam_knn_svm_tuning.yaml')
        schema = read_json(resolve_path('configs/feature_schema.json'))
        rng = np.random.default_rng(42); rows=[]
        for i,s in enumerate(SUBJECTS):
            for j in range(6):
                label=[1,1,2,2,3,3][j]
                r=dict(zip(schema['feature_columns'], rng.normal(label, .4, 68)))
                r.update(subject_id=s,window_id=f'{s}_{j}',start_s=j*5,end_s=j*5+60,
                         label_original=label,label_binary=int(label==2),label_fraction=1.0,quality_flag='synthetic')
                rows.append(r)
        frame=pd.DataFrame(rows)[schema['output_column_order']]
        table=self.root/'all_subjects.csv';frame.to_csv(table,index=False)
        write_json(self.root/'feature_schema.json',schema)
        self.config.update(features=str(table),schema=str(self.root/'feature_schema.json'),
            feature_sha256=file_hash(table),output_dir=str(self.root/'run'),outer_subjects=['S2'],run_kind='main_shard')
        for target,value in [('DATA_SHA',file_hash(table)),('N_ROWS',len(frame))]:
            p=patch.object(n,target,value);p.start();self.addCleanup(p.stop)

    def run_one(self):
        result=n.run(self.config)
        self.assertEqual((result['folds'],result['predictions']),(4,24))
        return Path(self.config['output_dir'])

    def test_nested_run_resume_and_common_merger_accepts_real_artifacts(self):
        out=self.run_one()
        before={str(p):file_hash(p) for p in (out/'folds').rglob('*') if p.is_file()}
        n.run(self.config,resume=True)
        after={str(p):file_hash(p) for p in (out/'folds').rglob('*') if p.is_file()}
        self.assertEqual(before,after)
        merged=merge_tuned([out],self.root/'merged',self.config)
        self.assertEqual(merged['n_completed_outer_folds'],4)
        self.assertEqual(merged['status'],'waiting_for_tuned_packages')
        self.assertFalse(read_json(out/'verification.json')['complete_15_subjects'])

    def test_factory_grid_and_data_lock_rejected_before_training(self):
        for edit in [dict(model_factories={'knn':'wesad.models_neighbors_svm:build_pipeline'}),
                     dict(feature_sha256='0'*64),dict(balance=True)]:
            with self.subTest(edit=edit),self.assertRaises(ValueError):n.preflight({**self.config,**edit})
        bad=copy.deepcopy(self.config);bad['search_spaces']['svm'][0]['model__C']=[999]
        with self.assertRaises(ValueError):n.preflight(bad)

    def test_scope_change_cannot_resume_or_relabel_old_output(self):
        self.run_one()
        bad={**self.config,'outer_subjects':['S3']}
        with self.assertRaisesRegex(ValueError,'Config/scope'):n.run(bad,resume=True)
        with self.assertRaisesRegex(ValueError,'Config/scope'):n.verify(bad)

    def test_snapshot_tampering_rejected(self):
        out=self.run_one()
        with zipfile.ZipFile(out/'source_snapshot.zip','a') as z:z.writestr('extra.py','oops')
        with self.assertRaisesRegex(ValueError,'snapshot inventory'):n.verify(self.config)

    def test_inner_outer_leakage_rejected_even_after_rehash(self):
        out=self.run_one();folder=out/'folds/multiclass/knn/S2'
        d=read_json(folder/'details.json');d['inner_splits'][0]['validation_subjects'].append('S2')
        write_json(folder/'details.json',d)
        done=read_json(folder/'complete.json');done['files']['details.json']=file_hash(folder/'details.json');write_json(folder/'complete.json',done)
        with self.assertRaisesRegex(ValueError,'Inner/test'):n.verify(self.config)

    def test_quick_is_never_packaged_or_merged_as_main(self):
        c={**self.config,'run_kind':'quick','output_dir':str(self.root/'quick_check')}
        self.assertEqual(n.run(c)['status'],'quick_completed_not_main_verified')
        with self.assertRaisesRegex(ValueError,'Quick'):n.verify(c)
        with self.assertRaises(ValueError):merge_tuned([self.root/'quick_check'],self.root/'badmerge',c)

    def test_full_scope_requires_all_people(self):
        with self.assertRaisesRegex(ValueError,'Run scope'):n.preflight({**self.config,'run_kind':'main'})

    def test_full_15_subject_run_has_60_folds_and_main_verification(self):
        c={**self.config,'outer_subjects':list(SUBJECTS),'run_kind':'main'}
        proof=n.run(c)
        self.assertEqual((proof['folds'],proof['predictions']),(60,360))
        self.assertTrue(proof['complete_15_subjects'])
        self.assertEqual(read_json(Path(c['output_dir'])/'summary.json')['status'],'complete')

    def test_report_uses_paired_people_and_packages_source_and_figures(self):
        out=self.run_one(); ref=[]
        for t in self.config['tasks']:
            for m in self.config['models']:
                for s in SUBJECTS:ref.append(dict(task=t,model=m,subject_id=s,macro_f1=.5,balanced_accuracy=.6))
        path=self.root/'g3.csv';pd.DataFrame(ref).to_csv(path,index=False)
        result=n.report(self.config,str(path))
        pair=pd.read_csv(out/'paired_g3_g4_by_subject.csv')
        self.assertEqual(set(pair.subject_id),{'S2'});self.assertEqual(len(pair),4)
        self.assertEqual(len(list((out/'figures').glob('*.svg'))),5)
        with zipfile.ZipFile(result['zip_path']) as z:
            self.assertIsNone(z.testzip())
            self.assertIn('run/source_snapshot.zip',z.namelist())
            self.assertIn('run/handoff_checksums.json',z.namelist())


if __name__=='__main__':unittest.main()
