"""Default protocol and OOF integrity, including tampering and incomplete batches."""
import tempfile
import zipfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

from wesad.data import SUBJECTS
from wesad.g3 import check_config, run_defaults, verify_run, merge_runs
from wesad.utils import read_config, read_json, write_json, file_hash


class G3BaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.config=read_config('configs/defaults.yaml')
        # Synthetic unit fixtures must not require the ignored group data ZIP.
        schema=read_json(Path(__file__).resolve().parents[1]/'configs/feature_schema.json')
        rng=np.random.default_rng(42)
        rows=[]
        for j,subject in enumerate(SUBJECTS):
            for i in range(6):
                label=i%3+1
                row=dict(zip(schema['metadata_columns'],[subject,f'{subject}_{i*3500}',i*5.,i*5.+60.,label,int(label==2),1.,'test_fixture']))
                row.update(dict(zip(schema['feature_columns'],rng.normal(size=68))))
                # All-missing features must not change width across held-out folds.
                row[schema['feature_columns'][-1]]=np.nan
                rows.append(row)
        pd.DataFrame(rows,columns=schema['output_column_order']).to_csv(self.root/'features.csv',index=False)
        write_json(self.root/'schema.json',schema)
        self.config.update(features=str(self.root/'features.csv'),schema=str(self.root/'schema.json'),
             feature_sha256=file_hash(self.root/'features.csv'),output_dir=str(self.root/'run'),models=['dummy'],run_kind='main')

    def run_batch(self,quick=False):
        if quick:
            self.config.update(output_dir=str(self.root/'quick_run'),run_kind='quick')
        with patch('wesad.evaluation.provenance',return_value={'git_commit':'fixture','git_dirty':True}):
            return run_defaults(self.config,quick_subject='S2' if quick else None)

    def test_feature_only_matrix_and_locked_checksum(self):
        frame,columns=check_config(self.config)
        self.assertEqual(len(columns),68)
        self.assertFalse(set(columns)&{'subject_id','label_original','quality_flag','start_s'})
        with self.assertRaisesRegex(ValueError,'checksum'):
            check_config({**self.config,'feature_sha256':'0'*64})
        with self.assertRaisesRegex(ValueError,'source scope'):
            check_config({**self.config,'source_paths':['src/wesad/g3.py']})

    def test_no_tuned_defaults_or_balancing(self):
        for changes in ({'nested':True},{'balance':True},{'variance_threshold':0.},
                        {'model_params':{'lr':{'max_iter':1000}}}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                check_config({**self.config,**changes})

    def test_missing_train_class_fails_explicitly(self):
        f=pd.read_csv(self.config['features'])
        f.loc[(f.subject_id!='S2')&(f.label_original==3),'label_original']=1
        f['label_binary']=f.label_original.eq(2).astype(int)
        f.to_csv(self.config['features'],index=False)
        self.config['feature_sha256']=file_hash(self.config['features'])
        with self.assertRaisesRegex(ValueError,'train missing class'):
            check_config(self.config)

    def test_complete_oof_and_pending_group_models(self):
        result,verified=self.run_batch()
        self.assertEqual(result['status'],'complete')
        self.assertEqual(verified['n_folds'],30)
        self.assertEqual(verified['n_predictions'],180)
        merged=merge_runs([self.root/'run'],self.root/'group',self.config)
        self.assertEqual(merged['status'],'waiting_for_models')
        self.assertEqual(set(merged['missing_models']),{'lr','knn','svm','rf','mlp'})
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            merge_runs([self.root/'run',self.root/'run'],self.root/'duplicates',self.config)

    def test_quick_cannot_be_promoted_or_merged(self):
        self.run_batch(quick=True)
        with self.assertRaisesRegex(ValueError,'Quick/partial'):
            verify_run(self.root/'quick_run',self.config)
        with self.assertRaisesRegex(ValueError,'Quick'):
            merge_runs([self.root/'quick_run'],self.root/'group',self.config)

    def test_wrong_truth_and_summary_are_rejected(self):
        self.run_batch()
        path=self.root/'run/predictions.csv'
        original=path.read_bytes()
        f=pd.read_csv(path);f.loc[0,'y_true']=3 if f.loc[0,'y_true']==1 else 1
        f.to_csv(path,index=False)
        with self.assertRaisesRegex(ValueError,'true labels'):
            verify_run(self.root/'run',self.config)
        path.write_bytes(original)
        summary=read_json(self.root/'run/summary.json')
        summary['results'][0]['macro_f1_mean']=.999
        write_json(self.root/'run/summary.json',summary)
        with self.assertRaisesRegex(ValueError,'Summary mean'):
            verify_run(self.root/'run',self.config)

    def test_train_test_leakage_rejected_even_with_updated_hash(self):
        self.run_batch()
        folder=self.root/'run/folds/multiclass/dummy/S2'
        detail=read_json(folder/'details.json');detail['outer_train_subjects'][0]='S2'
        write_json(folder/'details.json',detail)
        done=read_json(folder/'complete.json');done['files']['details.json']=file_hash(folder/'details.json')
        write_json(folder/'complete.json',done)
        with self.assertRaisesRegex(ValueError,'leakage'):
            verify_run(self.root/'run',self.config)

    def test_source_line_endings_are_portable_but_content_changes_fail(self):
        self.run_batch()
        path=self.root/'run/source_snapshot.zip'
        with zipfile.ZipFile(path) as z:
            files={n:z.read(n).replace(b'\r\n',b'\n').replace(b'\n',b'\r\n') for n in z.namelist()}
        with zipfile.ZipFile(path,'w') as z:
            for n,data in files.items():z.writestr(n,data)
        self.assertEqual(verify_run(self.root/'run',self.config)[0]['status'],'passed')
        files['src/wesad/g3.py']+=b'\r\n# Changed source evidence\r\n'
        with zipfile.ZipFile(path,'w') as z:
            for n,data in files.items():z.writestr(n,data)
        with self.assertRaisesRegex(ValueError,'Source snapshot code'):
            verify_run(self.root/'run',self.config)


if __name__=='__main__':
    unittest.main()
