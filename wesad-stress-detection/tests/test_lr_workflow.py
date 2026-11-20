"""Kiểm tra ranh giới CV, quy tắc chuyển bước và biến đổi dữ liệu LR."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.base import clone
from wesad.lr_workflow import (CorrelationSelector, WeightedLR, build,
    grouped_splits, plan, prerequisites, winner)


class WorkflowTests(unittest.TestCase):
    def test_corr_transform_does_not_learn_from_test(self):
        x=np.array([[0,0,2],[1,1,1],[2,2,2],[3,3,1]],dtype=float)
        selector=CorrelationSelector(0.9).fit(x)
        before=selector.get_support()
        selector.transform(np.array([[8,-100,1]],dtype=float))
        np.testing.assert_array_equal(selector.get_support(),before)
        self.assertEqual(before.tolist(),[True,False,True])

    def test_weights_use_only_fit_labels_and_clone(self):
        x=np.array([[0],[1],[2],[3],[4],[5]],dtype=float)
        model=clone(WeightedLR(weight_policy='sqrt_balanced')).fit(x,[0,0,0,0,1,1])
        self.assertAlmostEqual(model.fit_class_weights_[0],np.sqrt(6/8))
        self.assertAlmostEqual(model.fit_class_weights_[1],np.sqrt(6/4))
        self.assertEqual(model.predict(x).shape,(6,))

    def test_inner_groups_exclude_outer_subject(self):
        g=pd.Series(['S3','S3','S4','S4','S5','S5'])
        _,evidence=grouped_splits(np.arange(6).reshape(-1,1),np.array([0,1]*3),g,'S2')
        self.assertEqual(len(evidence),3)
        for fold in evidence:
            self.assertTrue(set(fold['train_subjects']).isdisjoint(fold['validation_subjects']))
        with self.assertRaises(ValueError):
            grouped_splits(np.arange(6).reshape(-1,1),np.array([0,1]*3),g,'S3')

    def test_winner_ignores_outer_test_scores(self):
        records=[{'inner_macro_f1':0.6,'outer_macro_f1':0.99},
                 {'inner_macro_f1':0.7,'outer_macro_f1':0.1}]
        self.assertIs(winner(records),records[1])

    def test_tuning_requires_both_previous_stages(self):
        with patch('wesad.lr_workflow.freeze',side_effect=ValueError('incomplete')) as freeze:
            with self.assertRaises(ValueError):
                prerequisites('tuning',plan())
            freeze.assert_called_once_with('feature_selection',plan())

    def test_balancing_refits_selection_inside_pipeline(self):
        cfg=plan()
        spec=next(s for s in cfg['experiments'] if s['run_id']=='BAL-02')
        pipe,grid,_=build(spec,cfg,{'method':'anova','selection_params':{'selection__k':16}})
        self.assertEqual(list(pipe.named_steps),['imputer','scaler','selection','model'])
        self.assertFalse(hasattr(pipe['selection'],'scores_'))
        self.assertEqual(pipe['selection'].k,16)
        self.assertEqual(grid,{})


if __name__=='__main__':
    unittest.main()
