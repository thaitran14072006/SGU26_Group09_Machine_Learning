"""Kiểm tra các lỗi dễ làm sai bảng tổng hợp model."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import model_handoff_common as common
import build_model_summary as summary
import export_dummy_model as dummy


class SubmissionTests(unittest.TestCase):
    def rows(self, variant, score):
        return [dict(task='binary', subject_id=s, variant=variant, macro_f1=score,
                     balanced_accuracy=score) for s in sorted(common.SUBJECTS)]

    def test_cannot_mix_variants(self):
        rows = self.rows('full68', .8) + self.rows('selected', .6)
        with self.assertRaises(ValueError):
            common.diem_theo_nguoi(rows)
        selected = common.loc_dong(rows, 'binary', {'variant': 'selected'})
        self.assertAlmostEqual(common.diem_theo_nguoi(selected)['macro_f1'], .6)

    def test_missing_subject_is_rejected(self):
        rows = self.rows('main', .8)
        rows[-1]['subject_id'] = rows[0]['subject_id']
        with self.assertRaises(ValueError):
            common.diem_theo_nguoi(rows)

    def test_nonfinite_score_is_rejected(self):
        rows = self.rows('main', .8)
        rows[0]['balanced_accuracy'] = 'nan'
        with self.assertRaises(ValueError):
            common.diem_theo_nguoi(rows)

    def test_changed_summary_score_is_reported(self):
        row = dict.fromkeys(summary.FIELDS, '')
        row.update(algorithm='rf', task='binary', model_path='rf/binary.joblib',
                   experiment_id='EXP-005', macro_f1=.8, balanced_accuracy=.85,
                   macro_f1_sd=.1, balanced_accuracy_sd=.1, loso_source='results.csv')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model_summary.csv'
            with path.open('w', encoding='utf8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=summary.FIELDS)
                writer.writeheader()
                writer.writerow({**row, 'macro_f1': .99})
            errors = summary.compare_summary(path, [row])
            self.assertEqual(len(errors), 1)
            self.assertIn('macro_f1 lệch', errors[0][3])

    def test_dummy_cannot_overwrite_existing_model(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'dummy'
            out.mkdir()
            path = out / 'binary.joblib'
            path.write_bytes(b'giu nguyen')
            with patch.object(dummy, 'GROUP', Path(folder)):
                with self.assertRaises(FileExistsError):
                    dummy.export('binary')
            self.assertEqual(path.read_bytes(), b'giu nguyen')

    def test_model_below_dummy_is_reported(self):
        rows = [dict(algorithm='knn', task='binary', score_check='matched',
                     macro_f1=.1, balanced_accuracy=.4)]
        errors = summary.compare_dummy(rows)
        self.assertEqual(len(errors), 2)


if __name__ == '__main__':
    unittest.main()
