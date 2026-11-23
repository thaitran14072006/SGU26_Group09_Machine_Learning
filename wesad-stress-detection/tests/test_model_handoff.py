"""Kiểm tra cách lấy điểm LOSO để tránh gộp nhầm các phiên bản."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('check_model_handoff', ROOT / 'scripts/check_model_handoff.py')
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


class ModelHandoffTests(unittest.TestCase):
    def write_rows(self, folder, rows):
        path = Path(folder) / 'results.csv'
        with path.open('w', encoding='utf8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['task', 'subject_id', 'variant', 'macro_f1', 'balanced_accuracy'])
            writer.writeheader()
            writer.writerows(rows)
        return path

    def rows(self, variant, score):
        return [dict(task='binary', subject_id=s, variant=variant, macro_f1=score,
                     balanced_accuracy=score) for s in sorted(handoff.SUBJECTS)]

    def test_variant_must_be_chosen_before_averaging(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_rows(folder, self.rows('full68', .8) + self.rows('selected', .6))
            with self.assertRaises(ValueError):
                handoff.loso_metrics(path, 'binary')
            metrics = handoff.loso_metrics(path, 'binary', {'variant': 'selected'})
            self.assertAlmostEqual(metrics['macro_f1'], .6)
            self.assertAlmostEqual(metrics['balanced_accuracy'], .6)
            self.assertAlmostEqual(metrics['macro_f1_sd'], 0)

    def test_duplicate_subject_cannot_replace_missing_subject(self):
        rows = self.rows('main', .8)
        rows[-1]['subject_id'] = rows[0]['subject_id']
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_rows(folder, rows)
            with self.assertRaises(ValueError):
                handoff.loso_metrics(path, 'binary')

    def test_nonfinite_metric_is_rejected(self):
        rows = self.rows('main', .8)
        rows[0]['balanced_accuracy'] = 'nan'
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_rows(folder, rows)
            with self.assertRaises(ValueError):
                handoff.loso_metrics(path, 'binary')


if __name__ == '__main__':
    unittest.main()
