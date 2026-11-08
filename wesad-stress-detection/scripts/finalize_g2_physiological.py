"""Package a complete single-environment physiological run without altering features."""
import argparse
import hashlib
from pathlib import Path
import shutil
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd
from wesad.data import SUBJECTS
from wesad.feature_packages import read_package, package_report
from wesad.utils import read_json, write_json, file_hash, resolve_path


def finalize(source, output, contract_path, raw_root, report_path):
    if output.exists():
        raise ValueError('Use a new output directory; keep previous evidence.')
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError('Output must be outside the source package')
    contract = read_json(contract_path)
    config = read_json(source / 'config.json')
    if config['processing']['feature_contract_sha256'] != file_hash(contract_path):
        raise ValueError('Source contract hash differs')
    status = read_json(source / 'run_status.json')
    if (status['status'] != 'complete' or set(status['subjects']) != set(SUBJECTS)
            or len(status['subjects']) != len(SUBJECTS)):
        raise ValueError('A complete unique 15-subject run is required')
    for name, digest in read_json(source / 'checksums.json').items():
        path = (source / name).resolve()
        if not path.is_relative_to(source.resolve()) or file_hash(path) != digest:
            raise ValueError(f'Invalid source checksum: {name}')
    for subject in SUBJECTS:
        manifest = read_json(source / f'{subject}.json')
        if file_hash(raw_root / subject / f'{subject}.pkl') != manifest['source_sha256']:
            raise ValueError(f'Raw hash mismatch: {subject}')
    shutil.copytree(source, output)
    evidence = output / 'producer_evidence'
    evidence.mkdir()
    # Preserve original generated schema/checksums before publishing the full
    # existing contract as package metadata; CSV/manifest/config stay unchanged.
    for name in ('feature_schema.json', 'checksums.json', 'run_status.json'):
        shutil.copyfile(source / name, evidence / name)
    shutil.copyfile(contract_path, output / 'physiological_contract.json')
    write_json(output / 'feature_schema.json', {
        **contract, 'processing_signature': config['signature'],
        'package_status': 'complete_15_subjects',
        'status': 'feature_delivery_complete', 'ready_for_extraction': True,
        'readiness': {**contract.get('readiness', {}),
            'nhu_g2_runtime_integration_confirmed': True,
            'full_15_subject_package_ready': True,
            'preprocessing_provider_configured': True,
            'production_raw_extraction_ready': True,
            'meaning': 'Locked baseline provider executed for all 15 subjects; delivery validated. Physiological validity not independently established.'},
        'description': 'Full locked physiological contract attached by packaging; feature CSVs unchanged.'})
    names = ['all_subjects.csv', 'feature_schema.json', *[f'{s}.csv' for s in status['subjects']]]
    write_json(output / 'checksums.json', {n: file_hash(output / n) for n in names})
    frame, snapshot, manifests = read_package(output, contract)
    report = package_report(output, contract, complete=True)
    write_json(output / 'run_status.json', {**status, 'scope': 'complete_15_subjects'})
    quality, by_label = [], []
    keys = frame[contract['metadata_columns']].copy()
    for group, columns in contract['sensor_groups'].items():
        keys[group + '_all_features_present'] = frame[columns].notna().all(axis=1)
    keys.to_csv(output / 'common_windows.csv', index=False)
    for subject in SUBJECTS:
        part = frame.loc[frame.subject_id.eq(subject)]
        missing = part.wrist_bvp_pulse_rate_mean_bpm.isna()
        quality.append({'subject_id': subject, **manifests[subject]['extraction_qa'],
                        'n_bvp_missing': int(missing.sum())})
        for label, group in part.groupby('label_original'):
            by_label.append({'subject_id': subject, 'label_original': int(label),
                             'n_windows': len(group),
                             'n_bvp_missing': int(group.wrist_bvp_pulse_rate_mean_bpm.isna().sum())})
    pd.DataFrame(quality).to_csv(output / 'subject_quality.csv', index=False)
    pd.DataFrame(by_label).to_csv(output / 'bvp_by_label.csv', index=False)
    for subject in SUBJECTS:
        if file_hash(source / f'{subject}.csv') != file_hash(output / f'{subject}.csv'):
            raise ValueError('Packaging changed feature bytes')
    report.update({'scope': 'complete_15_subjects_feature_delivery',
        'delivery_complete': True, 'group_G2_acceptance': False,
        'nhu_runtime_integration_confirmed': True,
        'raw_hashes_verified': True, 'single_source_environment': True,
        'producer_signature': snapshot['signature'],
        'csv_bytes_unchanged': True, 'source_directory': str(source),
        'common_signal_eligible_windows': len(keys),
        'common_all_features_present': int(keys.combined_all_features_present.sum()),
        'n_bvp_missing': int(frame.wrist_bvp_pulse_rate_mean_bpm.isna().sum()),
        'window_policy': 'Same original keys for chest/wrist/combined; retain reasoned NaN. No complete-case exclusion or imputation applied.',
        'limits': ['BVP heuristics unchanged; signal validity not independently established.',
                   'Delivery completion does not approve all group members tasks or G3 readiness.']})
    write_json(output / 'delivery_report.json', report)
    write_json(output / 'handoff_checksums.json', {
        str(p.relative_to(output)).replace('\\', '/'): file_hash(p)
        for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'handoff_checksums.json'})
    archive = output.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file():
                z.write(p, (Path(output.name) / p.relative_to(output)).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('ZIP verification failed')
        for name, digest in read_json(output / 'handoff_checksums.json').items():
            actual = hashlib.sha256(z.read(f'{output.name}/{name}')).hexdigest()
            if actual != digest:
                raise ValueError(f'ZIP content hash mismatch: {name}')
    report['zip_sha256'] = file_hash(archive)
    report['zip_path'] = str(archive)
    write_json(report_path, report)
    print(f"Completed delivery: {len(frame)} windows, {len(SUBJECTS)} subjects")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--contract', default='configs/feature_schema.json')
    parser.add_argument('--raw-root', required=True)
    parser.add_argument('--report', required=True)
    a = parser.parse_args()
    finalize(*[resolve_path(p) for p in (a.source, a.output, a.contract, a.raw_root, a.report)])
