"""Dependency-free integrity and completeness check for the selected package."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import struct

ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    required=['README.md','requirements.txt','SHA256SUMS.txt','scripts/run_all_figures.py',
              'scripts/frozen_parameters.json','scripts/m11_protocol.json','scripts/citysim_core.py',
              'scripts/m11_models.py','scripts/m11_data.py','scripts/run_m11_calibration.py',
              'scripts/run_m11_expanded_bound_audit.py','scripts/analyze_m11_results.py',
              'scripts/hefei_following_sample.csv','scripts/citysim_selected_following_pairs.csv',
              'outputs/fair_calibration_runs.csv','outputs/fair_calibration_convergence.csv',
              'outputs/recalibrated_baseline_metrics.csv','outputs/recalibrated_baseline_timeseries.csv',
              'outputs/recalibrated_disturbance_timeseries.csv','outputs/Fig22_normalized_values.csv',
              'outputs/M11/common_protocol/ga_convergence.csv','outputs/M11/common_protocol/ga_parameter_estimates.csv',
              'outputs/M11/common_protocol/ga_runs.csv','outputs/M11/source_hashes_and_cache_metadata.json',
              'outputs/M11/analysis/seed42_holdout_per_trajectory.csv','outputs/M11/analysis/robustness_audits.csv',
              'outputs/M11/analysis/Table_M11_common_protocol_comparison.csv',
              'outputs/M11/expanded_bound_audit/expanded_bound_summary.json']
    for i in range(9,25):
        name='Fig_21_recalibrated.tiff' if i==21 else (f'Fig_{i:02d}_recalibrated.png' if i in (15,16,22) else f'Fig_{i:02d}.png')
        required.append('outputs/'+name)
    required.extend('outputs/Fig_'+x+'.png' for x in ('C1','C2'))
    for name in required:assert (ROOT/name).is_file(),f'Missing: {name}'
    manifest={}
    for line in (ROOT/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        expected,name=line.split('  ',1);manifest[name]=expected
        assert (ROOT/name).is_file() and sha(ROOT/name)==expected,f'Checksum mismatch: {name}'
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()
            and '.git' not in p.relative_to(ROOT).parts
            and not any(x in p.relative_to(ROOT).parts for x in ('reproduced_outputs','external_data','__pycache__','.venv'))
            and not p.name.startswith('~$') and p.suffix not in ('.pyc','.nbc','.nbi')
            and p.name!='SHA256SUMS.txt'}
    assert actual==set(manifest),'Inventory differs from the released checksum set'
    for p in (ROOT/'scripts').glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    params=json.loads((ROOT/'scripts/frozen_parameters.json').read_text(encoding='utf-8'))
    assert params['table_c2_seed42_parameters']['seed']==42
    assert params['table_e1_seed42_citysim_parameters']['seed']==42
    kernel={}
    for node in ast.parse((ROOT/'scripts/citysim_core.py').read_text(encoding='utf-8')).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            if node.targets[0].id in ('SHORT_RANGE_THRESHOLD_M','HEFEI_TIDM_PARAMETERS','HEFEI_TFVD_PARAMETERS'):
                kernel[node.targets[0].id]=ast.literal_eval(node.value)
    table1=params['table1_parameters']
    expected=table1['T-IDM'].copy()
    expected['minimum_safe_arc_spacing_m']=expected.pop('minimum_desired_net_arc_gap_m')
    assert kernel['SHORT_RANGE_THRESHOLD_M']==table1['path_planner']['S_c_m']==15.8
    assert kernel['HEFEI_TIDM_PARAMETERS']==expected
    assert kernel['HEFEI_TFVD_PARAMETERS']==table1['T-FVD']
    protocol=json.loads((ROOT/'scripts/m11_protocol.json').read_text(encoding='utf-8'))
    assert protocol['optimizer']['seeds']==[42,43,44,45,46]
    assert protocol['optimizer']['evaluations_per_run']==8100
    assert len(rows(ROOT/'scripts/hefei_following_sample.csv'))==87
    assert len(rows(ROOT/'scripts/citysim_selected_following_pairs.csv'))==38
    runs=rows(ROOT/'outputs/M11/common_protocol/ga_runs.csv')
    for model in ('TMSFF','AV-IDM'):
        assert {int(r['seed']) for r in runs if r['model']==model}=={42,43,44,45,46}
    assert len(rows(ROOT/'outputs/M11/analysis/seed42_holdout_per_trajectory.csv'))==76
    metadata=json.loads((ROOT/'outputs/M11/source_hashes_and_cache_metadata.json').read_text(encoding='utf-8'))
    assert sha(ROOT/'scripts/citysim_selected_following_pairs.csv')==metadata['source_files']['selected_pairs']['sha256']
    readme=(ROOT/'README.md').read_text(encoding='utf-8')
    assert all(x in readme for x in ('Table 1','Table C2','Table E1','Figs. C3','Fig. C8'))
    figures=params['archived_manuscript_figures']
    assert set(figures)=={str(i) for i in range(9,25)}|{'C1','C2'}
    for number,record in figures.items():
        assert sha(ROOT/record['path'])==record['sha256'],f'Manuscript image mismatch: {number}'
    assert not list(ROOT.rglob('*.docx')) and not list(ROOT.rglob('*.xlsx'))
    assert not (ROOT/'scripts/plot_figure_c5.py').exists()
    assert not (ROOT/'outputs/Fig_C5.png').exists()
    print(f'PASS: {len(manifest)} checksummed files, {len(required)} required paths, 18 byte-matched manuscript figures, complete five-seed M11 outputs.')

if __name__=='__main__':main()
