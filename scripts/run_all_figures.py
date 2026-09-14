"""Reproduce selected revised figures without changing the archived outputs."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def run(script,*arguments):
    subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,arguments)],cwd=ROOT,check=True)

def common_figures(output):
    run('recalibrated_baseline_simulation.py',ROOT/'outputs/fair_calibration_runs.csv',output,'--seed',42)
    run('redraw_fig21_fig22_publication.py','--timeseries',output/'recalibrated_baseline_timeseries.csv',
        '--leader',output/'recalibrated_baseline_leader.csv','--metrics',output/'recalibrated_baseline_metrics.csv',
        '--output-dir',output)
    for old,new in [('Fig21_dynamic_response_smoothed_display.png','Fig_21_recalibrated.png'),
                    ('Fig22_normalized_performance_bar_line.png','Fig_22_recalibrated.png')]:
        shutil.copy2(output/old,output/new)

def m11_figures(output):
    import analyze_m11_results as m11
    import matplotlib.pyplot as plt
    m11.configure_matplotlib()
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'custom',
                        'mathtext.rm':'Times New Roman','mathtext.it':'Times New Roman:italic',
                        'mathtext.bf':'Times New Roman:bold'})
    m11.FIGURE_OUTPUT=output
    m11.plot_protocol_figure()
    per=pd.read_csv(ROOT/'outputs/M11/analysis/seed42_holdout_per_trajectory.csv')
    frame=pd.read_csv(ROOT/'outputs/M11/analysis/representative_pair_6_59_holdout_timeseries.csv')
    representative={'dt':float(np.median(np.diff(frame.time_s)))}
    for key,columns in {'leader':['leader_x_m','leader_y_m'],
                        'observed':['observed_x_m','observed_y_m'],
                        'TMSFF':['TMSFF_x_m','TMSFF_y_m'],
                        'AV-IDM':['AV_IDM_x_m','AV_IDM_y_m']}.items():
        representative[key]=frame[columns].to_numpy(float)
    m11.plot_holdout_figure(per,representative)
    for old,new in [('Fig_M11_1_common_protocol_and_generalization.png','Fig_23.png'),
                    ('Fig_M11_2_CitySim_holdout_comparison.png','Fig_24.png')]:
        shutil.copy2(output/old,output/new)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group',choices=['common','m11','all'],default='all')
    p.add_argument('--output-dir',type=Path,default=ROOT/'reproduced_outputs')
    a=p.parse_args();output=a.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    if a.group in ('common','all'):common_figures(output)
    if a.group in ('m11','all'):m11_figures(output)
    print('Completed:',output)

if __name__=='__main__':main()
