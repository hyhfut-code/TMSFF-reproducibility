from pathlib import Path
import argparse,sys
import numpy as np
import pandas as pd
from scipy import stats
from m11_models import trace_tmsff,trace_avidm
from run_m11_calibration import load_cache,MODEL_PARAMETERS
from audit_r1_1 import seed_parameters

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--cache',type=Path,default=root/'reproduced_outputs/cache/m11_citysim_cache_k16.npz')
p.add_argument('--m11-results',type=Path,default=root/'outputs/M11')
p.add_argument('--output-dir',type=Path,default=root/'reproduced_outputs/R1_1/analysis')
a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
d=load_cache(a.cache);par={m:seed_parameters(a.m11_results,m,42,MODEL_PARAMETERS) for m in ['TMSFF','AV-IDM']}
records=[]
for i,(length,split) in enumerate(zip(d['lengths'],d['split_indices'])):
    length=int(length);split=int(split);h=length-split
    for label,start,stop in [('late_calibration_equal_horizon',split-h,split),('holdout',split,length)]:
        tf=trace_tmsff(par['TMSFF'],i,start,stop,d['observed'],d['leader'],float(d['dt']))
        av=trace_avidm(par['AV-IDM'],i,start,stop,d['observed'],d['environment'],d['route_x'],d['route_y'],d['route_heading'],d['route_left_width'],d['route_right_width'],d['route_station'],float(d['dt']))
        r={'pair_index':i,'trajectory_no':int(d['pair_metadata'][i,0]),'block':label}
        for key,trace in [('tmsff',tf),('avidm',av)]:
            assert np.isfinite(trace).all()
            r[key+'_position_rmse_m']=float(np.sqrt(np.mean(np.sum((trace[:,:2]-d['observed'][i,start:stop,:2])**2,axis=1))))
        records.append(r)
raw=pd.DataFrame(records);raw.to_csv(a.output_dir/'equal_horizon_raw_position_rmse.csv',index=False,encoding='utf-8-sig')
paired=[]
for key in ['tmsff','avidm']:
    w=raw.pivot(index='pair_index',columns='block',values=key+'_position_rmse_m')
    late=w['late_calibration_equal_horizon'].to_numpy();hold=w['holdout'].to_numpy()
    paired.append({'model':key,'late_median':float(np.median(late)),'holdout_median':float(np.median(hold)),'median_paired_change':float(np.median(hold-late)),'paired_p':float(stats.wilcoxon(late,hold).pvalue)})
pd.DataFrame(paired).to_csv(a.output_dir/'equal_horizon_raw_position_paired_tests.csv',index=False,encoding='utf-8-sig')
print(pd.DataFrame(paired).to_string(index=False))
