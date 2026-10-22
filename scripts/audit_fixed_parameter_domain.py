from pathlib import Path
import sys,json,math,hashlib,csv
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parents[1]
import argparse
p=argparse.ArgumentParser(description='Post-hoc directed-heading-domain audit of the fixed Table 1 validation')
p.add_argument('--cache',type=Path,default=BASE/'reproduced_outputs/cache/m11_citysim_cache_k16.npz')
p.add_argument('--hefei-input',type=Path,default=BASE/'external_data/private_normalized_observed_trajectories.csv')
p.add_argument('--output-dir',type=Path,default=BASE/'reproduced_outputs/review_v2_domain')
a=p.parse_args();OUT=a.output_dir;OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(BASE/'scripts'))
import citysim_core as core
import run_rate_limiter_audit as rate

cache=a.cache
with np.load(cache,allow_pickle=True) as z:d={k:z[k] for k in z.files}
records=[]
def run_one(dataset,ident,observed,leaders,dt,model):
    core.DT=dt
    x,y,speed,heading=[float(t) for t in observed[0,:4]]
    count=len(observed)-1
    invalid=0;max_start=max_target=0.;minlocal=float('inf');cases=[0,0,0]
    for i in range(count):
        lx,ly,lv,lh=[float(t) for t in leaders[i,:4]]
        transformed_f,transformed_l,rotation=core.coordinate_system_transformation(x,y,heading,lx,ly,lh)
        dist=math.hypot(lx-x,ly-y)
        target=math.atan2(ly-y,lx-x) if dist<=15.8 else lh
        a=abs(float(core.wrap_angle(heading-rotation)))
        b=abs(float(core.wrap_angle(target-rotation)))
        max_start=max(max_start,a);max_target=max(max_target,b)
        invalid+=a>math.pi/2-.005+1e-12 or b>math.pi/2-.005+1e-12
        minlocal=min(minlocal,transformed_l[0]-transformed_f[0])
        gap=abs(float(core.wrap_angle(heading))-float(core.wrap_angle(lh)))
        case=0 if abs(gap/2-math.pi/2)<=.005 else (1 if gap>=math.pi else 2)
        cases[case]+=1
        x,y,heading,speed,*_=core.next_follower_state(model,x,y,heading,speed,lx,ly,lh,lv)
    records.append({'dataset':dataset,'trajectory_id':ident,'model':model,'updates':count,'outside_directed_domain':int(invalid),'maximum_start_angle_rad':max_start,'maximum_target_angle_rad':max_target,'minimum_local_x_m':minlocal,'Case1':cases[0],'Case2':cases[1],'Case3':cases[2]})

for model in ['T-IDM','T-FVD']:
    for i,length in enumerate(d['lengths']):
        run_one('CitySim',str(i+1),d['observed'][i,:int(length)],d['leader'][i,:int(length)],float(np.asarray(d['dt']).reshape(-1)[0]),model)
    print('Finished CitySim',model,flush=True)
hefei=a.hefei_input
for ident,pair in rate.load_hefei_trajectories(hefei):
    observed=pair[['carCenterXm_follower','carCenterYm_follower','velocity_follower','heading_follower']].to_numpy(float)
    leaders=pair[['carCenterXm_leader','carCenterYm_leader','velocity_leader','heading_leader']].to_numpy(float)
    for model in ['T-IDM','T-FVD']:run_one('Hefei',ident,observed,leaders,.1,model)
frame=pd.DataFrame(records)
frame.to_csv(OUT/'fixed_parameter_directed_domain_per_trajectory.csv',index=False,encoding='utf-8-sig')
summary=[]
for (dataset,model),group in frame.groupby(['dataset','model']):
    summary.append({'dataset':dataset,'model':model,'trajectories':len(group),'updates':int(group['updates'].sum()),'outside_directed_domain':int(group['outside_directed_domain'].sum()),'maximum_start_angle_rad':float(group['maximum_start_angle_rad'].max()),'maximum_target_angle_rad':float(group['maximum_target_angle_rad'].max()),'minimum_local_x_m':float(group['minimum_local_x_m'].min()),'Case1':int(group.Case1.sum()),'Case2':int(group.Case2.sum()),'Case3':int(group.Case3.sum())})
assert frame.outside_directed_domain.sum()==0
expected={('CitySim','T-IDM'):(8971,597,8.878),('CitySim','T-FVD'):(8971,423,4.232),('Hefei','T-IDM'):(1118,0,15.288),('Hefei','T-FVD'):(1118,0,15.128)}
for r in summary:
    count,case2,minx=expected[(r['dataset'],r['model'])]
    assert r['updates']==count and r['Case2']==case2 and r['Case1']==0
    assert abs(r['minimum_local_x_m']-minx)<.0006
report={'fixed_parameter_domain_audit':summary,'no_new_calibration':True,'hefei_input_sha256':hashlib.sha256(hefei.read_bytes()).hexdigest(),'citysim_cache_sha256':hashlib.sha256(cache.read_bytes()).hexdigest()}
(OUT/'audit_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2),flush=True)
