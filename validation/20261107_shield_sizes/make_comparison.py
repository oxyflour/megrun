"""Compute field and shield-induced change errors at identical air coordinates."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parent
TOOL=Path(r'C:\Projects\magtense-gpu\repaired-skill\magtense-cupy-magnetostatics\scripts\aedt_magtense_bridge')
parser=argparse.ArgumentParser()
parser.add_argument('--allow-partial',action='store_true')
args=parser.parse_args()
groups=np.asarray(json.loads((ROOT/'point_groups.json').read_text())['labels'])

def read_array(path):
    value=np.loadtxt(path,delimiter=',',skiprows=1)
    assert value.shape==(566,6) and np.isfinite(value).all(),str(path)
    return value

def metrics(predicted,reference):
    error=predicted-reference
    norm_ref=float(np.linalg.norm(reference))
    vector=np.linalg.norm(error,axis=1)
    return {'points':len(vector),
       'relative_L2_percent':float(np.linalg.norm(error)/norm_ref*100) if norm_ref>1e-15 else None,
       'RMS_vector_error_mT':float(np.sqrt(np.mean(vector**2))*1000),
       'max_vector_error_mT':float(vector.max()*1000),
       'reference_RMS_mT':float(np.linalg.norm(reference)/np.sqrt(len(reference))*1000),
       'predicted_RMS_mT':float(np.linalg.norm(predicted)/np.sqrt(len(predicted))*1000)}

def grouped(predicted,reference):
    return {'all':metrics(predicted,reference),
            **{name:metrics(predicted[groups==name],reference[groups==name]) for name in sorted(set(groups))}}

baseline_g=read_array(ROOT/'baseline_gpu_B.csv')
baseline_r=read_array(ROOT/'baseline_reference/maxwell_B.csv')
np.testing.assert_allclose(baseline_g[:,:3],baseline_r[:,:3],rtol=0,atol=1e-13)
summary={'points':566,'units':{'coordinates':'m','fields':'T','absolute_metrics':'mT'},
  'definitions':{'relative_L2':'norm(predicted-reference)/norm(reference)*100',
                 'delta_B':'B_with_shield minus B_without_shield, separately in each solver',
                 'RMS_vector':'sqrt(mean(sum((predicted-reference)^2,axis=vector_components)))'},
  'baseline':grouped(baseline_g[:,3:],baseline_r[:,3:]),'cases':{}}
for name in ['Shield_T010','Shield_T020','Shield_Large_T020','Shield_T020_refined']:
    reference_name='Shield_T020' if name.endswith('_refined') else name
    out=ROOT/name;reference_out=ROOT/f'{reference_name}_reference'
    required=[out/'independent_audit.json',out/'gpu_B.csv',reference_out/'export_status.json',reference_out/'maxwell_B.csv']
    if not all(p.exists() for p in required):
        failure=out/'nonconvergence_audit.json'
        if name.endswith('_refined') and failure.exists():
            summary['middle_case_mesh_refinement']=json.loads(failure.read_text())
            continue
        if args.allow_partial:
            continue
        raise FileNotFoundError(f'Result missing for {name}')
    reference_status=json.loads((reference_out/'export_status.json').read_text())
    assert reference_status['adaptive_converged'],name
    gpu=read_array(out/'gpu_B.csv');ref=read_array(reference_out/'maxwell_B.csv')
    np.testing.assert_allclose(gpu[:,:3],baseline_r[:,:3],rtol=0,atol=1e-13)
    np.testing.assert_allclose(ref[:,:3],baseline_r[:,:3],rtol=0,atol=1e-13)
    gp,rf=gpu[:,3:],ref[:,3:]
    dg,dr=gp-baseline_g[:,3:],rf-baseline_r[:,3:]
    info=json.loads((out/'gpu_solve_info.json').read_text())
    audit=json.loads((out/'independent_audit.json').read_text())
    case={'total_B':grouped(gp,rf),'delta_B':grouped(dg,dr),
          'reference_change_relative_to_baseline_L2_percent':float(np.linalg.norm(dr)/np.linalg.norm(baseline_r[:,3:])*100),
          'maxwell':reference_status,'gpu':info,'independent_audit':audit}
    summary['cases'][name]=case
    path=out/'pointwise_comparison.csv'
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream)
        writer.writerow(['group','x_m','y_m','z_m','GPU_Bx_T','GPU_By_T','GPU_Bz_T',
                         'Maxwell_Bx_T','Maxwell_By_T','Maxwell_Bz_T','B_vector_error_T',
                         'GPU_dBx_T','GPU_dBy_T','GPU_dBz_T',
                         'Maxwell_dBx_T','Maxwell_dBy_T','Maxwell_dBz_T','dB_vector_error_T'])
        for i in range(566):
            writer.writerow([groups[i],*gpu[i,:3],*gp[i],*rf[i],np.linalg.norm(gp[i]-rf[i]),
                             *dg[i],*dr[i],np.linalg.norm(dg[i]-dr[i])])
    (out/'field_metrics.json').write_text(json.dumps({k:case[k] for k in ['total_B','delta_B',
         'reference_change_relative_to_baseline_L2_percent']},indent=2),encoding='utf-8')
    # Exercise the packaged comparison command as well as the Delta-B analysis.
    if not (out/'shield_metrics.json').exists():
        shutil.copy2(reference_out/'maxwell_B.fld',out/'maxwell_B.fld')
        with (out/'compare_cli.log').open('w') as log:
            p=subprocess.run([sys.executable,str(TOOL/'bridge.py'),'compare','--out',str(out),
                              '--prediction','gpu_B.csv','--prefix','shield_'],
                             cwd=TOOL,stdout=log,stderr=subprocess.STDOUT,timeout=60)
        assert p.returncode==0,name
    packaged=json.loads((out/'shield_metrics.json').read_text())
    assert abs(packaged['rms_vector_relative_error_percent']-case['total_B']['all']['relative_L2_percent'])<1e-10
if 'Shield_T020_refined' in summary['cases']:
    coarse=read_array(ROOT/'Shield_T020/gpu_B.csv')[:,3:]
    fine=read_array(ROOT/'Shield_T020_refined/gpu_B.csv')[:,3:]
    summary['middle_case_mesh_refinement']={
      'coarse_shield_elements':330,'refined_shield_elements':1188,
      'coarse_in_plane_target_mm':.8,'refined_in_plane_target_mm':.4,
      'cells_through_thickness_both':1,
      'field_difference_between_meshes':grouped(coarse,fine),
      'note':'In-plane refinement only. Same fixed camera mesh and Maxwell reference; not a strict error bound or thickness-convergence study.'}
summary['all_requested_results_complete']=(all(n in summary['cases'] for n in
    ['Shield_T010','Shield_T020','Shield_Large_T020']) and 'middle_case_mesh_refinement' in summary)
summary['all_requested_converged_solutions_available']=len(summary['cases'])==4
target=ROOT/('PROGRESS_METRICS.json' if args.allow_partial else 'comparison_summary.json')
target.write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({'baseline':summary['baseline']['all'],
   'cases':{k:{'total':v['total_B']['all'],'delta':v['delta_B']['all']} for k,v in summary['cases'].items()},
   'complete':summary['all_requested_results_complete']},indent=2),flush=True)
