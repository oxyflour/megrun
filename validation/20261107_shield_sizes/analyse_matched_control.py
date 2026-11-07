"""Audit the completed single-pass control and quantify reference-grid sensitivity."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
source_out=ROOT/'matched_source_grid';control_out=ROOT/'matched_no_shield_reference'
source_mesh=json.loads((source_out/'mesh_fingerprint.json').read_text())
control_mesh=json.loads((control_out/'mesh_fingerprint.json').read_text())
assert source_mesh['tetrahedra']==control_mesh['tetrahedra']==351031
assert source_mesh['corner_coordinate_set_sha256']==control_mesh['corner_coordinate_set_sha256']
assert control_mesh['MID_SHIELD_material']=='vacuum'
assert 'Normal Completion' in (control_out/'profile.txt').read_text(errors='replace')
source=np.loadtxt(source_out/'maxwell_B.csv',delimiter=',',skiprows=1)
control=np.loadtxt(control_out/'maxwell_B.csv',delimiter=',',skiprows=1)
original=np.loadtxt(ROOT/'Shield_T010_reference/maxwell_B.csv',delimiter=',',skiprows=1)
baseline=np.loadtxt(ROOT/'baseline_reference/maxwell_B.csv',delimiter=',',skiprows=1)
gpu=np.loadtxt(ROOT/'Shield_T010/gpu_B.csv',delimiter=',',skiprows=1)
gpu0=np.loadtxt(ROOT/'baseline_gpu_B.csv',delimiter=',',skiprows=1)
np.testing.assert_allclose(source,original,rtol=0,atol=1e-11)
for field in [source,control,original,baseline,gpu,gpu0]:
    assert field.shape==(566,6) and np.isfinite(field).all()
    np.testing.assert_allclose(field[:,:3],source[:,:3],rtol=0,atol=1e-13)

def metric(g,r):
    e=g-r;v=np.linalg.norm(e,axis=1)
    return {'points':len(g),'relative_L2_percent':float(np.linalg.norm(e)/np.linalg.norm(r)*100),
       'RMS_vector_error_mT':float(np.sqrt(np.mean(v*v))*1000),'max_vector_error_mT':float(v.max()*1000),
       'reference_change_RMS_mT':float(np.linalg.norm(r)/np.sqrt(len(r))*1000),
       'predicted_change_RMS_mT':float(np.linalg.norm(g)/np.sqrt(len(g))*1000)}

groups=np.asarray(json.loads((ROOT/'point_groups.json').read_text())['labels'])
dg=gpu[:,3:]-gpu0[:,3:];dr=source[:,3:]-control[:,3:]
result={'case':'Shield_T010','same_corner_tetrahedron_set_verified':True,
        'source_mesh':source_mesh,'control_mesh':control_mesh,
        'all':metric(dg,dr),'groups':{g:metric(dg[groups==g],dr[groups==g]) for g in sorted(set(groups))},
        'two_no_shield_Maxwell_references_difference':metric(control[:,3:],baseline[:,3:]),
        'native_abort_requested_at_eight_minutes':True,
        'native_actual_comengine_completion':'Normal Completion, one fixed-mesh pass',
        'native_comengine_profile_seconds':645,'native_batch_wall_seconds_approx_from_logs':667,
        'independent_adaptive_convergence':False,'additional_native_solve_submitted':False,
        'note':'Control completed its full pass despite the abort request. Adopted only after actual profile/field and 351031 corner-tetrahedron checks. Quadratic midside nodes not compared; this is a diagnostic for one size, not an independent mesh-convergence bound.'}
(ROOT/'matched_mesh_delta_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
status=ROOT/'matched_control_status.json'
if status.exists() and not (ROOT/'matched_control_initial_monitor_status.json').exists():
    (ROOT/'matched_control_initial_monitor_status.json').write_bytes(status.read_bytes())
status.write_text(json.dumps({'state':'Completed full-pass diagnostic and mesh-audited',
    'independent_adaptive_convergence':False,'same_corner_tetrahedron_set_verified':True,
    'abort_requested':True,'native_comengine_completion':'Normal Completion',
    'used_only_for_paired_mesh_diagnostic':True,'metrics':result['all']},indent=2),encoding='utf-8')
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})
fig,axes=plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True)
fig.suptitle('0.10×8.8×3.4 mm：参考网格对 ΔB 的影响\n相同351031个四面体角点集合；去片为单轮固定网格诊断',fontsize=13)
for ax,group,title in zip(axes,['near_top_0.5mm','gap_A_side','gap_B_side'],['顶面 y=0，z=5.3 mm','间隙 A 侧 x=5.2 mm','间隙 B 侧 x=6.6 mm']):
    mask=groups==group
    if group=='near_top_0.5mm':mask &= np.isclose(source[:,1],0)
    indices=np.flatnonzero(mask);axis=0 if group=='near_top_0.5mm' else 2
    indices=indices[np.argsort(source[indices,axis])];x=source[indices,axis]*1000
    old=source[:,3:]-baseline[:,3:]
    ax.plot(x,np.linalg.norm(old[indices],axis=1)*1000,color='#92999f',label='不同网格 Maxwell 作差',lw=1.2)
    ax.plot(x,np.linalg.norm(dr[indices],axis=1)*1000,color='#245c84',label='相同角点网格 Maxwell 作差',lw=1.8)
    ax.plot(x,np.linalg.norm(dg[indices],axis=1)*1000,color='#d36b30',ls='--',label='MagTense/CuPy 作差',lw=1.6)
    ax.set(title=title,xlabel=('x' if axis==0 else 'z')+' (mm)',ylabel='|ΔB| (mT)')
    ax.grid(alpha=.17);ax.legend(fontsize=8)
fig.savefig(ROOT/'paired_mesh_delta_check.png',dpi=170);plt.close(fig)
print(json.dumps({'paired_delta':result['all'],'groups':result['groups'],
                  'Maxwell_no_shield_grid_difference':result['two_no_shield_Maxwell_references_difference']},indent=2))
