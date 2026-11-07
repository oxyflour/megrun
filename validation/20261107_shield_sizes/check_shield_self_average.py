"""Compare uniform-M shield self averages to an independent analytic prism.

This measures a quadrature diagnostic, not the nonlinear solution's field error.
"""
import json
from pathlib import Path
import sys

import numpy as np
from magtense.magstatics import Tiles,get_demag_tensor

ROOT=Path(__file__).resolve().parent
cases=json.loads((ROOT/'export_models.json').read_text())
results={}
for record in cases:
    out=Path(record['directory'])
    cache=out/'K_volume_r2_f1.5_float64.npy'
    if not Path(str(cache)+'.json').exists():
        continue
    model=json.loads((out/'model.json').read_text())
    obj=model['objects'][-1]
    v=np.load(out/obj['mesh_npz'])['tetra_m']
    n=len(v)
    weights=np.abs(np.linalg.det(v[:,1:]-v[:,:1]));weights/=weights.sum()
    shape=model['shield_geometry']
    dimensions=np.array([shape[k] for k in ['thickness_mm','width_mm','height_mm']])*1e-3
    center=np.asarray(shape['center_mm'])*1e-3
    prism=Tiles(n=1,tile_type=8,size=dimensions,offset=center,M_rem=0.,easy_axis=[1.,0.,0.],mu_r_ea=1.,mu_r_oa=1.)
    expected=np.asarray(get_demag_tensor(prism,center[None],obs_size=dimensions[None]))[0,0]
    K=np.load(cache,mmap_mode='r')
    # Receiver/out-component, source/in-component ordering matches the production packer.
    CC=np.asarray(K[-3*n:,-3*n:]).reshape(n,3,n,3)
    average=np.einsum('i,iajb->ab',weights,CC)
    relative=float(np.linalg.norm(average-expected)/np.linalg.norm(expected))
    diagonal=np.abs(np.diag(average-expected))/np.maximum(abs(np.diag(expected)),1e-30)*100
    results[out.name]={'shield_tetrahedra':n,'analytic_volume_average_K':expected.tolist(),
        'uniform_M_average_of_discrete_K':average.tolist(),'relative_Frobenius_difference_percent':relative*100,
        'relative_diagonal_difference_percent_xyz':diagonal.tolist(),
        'note':'Uniform-M aggregate only; does not bound nonuniform nonlinear solution accuracy or local tetrahedron quadrature errors'}
(ROOT/'shield_self_average_diagnostic.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps({name:{'overall_percent':v['relative_Frobenius_difference_percent'],
                       'diagonal_percent':v['relative_diagonal_difference_percent_xyz']} for name,v in results.items()},indent=2))
