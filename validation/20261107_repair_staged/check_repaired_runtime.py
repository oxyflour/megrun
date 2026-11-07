"""Validate the packaged CLI on the full matrix, reusing attested identical quadrature."""
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / '20261107_gap3mm_vacoflux50'
TOOL = Path(r'C:\Projects\magtense-gpu\repaired-skill\magtense-cupy-magnetostatics\scripts\aedt_magtense_bridge')
sys.path.insert(0,str(TOOL))
from gpu_solver import hash_geometry
from magtense_adapter import assemble_tiles
from tetra_volume import quadrature, geometry
from average_pilot import quadrature as original_q4, red_quadrature as original_red

source = ROOT/'pair_volume_q4_near_red_q256'
info = json.loads((source/'operator_info.json').read_text())
assert info['native_averaged_H_check'] < 1e-9
output = ROOT/'full_pair_repaired_runtime'
output.mkdir(exist_ok=True)
model = json.loads((BASE/'coarse_export/model.json').read_text())
for obj in model['objects']:
    obj['mesh_npz'] = str(BASE/'coarse_export'/obj['mesh_npz'])
model['reference_source'] = {'project':str(ROOT/'full_pair_reference.aedt'),
    'design':'Gap_3mm_CCW','setup':'Reference : LastAdaptive','current_A_turn':0}
(output/'model.json').write_text(json.dumps(model,indent=2))
shutil.copy2(BASE/'points.csv',output/'points.csv')
shutil.copy2(ROOT/'full_pair_maxwell/maxwell_B.fld',output/'maxwell_B.fld')
tiles,_,_=assemble_tiles(model['objects'],output)
vertices,near=geometry(tiles,1.5)
np.testing.assert_array_equal(quadrature(vertices,0),original_q4(vertices,0))
np.testing.assert_array_equal(quadrature(vertices,2),original_red(vertices,2))
assert hash_geometry(tiles)==info['geometry_sha256']
assert float(np.mean([len(x) for x in near]))==info['near_source_count_mean']
cache=output/'K_volume_r2_f1.5_float64.npy'
if not cache.exists():
    shutil.copy2(source/'K.npy',cache)
meta={'schema':'magtense-tetra-volume-1','geometry_sha256':hash_geometry(tiles),
      'n':tiles.n,'dtype':'float64','receiver_rule':'gauss4','near_rule':'red8',
      'near_levels':2,'near_factor':1.5,'magtense_version':importlib.metadata.version('magtense')}
Path(str(cache)+'.json').write_text(json.dumps(meta,indent=2))
(output/'cache_origin.json').write_text(json.dumps({'source':str(source/'K.npy'),
 'quadrature_coordinates_bitwise_identical_for_all_receivers':True,
 'geometry_hash_verified':True,'near_pair_mean_verified':True,
 'native_field_check_will_be_repeated_by_packaged_CLI':True},indent=2))
command=[sys.executable,str(TOOL/'gpu_solver.py'),'--out',str(output),'--backend','cupy',
         '--dtype','float64','--operator','volume','--method','proximal','--tol','1e-8','--cache-k']
env={**os.environ,'PYTHONPATH':str(TOOL),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
p=subprocess.run(command,cwd=TOOL,env=env,capture_output=True,text=True,timeout=180)
(output/'run.log').write_text(p.stdout+p.stderr)
if p.returncode:
    raise RuntimeError(p.stdout+p.stderr)
actual=np.load(output/'gpu_magnetization.npz')['M_Apm']
baseline=np.load(source/'proximal_real_bh_solution.npz')['M_Apm']
relative=np.linalg.norm(actual-baseline)/np.linalg.norm(baseline)
assert relative < 1e-7,relative
print(json.dumps({'packaged_CLI_exit_code':p.returncode,'M_relative_difference_from_benchmark':float(relative),
                  'solve_info':json.loads((output/'gpu_solve_info.json').read_text())},default=str),flush=True)
