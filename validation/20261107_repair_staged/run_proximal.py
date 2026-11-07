from datetime import datetime
import argparse
import json
from pathlib import Path
import sys
import time

import cupy as cp
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / '20261107_gap3mm_vacoflux50'
TOOL = ROOT.parents[1] / 'skills/magtense-cupy-magnetostatics/scripts/aedt_magtense_bridge'
sys.path.insert(0, str(TOOL))
from gpu_solver import MaterialLaws
from magtense_adapter import assemble_tiles
from magtense.magstatics import get_H_field
from proximal_solver import solve

parser = argparse.ArgumentParser()
parser.add_argument('--operator', default='single_volume_q4_near_q256')
parser.add_argument('--budget-record', type=Path, default=ROOT/'session.json')
args = parser.parse_args()
output = ROOT / args.operator
metadata = json.loads((output / 'operator_info.json').read_text())
assert metadata['native_averaged_H_check'] < 1e-9
session = json.loads(args.budget_record.read_text())
deadline = datetime.fromisoformat(session['started_UTC']).timestamp() + session['budget_seconds']
if metadata['tiles'] == 1727:
    model = json.loads((ROOT / 'single_real_bh/model.json').read_text())
else:
    model = json.loads((BASE / 'coarse_export/model.json').read_text())
    for obj in model['objects']:
        obj['mesh_npz'] = str(BASE / 'coarse_export' / obj['mesh_npz'])
tiles, curves, _ = assemble_tiles(model['objects'], ROOT)
K = cp.asarray(np.load(output / 'K.npy', mmap_mode='r'))
law = MaterialLaws(tiles, curves, cp, 'float64')
try:
    M, info = solve(K, law, deadline=deadline)
    tiles.M = cp.asnumpy(M)
    points = np.loadtxt(BASE / 'points.csv', delimiter=',', skiprows=1)
    B = 4e-7*np.pi*get_H_field(tiles, points)
    np.savetxt(output / 'proximal_real_bh_B.csv', np.c_[points, B], delimiter=',',
               header='x_m,y_m,z_m,Bx_T,By_T,Bz_T', comments='')
    np.savez_compressed(output / 'proximal_real_bh_solution.npz', M_Apm=tiles.M)
    H = (np.load(output/'K.npy', mmap_mode='r') @ tiles.M.ravel()).reshape(tiles.n, 3)
    cpu_law = MaterialLaws(tiles, curves, np, 'float64')
    independent = np.linalg.norm(tiles.M-cpu_law.eval(H)[0]) / np.linalg.norm(tiles.M)
    assert independent < 1e-8
    info.update(independent_CPU_material_residual=float(independent),
                max_M_Apm=float(np.linalg.norm(tiles.M, axis=1).max()))
except Exception as error:
    info = {'converged': False, 'error': repr(error)}
(output / 'proximal_solve_info.json').write_text(json.dumps(info, indent=2))
print(json.dumps({k:v for k,v in info.items() if k!='history'}), flush=True)
