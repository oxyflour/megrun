"""Fresh native CPU K benchmark; retain all pre-existing validation artifacts."""
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
import json
from pathlib import Path
import sys
import time
import numpy as np
import cupy as cp

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT/'repaired-skill/magtense-cupy-magnetostatics/scripts/aedt_magtense_bridge'
sys.path.insert(0,str(TOOL))
from bridge import read_model
from magtense_adapter import assemble_tiles
from tetra_volume import assemble

source = ROOT/'validation/20261107_shield_sizes/Shield_T020'
out = Path(__file__).resolve().parent/'native_Shield_T020'
out.mkdir(exist_ok=False)
model = read_model(source)
tiles,_,_ = assemble_tiles(model['objects'],source)
started = time.perf_counter()
K,_,info = assemble(tiles,cp,dtype='float64',batch=16,cache=None)
cp.cuda.Stream.null.synchronize()
info.update(wall_assemble_seconds=time.perf_counter()-started,
            OMP_NUM_THREADS=os.environ['OMP_NUM_THREADS'], MKL_NUM_THREADS=os.environ['MKL_NUM_THREADS'],
            no_concurrent_Maxwell_or_real_mesh_nonlinear_solve=True,
            development_probes='Small unit tests, synthetic solves and one CUDA K pilot overlapped this run',
            disk_K_cache_enabled=False)
(out/'timing.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
baseline = np.load(source/'K_volume_r2_f1.5_float64.npy',mmap_mode='r')
numerator=denominator=0.; maximum=0.
for lo in range(0,len(K),384):
    block = cp.asnumpy(K[lo:lo+384]); reference = np.asarray(baseline[lo:lo+384])
    numerator += np.sum((block-reference)**2); denominator += np.sum(reference**2)
    maximum=max(maximum,float(np.max(abs(block-reference))))
info.update(relative_Frobenius_vs_previous=float(np.sqrt(numerator/denominator)),
            max_absolute_vs_previous=maximum)
(out/'timing.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
print(json.dumps(info,indent=2),flush=True)
