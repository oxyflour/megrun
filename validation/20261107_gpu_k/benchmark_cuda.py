"""Rebuild identical K on CUDA; compare every entry, solve and check Maxwell B.

No AEDT solve, geometry change, material change, warm start, block reuse or K
matrix cache is used. Repeated K timings exclude only an explicitly warm JIT.
"""
import os
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
os.environ['CUPY_CACHE_DIR']=os.path.join(os.path.dirname(os.path.abspath(__file__)),'.cupy_cache')
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import cupy as cp

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
TOOL=ROOT/'skills/magtense-cupy-magnetostatics/scripts/aedt_magtense_bridge'
sys.path.insert(0,str(TOOL))
from bridge import read_model,load_points,check_points_in_air,read_maxwell_fld
from magtense_adapter import assemble_tiles
from gpu_solver import MaterialLaws,solve_nonlinear,MU0,hash_geometry
from tetra_cuda import assemble
from tetra_volume import geometry,verify
from air_field import evaluate


def metrics(pred,ref):
    errors=np.linalg.norm(pred-ref,axis=1)
    return {'relative_L2_percent':float(np.linalg.norm(pred-ref)/np.linalg.norm(ref)*100),
            'RMS_vector_error_mT':float(np.sqrt(np.mean(errors**2))*1000),
            'max_vector_error_mT':float(errors.max()*1000)}


def save(path,value):
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def compare_K(K,reference,near):
    totals=np.zeros(4);maximum=0.
    for lo in range(0,len(near),64):
        count=min(64,len(near)-lo)
        block=cp.asnumpy(K[3*lo:3*(lo+count)]).reshape(count,3,len(near),3).transpose(0,2,1,3)
        ref=np.asarray(reference[3*lo:3*(lo+count)]).reshape(count,3,len(near),3).transpose(0,2,1,3)
        diff=np.sum((block-ref)**2,axis=(2,3));norm=np.sum(ref**2,axis=(2,3))
        mask=np.zeros(diff.shape,bool)
        for j in range(count):mask[j,near[lo+j]]=True
        totals += [diff[mask].sum(),norm[mask].sum(),diff[~mask].sum(),norm[~mask].sum()]
        maximum=max(maximum,float(np.max(abs(block-ref))))
    return {'relative_Frobenius':float(np.sqrt((totals[0]+totals[2])/(totals[1]+totals[3]))),
            'near_relative_Frobenius':float(np.sqrt(totals[0]/totals[1])),
            'far_relative_Frobenius':float(np.sqrt(totals[2]/totals[3])),
            'max_absolute_entry':maximum}


def run(name,repeats):
    if name=='NoShield':
        source=ROOT/'validation/20261107_repair_staged/full_pair_repaired_runtime'
    else:
        source=ROOT/'validation/20261107_shield_sizes'/name
    out=OUT/name;out.mkdir(exist_ok=False)
    model=read_model(source);points=load_points(source,model)
    check_points_in_air(source,model,points)
    tiles,curves,stats=assemble_tiles(model['objects'],source)
    native_reference=np.load(source/'K_volume_r2_f1.5_float64.npy',mmap_mode='r')
    timings=[]
    for trial in range(repeats):
        K,diag,info=assemble(tiles,cp,dtype='float64',cache=None)
        assert not info['loaded_from_cache']
        timings.append(info)
        save(out/'K_timings.json',timings)
        print(name,'K trial',trial,json.dumps(info),flush=True)
        if trial+1 < repeats:
            del K,diag
            gc.collect();cp.get_default_memory_pool().free_all_blocks()
    _,near=geometry(tiles,1.5)
    k_error=compare_K(K,native_reference,near)
    assert k_error['relative_Frobenius']<1e-9 and k_error['max_absolute_entry']<1e-8,k_error
    start=time.perf_counter();native_check=verify(tiles,K,cp)
    native_verify_seconds=time.perf_counter()-start
    save(out/'K_accuracy.json',{**k_error,'native_H_check':native_check,
                                'native_verify_seconds':native_verify_seconds})
    laws=MaterialLaws(tiles,curves,cp,'float64')
    before_solve=None
    if name=='Shield_T020':
        # This CPU tensor is independently checked against the fresh native
        # rebuild by benchmark_native.py. Re-solve it with the same current
        # CuPy solver, without a warm start, for a contemporaneous solve timer.
        old_K=cp.asarray(native_reference)
        from tetra_volume import diagonal_blocks
        before_M,before_solve=solve_nonlinear(old_K,diagonal_blocks(old_K,cp,tiles.n),laws,
                            method='proximal',tol=1e-8,maxiter=24000,verbose=False)
        before_solve={k:v for k,v in before_solve.items() if k not in ['residual_history','iteration_history']}
        before_M=cp.asnumpy(before_M)
        np.savez_compressed(out/'fresh_native_K_magnetization.npz',M_Apm=before_M)
        del old_K
    M,solve=solve_nonlinear(K,diag,laws,method='proximal',tol=1e-8,maxiter=24000,verbose=False)
    cp.cuda.Stream.null.synchronize()
    magnetization=cp.asnumpy(M)
    Hcpu=np.empty((tiles.n,3))
    start=time.perf_counter()
    for lo in range(0,len(K),384):
        Hcpu.ravel()[lo:lo+384]=cp.asnumpy(K[lo:lo+384])@magnetization.ravel()
    residual=float(np.linalg.norm(magnetization-MaterialLaws(tiles,curves,np,'float64').eval(Hcpu)[0])
                   /max(np.linalg.norm(magnetization),1))
    assert residual<=1e-8,residual
    independent_residual_seconds=time.perf_counter()-start
    old_M=np.load(source/'gpu_magnetization.npz')['M_Apm']
    tiles.M=magnetization
    start=time.perf_counter()
    H,air_info=evaluate(tiles,points,validate_air=lambda p:check_points_in_air(source,model,p))
    field_seconds=time.perf_counter()-start
    B=MU0*H
    old=np.loadtxt(source/'gpu_B.csv',delimiter=',',skiprows=1)
    np.testing.assert_allclose(points,old[:,:3],rtol=0,atol=1e-14)
    if name=='NoShield':
        reference,_=read_maxwell_fld(source/'maxwell_B.fld',points)
    else:
        data=np.loadtxt(source.parent/(name+'_reference')/'maxwell_B.csv',delimiter=',',skiprows=1)
        np.testing.assert_allclose(points,data[:,:3],rtol=0,atol=1e-14)
        reference=data[:,3:]
    np.savetxt(out/'gpu_B.csv',np.c_[points,B],delimiter=',',comments='',header='x_m,y_m,z_m,Bx_T,By_T,Bz_T')
    np.savez_compressed(out/'gpu_magnetization.npz',M_Apm=magnetization,geometry_sha256=hash_geometry(tiles))
    result={'case':name,'magnetic_tiles':tiles.n,'points':len(points),'K_matrix_cache_used':False,
            'CPU_camera_block_reused':False,'K_timings':timings,'K_accuracy':k_error,
            'fresh_native_K_solve':before_solve,
            'native_H_check':native_check,'native_verify_seconds':native_verify_seconds,
            'solve':solve,'independent_CPU_residual':residual,
            'independent_residual_seconds':independent_residual_seconds,
            'air_field_seconds':field_seconds,'air_field_regularization':air_info,
            'M_relative_difference_from_previous_CPU_K_solution':float(np.linalg.norm(magnetization-old_M)/np.linalg.norm(old_M)),
            'B_difference_from_previous_CPU_K_solution':metrics(B,old[:,3:]),
            'Maxwell_error_before':metrics(old[:,3:],reference),'Maxwell_error_after':metrics(B,reference)}
    assert result['B_difference_from_previous_CPU_K_solution']['relative_L2_percent']<1e-3
    save(out/'result.json',result)
    print('COMPLETE',name,json.dumps({k:v for k,v in result.items() if k not in ['solve','K_timings','air_field_regularization']}),flush=True)
    del K,diag,M,laws
    gc.collect();cp.get_default_memory_pool().free_all_blocks()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cases',nargs='+',default=['NoShield','Shield_T010','Shield_T020','Shield_Large_T020'])
    parser.add_argument('--repeats',type=int,default=4)
    args=parser.parse_args()
    for name in args.cases:run(name,args.repeats)
