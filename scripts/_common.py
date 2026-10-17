import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_mag.geometry import load_camera
from camera_mag.materials import benchmark_materials
from camera_mag.solver import CameraPairSolver


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--step", required=True)
    p.add_argument("--cell-mm", type=float, default=.7)
    p.add_argument("--current-a", type=float, default=.1)
    p.add_argument("--turns", type=int, default=100)
    p.add_argument("--br-t", type=float, default=1.2)
    p.add_argument("--recoil-mu-r", type=float, default=1.05)
    p.add_argument("--bh-csv", help="Measured B-H table with header H_A_per_m,B_T")
    p.add_argument("--cache-dir", default=".cache")
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    p.add_argument("--omega", type=float, default=.7)
    p.add_argument("--tol", type=float, default=1e-7)
    p.add_argument("--max-iterations", type=int, default=1500)
    p.add_argument("--method", choices=["fixed", "anderson"], default="anderson")
    p.add_argument("--coil-segments", type=int, default=20)
    p.add_argument("--coil-cross-samples", type=int, default=3)
    p.add_argument("--blas-threads", type=int, default=1)
    return p


def make_solver(args):
    camera = load_camera(args.step, args.cell_mm, args.cache_dir, args.br_t)
    print(f"Imported {len(camera.centers)} cells/camera, envelope {camera.extent * 1e3} mm", flush=True)
    solver = CameraPairSolver(camera, camera, benchmark_materials(args.br_t, args.recoil_mu_r, args.bh_csv),
              cache_dir=args.cache_dir, dtype=args.dtype, current_a=args.current_a, turns=args.turns,
              omega=args.omega, tol=args.tol, max_iterations=args.max_iterations, method=args.method,
              segments_per_side=args.coil_segments, cross_samples=args.coil_cross_samples, blas_threads=args.blas_threads)
    print("Self cache:", solver.precompute_self_operators(), flush=True)
    return solver
