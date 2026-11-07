"""Create auditable, portable summaries and figures from completed solves."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / '20261107_gap3mm_vacoflux50'
SKILL = Path(r'C:\Projects\magtense-gpu\skills\magtense-cupy-magnetostatics')
TOOL = SKILL / 'scripts/aedt_magtense_bridge'
sys.path.insert(0, str(TOOL))
from magtense_adapter import assemble_tiles
from gpu_solver import MaterialLaws

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

runtime = ROOT / 'full_pair_repaired_runtime'
reference = ROOT / 'full_pair_maxwell'
model = read(runtime / 'model.json')
info = read(runtime / 'gpu_solve_info.json')
metrics = read(ROOT / 'full_pair_metrics.json')
native = read(ROOT / 'full_native_record.json')
provenance = read(BASE / 'material_provenance.json')
audit = read(reference / 'constitutive_audit.json')
base_k = read(ROOT / 'pair_volume_q4/operator_info.json')
near_k = read(ROOT / 'pair_volume_q4_near_red_q256/operator_info.json')
old_solve = read(ROOT / 'pair_volume_q4_near_red_q256/proximal_solve_info.json')
finite_audit = read(ROOT / 'finite_air_worst_point_audit.json')
groups = np.asarray(read(BASE / 'point_groups.json')['labels'])
pred = np.loadtxt(runtime / 'gpu_B.csv', delimiter=',', skiprows=1)
ref = np.loadtxt(reference / 'maxwell_B.csv', delimiter=',', skiprows=1)
assert pred.shape == ref.shape == (485, 6)
assert np.all(np.isfinite(pred)) and np.all(np.isfinite(ref))
np.testing.assert_allclose(pred[:, :3], ref[:, :3], rtol=0, atol=1e-13)
error = np.linalg.norm(pred[:, 3:] - ref[:, 3:], axis=1)
actual_l2 = np.linalg.norm(pred[:, 3:] - ref[:, 3:]) / np.linalg.norm(ref[:, 3:]) * 100
assert abs(actual_l2 - metrics['all']['relative_L2_percent']) < 1e-10

tiles, curves, stats = assemble_tiles(model['objects'], runtime)
M = np.load(runtime / 'gpu_magnetization.npz')['M_Apm']
K = np.load(runtime / 'K_volume_r2_f1.5_float64.npy', mmap_mode='r')
H = (K @ M.reshape(-1)).reshape(-1, 3)
laws = MaterialLaws(tiles, curves, np, 'float64')
F = laws.eval(H)[0]
residual_cpu = float(np.linalg.norm(M - F) / max(np.linalg.norm(M), 1))
assert residual_cpu < 1e-8
soft_h = np.linalg.norm(H[np.asarray(tiles.magnet_type) == 2], axis=1)
working_range = {
    'nonlinear_tiles': len(soft_h),
    'min_H_Apm': float(soft_h.min()),
    'max_H_Apm': float(soft_h.max()),
    'below_first_manufacturer_knot_H300': int(np.sum(soft_h < 300)),
    'above_last_manufacturer_knot_H16000': int(np.sum(soft_h > 16000)),
    'beyond_model_endpoint_H200000': int(np.sum(soft_h > 200000)),
    'independent_CPU_material_residual': residual_cpu,
    'max_magnetization_Apm': float(np.linalg.norm(M, axis=1).max()),
}
write(ROOT / 'final_constitutive_range.json', working_range)

source = Path(provenance['source_project'])
v2 = Path(r'C:\Projects\magtense-gpu\MagTense_CuPy_Skill_v2.zip')
integrity = {
    'original_project': str(source),
    'original_project_sha256_before': provenance['source_sha256'],
    'original_project_sha256_after': digest(source),
    'original_v2_zip_sha256_after': digest(v2),
    'original_v2_zip_sha256_before': '7af35276baebdce853612e570fc8fdd8b7c3c244687e000ca3342bcdc8e51c36',
}
integrity['original_project_unchanged'] = integrity['original_project_sha256_before'] == integrity['original_project_sha256_after']
integrity['original_v2_zip_unchanged'] = integrity['original_v2_zip_sha256_before'] == integrity['original_v2_zip_sha256_after']
assert integrity['original_project_unchanged'] and integrity['original_v2_zip_unchanged']
write(ROOT / 'source_integrity.json', integrity)

summary = {
    'date': '2026-11-07',
    'status': 'Completed real Maxwell/CuPy comparison; no user field-error threshold specified',
    'case': '3 mm two-module magnetostatics, both coil excitations zero',
    'material': {k: v for k, v in provenance.items() if k not in ('source_project', 'source_sha256', 'current_scope')},
    'other_materials': {'permanent_magnets': 'Original Br=1.2 T, isotropic recoil mu_r=1.05 and directions retained',
                        'copper': 'Weak diamagnetism mu_r=0.999991 replaced by 1 in both solvers'},
    'maxwell': {'version': '2022 R2', 'project': str(ROOT / 'full_pair_reference.aedt'),
                'design': 'Gap_3mm_CCW', 'setup': 'Reference', 'completion': 'Normal Completion',
                'passes': 5, 'total_tetrahedra': 333613, 'energy_error_percent': 0.35169,
                'delta_energy_percent': 0.010764, 'target_percent': 0.5,
                'minimum_converged_passes': 2, 'wall_seconds': native['seconds'],
                'BH_smoothing': False, 'nonlinear_residual_setting': 1e-5,
                'constitutive_field_audit': audit},
    'magtense_cupy': {'magtense_version': '3.0.1', 'gpu': info['gpu_model'],
                       'magnetic_tetrahedra': stats['tiles'], 'per_object': stats['per_object'],
                       'precision': 'float64', 'operator': 'Receiving-volume average of native source kernel',
                       'far_quadrature': 4, 'near_quadrature': 256, 'near_factor': 1.5,
                       'method': 'Implicit material proximal with safeguarded Anderson mixing',
                       'iterations': info['iterations'], 'relative_material_residual': info['residual'],
                       'independent_CPU_residual': residual_cpu,
                       'native_average_K_H_check': info['K_magtense_field_check_relative_error'],
                       'K_GB': info['K_GB'], 'first_K_base_seconds': base_k['K_build_seconds'],
                       'first_K_near_seconds': near_k['K_build_seconds'],
                       'first_K_total_seconds': base_k['K_build_seconds'] + near_k['K_build_seconds'],
                       'GPU_iteration_seconds': info['solve_seconds'],
                       'packaged_runtime_cache_load_seconds': info['K_build_or_load_seconds'],
                       'packaged_runtime_used_attested_identical_cached_K': True,
                       'spectrum_clipping': False, 'working_range': working_range},
    'field_comparison': metrics,
    'air_field_limit_evaluation': info['air_field_regularization'],
    'finite_worst_point_native_limit_max_difference_mT': finite_audit['maximum_difference_mT'],
    'verification': {'pytest_passed': 44, 'pytest_skipped': 1,
                      'skip_reason': 'Optional standalone Fortran patch test requires gfortran',
                      'native_export_verified_with_legacy_pyaedt': '0.8.11',
                      'CLI_backends_and_precisions': ['NumPy float64', 'CuPy float64', 'CuPy float32'],
                      'original_files_unchanged': integrity},
    'limitations': [
        'Manufacturer typical DC table is sparse (first nonzero H=300 A/m), not a dense measured curve for the original camera alloy.',
        'Most yoke observations operate below the first manufacturer knot; behavior there depends on the added origin and interpolation.',
        'Isotropic, single-valued, no hysteresis; TET10 meshes linearized to four corners; uniform magnetization per magnetic tetrahedron.',
        'Maxwell energy convergence is not a strict bound for the sampled B field. No additional air-boundary or magnetic-tile refinement study was performed.',
        'Material tolerance, remanence tolerance, assembly tolerance, live coils and force accuracy are outside this benchmark.',
        'The final material residual is not a field-error bound. No same-accuracy total speedup has been established.',
        'Raw failure records and raw nonfinite native-field output are retained; final field data use a checked limit at one nominal air point.',
    ],
    'recommended_command': 'python gpu_solver.py --out out --backend cupy --dtype float64 --operator volume --method proximal --tol 1e-8 --cache-k',
}
write(ROOT / 'final_summary.json', summary)
portable = {k: v for k, v in summary.items() if k not in ('verification',)}
portable['maxwell'].pop('project', None)
portable['verification'] = {k: v for k, v in summary['verification'].items() if k != 'original_files_unchanged'}
write(SKILL / 'references/validation-20261107.json', portable)

plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False,
                     'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(3, 2, figsize=(12.8, 11), constrained_layout=True)
fig.suptitle('3 mm 双模组 · VACOFLUX 50 厂家表 · 线圈电流为零\n485 点 B 向量相对 L2 差 3.66% ｜ RMS 1.48 mT ｜ 最大 7.49 mT', fontsize=15)
xyz = pred[:, :3] * 1000
near_center = (groups == 'near_top_0.5mm') & np.isclose(xyz[:, 1], 0)
x = xyz[near_center, 0]
for col, (component, label) in enumerate(((3, 'Bx'), (5, 'Bz'))):
    ax = axes[0, col]
    ax.plot(x, ref[near_center, component] * 1000, label='Maxwell 细参考', color='#172b4d', lw=2)
    ax.plot(x, pred[near_center, component] * 1000, label='MagTense/CuPy', color='#df6a28', ls='--', lw=1.6)
    ax.set(xlabel='x (mm)', ylabel=f'{label} (mT)', title=f'顶面上方 0.5 mm，y=0：{label}')
    ax.legend(fontsize=9)
ax = axes[1, 0]
for y in (-3, 0, 3):
    mask = (groups == 'near_top_0.5mm') & np.isclose(xyz[:, 1], y)
    ax.plot(xyz[mask, 0], error[mask] * 1000, label=f'y={y} mm', lw=1.3)
ax.set(xlabel='x (mm)', ylabel='B 向量绝对差 (mT)', title='顶面近场三条采样线')
ax.legend(fontsize=9)
ax = axes[1, 1]
mask = groups == 'far_top_3mm'
ax.plot(xyz[mask, 0], error[mask] * 1000, color='#146d73')
ax.set(xlabel='x (mm)', ylabel='B 向量绝对差 (mT)', title='顶面上方 3 mm：相对 L2 差 2.78%')
ax = axes[2, 0]
mask = groups == 'gap_vertical'
ax.plot(xyz[mask, 2], np.linalg.norm(ref[mask, 3:], axis=1) * 1000, label='Maxwell 细参考', color='#172b4d', lw=2)
ax.plot(xyz[mask, 2], np.linalg.norm(pred[mask, 3:], axis=1) * 1000, label='MagTense/CuPy', color='#df6a28', ls='--')
ax.set(xlabel='z (mm)', ylabel='|B| (mT)', title='间隙垂线 x=5.9 mm，y=0')
ax.legend(fontsize=9)
ax = axes[2, 1]
history = info['iteration_history']
ax.semilogy([v['iteration'] for v in history], [v['residual'] for v in history], color='#146d73', lw=1.8)
ax.axhline(1e-8, color='#8b4557', ls=':', label='残差验收 1e-8')
ax.set(xlabel='GPU 迭代次数', ylabel='相对本构残差', title='隐式本构迭代：734 步，约 3.75 s')
ax.legend(fontsize=9)
for ax in axes.flat:
    ax.grid(alpha=.17)
fig.savefig(ROOT / 'B_field_comparison.png', dpi=180)
fig.savefig(ROOT / 'B_field_comparison.pdf')
plt.close(fig)

rows = []
for label, title in [('all', '全部'), ('near_top_0.5mm', '顶面上方0.5 mm'),
                     ('far_top_3mm', '顶面上方3 mm'), ('gap_vertical', '模组间隙垂线')]:
    m = metrics[label]
    rows.append(f"{title:<20} {m['points']:>3} 点  相对L2 {m['relative_L2_percent']:.4f}%  RMS {m['RMS_vector_error_mT']:.4f} mT  最大 {m['max_vector_error_mT']:.4f} mT")
report = f'''3 mm 双模组公开材料 B–H 验证报告
日期：2026-11-07

结论
完整求解及同坐标 B 场对比已完成。旧故障发生在 MagTense/CuPy 非线性路径；本次 Maxwell 细参考正常完成并达到其原生自适应准则。修复后本构残差 {info['residual']:.9e}，独立 CPU 复算 {residual_cpu:.9e}。当前粗磁性网格对细 Maxwell 的场差约为百分之几，没有给定通过阈值，不能据此称为高精度或实物材料验证。

场差定义与结果
相对L2 = sqrt(sum(|B_GPU-B_Maxwell|²)/sum(|B_Maxwell|²))；RMS和最大值均为 B 向量绝对差，而非单个分量或逐点相对差。
{chr(10).join(rows)}
485 点全部保留且最终数据有限。近场 303 点为 y=-3/0/3 mm、z=5.3 mm 的三条 x 线；远场 101 点为 z=7.8 mm；间隙垂线 81 点为 x=5.9 mm、y=0。所有比较坐标经 SI 单位核对。

材料与激励
软轭铁采用 VAC 厂家 VACOFLUX 50、0.35 mm 带材、最终磁退火后的典型 DC 数据表。来源：
{provenance['url']}
PDF 第8页：H(A/m)=[300,800,1600,4000,8000,16000]；B(T)=[1.90,2.10,2.20,2.255,2.28,2.30]。
加原点(0,0)，分段线性插值；16000 A/m后按恒定 M 延伸，并加 H=200000 A/m 模型端点。这是公开可追溯的稀疏典型表，不是原相机实物牌号的密集实测曲线。无磁滞、各向同性。
两路线圈在副本中均为零；永磁 Br=1.2 T、recoil μr=1.05 及方向保留。铜 μr=0.999991 在两求解器中均改为1。
本次 MagTense 2686 个软磁单元的工作 H 范围 {working_range['min_H_Apm']:.6g}–{working_range['max_H_Apm']:.6g} A/m；{working_range['below_first_manufacturer_knot_H300']} 个处于首个厂家非零节点300 A/m以下，{working_range['above_last_manufacturer_knot_H16000']} 个超过厂家最后节点。低场结果主要受补原点与插值假设影响，不能声称验证了整条饱和曲线或低场实测材料性质。
Maxwell 材料回读与导入表一致。200个轭铁点的独立 B/H 导出核查：H=7.97–244.92 A/m，中位相对差1.49e-16，95分位3.95e-16，最大6.47%；场重建/插值仍有局部不一致，保留原始审计，不能把最大差消去。

网格与求解设置
Maxwell 2022 R2：Gap_3mm_CCW / Reference，0.5%自适应目标，3–8轮，要求连续2轮满足；实际5轮、333613总四面体，最后能量误差0.35169%、能量变化0.010764%，正常完成。NonLinearResidual=1e-5，B–H smoothing关闭。完整收敛表在 full_pair_maxwell/convergence.txt。
MagTense 3.0.1 + CuPy，RTX3090Ti，FP64；原几何10个磁性对象、3629个磁性四面体，2686软磁、943永磁。粗 Maxwell 网格仅1轮（总13019四面体、能量误差11.897%）用于提取几何离散，不作为细场参考。TET10只取角点线性化，每个磁单元均匀磁化；磁体积与 CAD 一致至约3e-15。
接收体积积分：远源4点，近邻/自作用均匀八分两层共256点，近邻因子1.5。仍用 CPU 原生解析源核，未裁剪特征值、未降低材料磁导率。对应接收规则的原生 K/H 自检相对差 {info['K_magtense_field_check_relative_error']:.4e}。
隐式局部本构更新配合有保护的 Anderson 混合，734步达到真实 M-F(KM) 残差小于1e-8。未使用更新量代替残差，也未关闭原生一致性校验。

失败原因与修复
旧 Newton 线搜索比较 ||R||/||M||，随 M 变化的分母会掩盖实际残差上升；已改为实际残差 Armijo 下降。但仅修此处仍不能收敛本例。重心离散的体积加权对称算子存在明显正方向；体积积分减小该问题，但本次没有证明这是唯一失败原因。最终采用接收体积积分和隐式本构更新跑通，保留所有旧失败结果供审查。
原生观测场在索引464、(5.9,0,4.45)mm产生一个 NaN。修复版在保持名义坐标不变的条件下，用两方向对称 Richardson 极限求值；全部偏置点经验证在空气内，步长/方向差4.10e-5 A/m（约0.052 nT）。最终485点全部纳入统计。误差最大的10个有限点另以相同极限核查，与直接原生值最大仅差 {finite_audit['maximum_difference_mT']:.3e} mT，不能解释7.49 mT的模型场差。

耗时与速度解释
Maxwell细参考原生进程：{native['seconds']:.3f} s（约10分15秒）。
CPU K：4点基算子 {base_k['K_build_seconds']:.3f} s，近邻积分 {near_k['K_build_seconds']:.3f} s，合计 {base_k['K_build_seconds'] + near_k['K_build_seconds']:.3f} s（约13分2秒）。这些为试验实测阶段耗时，不是新封装CLI首次重建的计时。
GPU本构迭代：修复包复现 {info['solve_seconds']:.3f} s；其缓存加载 {info['K_build_or_load_seconds']:.3f} s。该复现经全体积分坐标逐位一致、几何哈希与近邻集合核对后复用同一K，不能把缓存复现计为首次构建。K本体约0.948 GB。
两种方法的网格与精度不同，未证明同精度整体加速。GPU的数秒迭代不代表完整首次计算耗时。

封装验证与推荐命令
修复后单元检查44通过、1因缺少gfortran跳过（可选Fortran补丁测试）；另有真实原生核、NumPy/CuPy和FP64/FP32 CLI线性解析解核查。原生多B–H ABI补丁未安装。实际AEDT导出使用兼容的PyAEDT0.8.11环境；求解器环境独立。
进入修复skill的 scripts/aedt_magtense_bridge 后运行：
{summary['recommended_command']}
python bridge.py compare --out out --prediction gpu_B.csv --prefix gpu_
兼容默认值仍为centroid/newton，因此本例应明确传入volume/proximal参数。

产物与复核
final_summary.json：全部数值、条件、计时与局限。
B_field_comparison.png / .pdf：场分量、误差和本构残差曲线。
full_pair_reference.aedt：新材料、零电流的细参考工程副本。
full_pair_repaired_runtime/gpu_B.csv、gpu_magnetization.npz、gpu_solve_info.json：修复包实际运行结果。
full_pair_repaired_runtime/repaired_comparison.csv、repaired_metrics.json：封装compare命令的逐点结果。
full_pair_maxwell/：真实原生场、网格、profile、收敛与材料核查。
source_integrity.json：原始AEDT与v2ZIP哈希前后一致。原始文件未修改，旧失败记录未覆盖。

精度边界
Maxwell能量收敛不是空气B场的严格误差界；尚无额外空气边界/磁性网格细化序列。不含材料批次、装配公差、磁滞、线圈带电或磁力精度。1e-8本构残差不等于1e-8场误差。本轮完成公开曲线条件下的求解器对比，未证明原模组实物精度。
'''
(ROOT / 'FINAL_REPORT.txt').write_text(report, encoding='utf-8')
print(json.dumps({'working_range': working_range, 'field_relative_L2_percent': actual_l2,
                  'source_integrity': integrity, 'report': str(ROOT / 'FINAL_REPORT.txt')}, ensure_ascii=False), flush=True)
