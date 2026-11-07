"""Archive the actual unsuccessful accuracy validation without inventing errors."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
record = json.loads((ROOT / 'run_record.json').read_text())
provenance = json.loads((ROOT / 'material_provenance.json').read_text())
model = json.loads((ROOT / 'coarse_export/model.json').read_text())
source = Path(provenance['source_project'])
unchanged = hashlib.sha256(source.read_bytes()).hexdigest() == provenance['source_sha256']
assert unchanged
assert not (ROOT / 'coarse_export/gpu_B.csv').exists()
assert 'Gap_3mm_CCW' not in record['stages']
now = datetime.now(timezone.utc)
elapsed = (now - datetime.fromisoformat(record['study_started_UTC'])).total_seconds()
record.update(state='Stopped: target nonlinear solve did not converge; accuracy unverified',
              stopped_UTC=now.isoformat(), elapsed_study_seconds=elapsed,
              fine_reference_submitted=False, accuracy_metrics=None,
              source_unchanged=unchanged)
attempts = [
    {'name': 'original_skill', 'log': 'gpu_solve.log', 'result': 'Newton line search failed at iteration 1'},
    {'name': 'absolute_merit_fix', 'log': 'gpu_corrected_solve.log', 'result': 'GMRES info=120 at iteration 2'},
    {'name': 'absolute_merit_inner_1000', 'log': 'gpu_corrected_inner1000.log', 'result': 'GMRES info=1020 at iteration 4'},
    {'name': 'dense_Newton_diagnostic', 'log': 'gpu_dense_solve.log', 'result': '80 iterations; final residual 0.090299'},
    {'name': 'excitation_continuation_diagnostic', 'log': 'gpu_continuation.log', 'result': 'Br scales .01 and .05 converged; .1 failed at iteration 15'}]
record['nonlinear_attempts'] = attempts
(ROOT / 'run_record.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
spectrum = json.loads((ROOT / 'operator_spectrum.json').read_text())
versions = {'python_environment': r'C:\Projects\magtense-gpu\skill-test\.venv'}
for name in ['numpy', 'scipy', 'magtense', 'cupy-cuda13x', 'pytest']:
    versions[name] = importlib.metadata.version(name)
summary = {'outcome': 'Accuracy validation did not pass; no converged target solution',
    'source_unchanged': unchanged, 'point_count': 485, 'tiles': model['total_magnetic_tetra'],
    'native_coarse_total_tetra': 13019, 'native_coarse_adaptive_converged': False,
    'native_coarse_energy_error_percent': 11.897, 'native_coarse_seconds': record['stages']['MT_Coarse']['seconds'],
    'K_FP64_bytes': 72 * model['total_magnetic_tetra'] ** 2,
    'K_native_H_relative_error': 1.135e-15, 'final_Maxwell_comparison': None,
    'fine_Maxwell_solve_submitted': False,
    'tet10_corner_linearized': sum(x['mesh_stats']['tet10'] for x in model['objects']),
    'curved_geometry_present': any(x['mesh_stats']['curved'] for x in model['objects']),
    'maximum_mesh_CAD_volume_relative_difference': max(x['mesh_stats']['volume_relative_difference'] for x in model['objects']),
    'operator_spectrum': spectrum, 'attempts': attempts, 'versions': versions,
    'approved_budget_seconds': 3000, 'actual_study_elapsed_seconds': elapsed,
    'original_solver_source_edited': False,
    'diagnostic_solver_files': ['gpu_solver_corrected.py', 'gpu_solver_dense_newton.py', 'gpu_continuation.py']}
(ROOT / 'validation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
fig, axes = plt.subplots(1, 2, figsize=(14, 5.1), layout='constrained')
curve = np.loadtxt(ROOT / 'vacoflux50_bh.csv', delimiter=',', skiprows=1)
hs = np.geomspace(1, 200000, 1000)
bs = np.interp(hs, curve[:, 0], curve[:, 1])
axes[0].semilogx(hs[hs <= 16000], bs[hs <= 16000], color='#246aa8', label='双方采用的分段线性本构')
axes[0].semilogx(hs[hs >= 16000], bs[hs >= 16000], '--', color='#777777', label='恒定 M 外推（模型假设）')
axes[0].scatter(curve[1:-1, 0], curve[1:-1, 1], color='#c45b2c', s=38, zorder=4, label='VAC 厂家典型 DC 数据点')
axes[0].set(xlabel='H (A/m)', ylabel='B (T)', title='已替换：VACOFLUX 50 最终磁性退火带材')
axes[0].grid(alpha=.22)
axes[0].legend(fontsize=9, loc='lower right')
for file, label, color in [('gpu_solve.log', '原 skill：线搜索失败', '#bd4635'),
                           ('gpu_dense_solve.log', '稠密 Newton 诊断：80 步未收敛', '#246aa8')]:
    text = (ROOT / file).read_text(errors='replace')
    rows = re.findall(r'(\d+): relative nonlinear residual = ([\deE+.-]+)', text)
    axes[1].semilogy([int(i) for i, v in rows], [float(v) for i, v in rows], color=color, label=label)
axes[1].axhline(1e-8, color='#606060', linestyle='--', label='目标残差 1e-8')
axes[1].set(xlabel='Newton 步数', ylabel='相对非线性残差', ylim=(1e-9, 10),
            title='目标 Br=1.2 T 的真实模型尚未收敛')
axes[1].grid(alpha=.22)
axes[1].legend(fontsize=9, loc='center right')
fig.savefig(ROOT / 'material_and_convergence.png', dpi=180)
plt.close(fig)

report = f'''真实 B-H 模型精度验证：未通过
日期：2026-11-07（Asia/Shanghai）

本次结论
已在独立 Maxwell 工程副本中替换轭铁材料并将两路线圈电流置零。
当前 skill 在该真实材料模型上未得到收敛的目标场，因此无法计算可信的 MagTense/Maxwell B 场误差。
没有把非收敛输出或旧材料结果作为精度参考；细 Maxwell 参考没有提交。

工程与材料
副本：{ROOT / 'gap3mm_vacoflux50.aedt'}
源工程：{source}
源工程未修改：{unchanged}
轭铁采用 VAC 官方《Soft Magnetic Cobalt-Iron Alloys》第 8 页 VACOFLUX 50 典型 DC 数据。
条件：最终磁性退火，带材厚度 0.35 mm；这是公开实际材料的替代数值基准，并非原摄像头材料认证。
来源：{provenance['url']}
原厂数据 H(A/m)/B(T)：300/1.90、800/2.10、1600/2.20、4000/2.255、8000/2.28、16000/2.30。
补充原点，点间线性插值，超过 16000 A/m 按恒定 M 外推；外推不是原厂实测数据。
工程内材料名称仍为 Magrun_Steel_BH，曲线已替换，读回证据见 maxwell_material_readback.json。
永磁体保留 Br=1.2 T、回线 mu_r=1.05 及原方向。铜的微弱抗磁性统一省略，双方使用 mu_r=1。

真实 Maxwell 粗网格
设计 MT_Coarse，0 A·匝，完整原 CAD，1.5 mm 长度约束，单轮。
原生 Normal Completion，计算 {summary['native_coarse_seconds']:.2f} 秒，总计 13019 个四面体。
磁性四面体 3629 个：两个轭铁和八块永磁体；空气与零电流非磁性铜省略。
单轮能量误差 11.897%，原生自适应未收敛；该场只作为导出/流程检查，不能用作高精度参考。
485 个空气观测点，分组与 SI 坐标核对通过。
3629 个 TET10 取四个角点转为 TET4；无曲面几何标记。
各体 CAD/网格体积相对差最大 {summary['maximum_mesh_CAD_volume_relative_difference']:.3e}，包围盒核对通过。
FP64 完整 K 为 {summary['K_FP64_bytes']/1e9:.6f} GB。
CPU 原生四面体张量构建，GPU 仅用于迭代。原生 K/H 随机磁化检查相对差 1.135e-15，通过。

收敛失败与修复诊断
1. 原 skill：Newton 第 1 次更新后在线搜索失败。原始日志 gpu_solve.log 保留。
2. 发现可复现代码缺陷：线搜索以 ||R||/||M|| 作变化分母的下降判据。
   第一步实际 ||R|| 从 9.3994e7 增至 1.9351e8，却因归一化分母增大而被接受。
   下一步存在 ||R|| 下降的步长，但均被相对判据拒绝。
3. 独立修正版采用实际残差的 Armijo 下降判据，保持最终收敛容差不变。
   新增解析双单元回归可复现原缺陷并验证修复；修正版合计 35 passed、1 skipped（无 gfortran）。
4. 修正版仍遇到 GMRES 上限；1000 内层上限诊断也未收敛。
5. 稠密 Jacobian/LU 的直接 Newton 诊断迭代 80 步，最后相对残差约 0.0903，未达到 1e-8。
6. 不依赖 Maxwell 初值的激励延拓：Br 比例 0.01 与 0.05 可收敛，0.10 时第 15 步线搜索失败。
   未到达比例 1.0，不能作为 Br=1.2 T 目标模型的有效解。
7. 原 skill 源文件未改；所有修复与诊断代码单独放在本目录。

进一步的算子诊断（推断，尚非误差结论）
体积权重变换 sqrt(V) K inv(sqrt(V)) 的对称部分最大特征值约 0.02489336，
该特征对残差约 2.1e-15，表明重心配点算子存在正方向。
这提示应优先检查重心配点与体积平均退磁算子的能量稳定性和强磁导材料耦合，
不能把随机 K/H 排列检查通过等同于空间离散精度通过。
该检查尚未单独证明非收敛的唯一原因，也不是 Maxwell B 场误差测量。

未完成的精度指标
目标残差：未达到；B 场相对 L2、RMS 和最大绝对误差：无有效值。
细参考 Gap_3mm_CCW 已准备并验证，但未提交求解；不会引用旧材料的结果。
未评估工程规模加速比或实物精度。

预算和状态
用户批准整组 50 分钟；从正式粗网格提交到封存约 {elapsed/60:.2f} 分钟。
已停止全部本次求解，未追加参考加密或恢复旧队列。
全部失败记录、阶段计时与状态保存在 run_record.json 及对应日志中。

主要交付
validation_summary.json：机器可读结论。
material_and_convergence.png：材料曲线与目标模型的残差。
vacoflux50_bh.csv、VAC_CoFe_manufacturer.pdf、material_provenance.json：曲线与来源。
newton_fix.patch、test_newton_regression.py：可独立审查的线搜索修复。
nonconvergence_diagnostic.json、operator_spectrum.json：诊断原始数据。
coarse_export/：真实 Maxwell 网格、粗场、材料和 K 缓存。
'''
(ROOT / 'VALIDATION_REPORT.txt').write_text(report, encoding='utf-8')
print(json.dumps({'state': record['state'], 'study_minutes': elapsed/60,
                  'source_unchanged': unchanged, 'report': str(ROOT / 'VALIDATION_REPORT.txt')}, ensure_ascii=False))
