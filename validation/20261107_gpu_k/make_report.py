"""Summarize measured timings and accuracy without overwriting prior studies."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SKILL=ROOT/'skills/magtense-cupy-magnetostatics'
names=['NoShield','Shield_T010','Shield_T020','Shield_Large_T020']
native=json.loads((HERE/'native_Shield_T020/timing.json').read_text())
cases={}
rows=[]
for name in names:
    d=json.loads((HERE/name/'result.json').read_text())
    t=d['K_timings'][-3:]
    medians={key:statistics.median(x[key] for x in t) for key in t[0]
             if key.endswith('_seconds')}
    cases[name]={k:v for k,v in d.items() if k not in ['K_timings','solve','air_field_regularization']}
    cases[name]['solve']={k:v for k,v in d['solve'].items() if not isinstance(v,(list,dict))}
    cases[name]['K_warm_JIT_no_matrix_cache_median_seconds']=medians
    cases[name]['K_total_seconds_min_max']=[min(x['K_total_seconds'] for x in t),max(x['K_total_seconds'] for x in t)]
    cases[name]['K_first_trial']=d['K_timings'][0]
    rows.append({'case':name,'tiles':d['magnetic_tiles'],'points':d['points'],
         'GPU_K_prepare_s':medians['K_prepare_seconds'],'GPU_K_far_s':medians['K_far_seconds'],
         'GPU_K_near_s':medians['K_near_seconds'],'GPU_K_total_s':medians['K_total_seconds'],
         'GPU_solve_s':d['solve']['solve_seconds'],'iterations':d['solve']['iterations'],
         'K_relative_Frobenius':d['K_accuracy']['relative_Frobenius'],
         'CPU_material_residual':d['independent_CPU_residual'],
         'B_GPU_vs_CPU_RMS_mT':d['B_difference_from_previous_CPU_K_solution']['RMS_vector_error_mT'],
         'B_GPU_vs_CPU_max_mT':d['B_difference_from_previous_CPU_K_solution']['max_vector_error_mT'],
         'Maxwell_L2_before_percent':d['Maxwell_error_before']['relative_L2_percent'],
         'Maxwell_L2_after_percent':d['Maxwell_error_after']['relative_L2_percent']})
middle=cases['Shield_T020'];gpu=middle['K_warm_JIT_no_matrix_cache_median_seconds']
speed={key:native[key]/gpu[key] for key in ['K_far_seconds','K_near_seconds','K_total_seconds']}
before_compute=native['K_total_seconds']+middle['fresh_native_K_solve']['solve_seconds']
after_compute=gpu['K_total_seconds']+middle['solve']['solve_seconds']
summary={'status':'Completed CUDA K construction and four existing-Maxwell B comparisons',
 'method':'Independent analytical triangle surface-charge kernel; same 4/256 quadrature and near pairs',
 'hardware':{'CPU':'Intel Core i7-8700','native_CPU_threads':4,'GPU':'NVIDIA GeForce RTX 3090 Ti',
             'matrix_dtype':'float64','integral_dtype':'float64'},
 'timing_scope':{'fresh_native_full_rebuild_case':'Shield_T020','native_repeats':1,
        'GPU_repeats_per_case':4,'GPU_reported_statistic':'median of last 3 full rebuilds, warm JIT, no matrix cache',
        'CUDA_compilation':'First NoShield trial uses a separate initially empty CuPy compilation cache',
        'nonlinear_first_case':'NoShield solve includes initial compilation of CuPy material/iteration kernels in the new cache',
        'development_probes':'Small CUDA tests, synthetic solves and one K pilot overlapped the native CPU run; no concurrent Maxwell or real-mesh nonlinear solve',
        'full_rebuild':'No CPU camera block reuse; every K entry recomputed',
        'input_preparation_and_native_checks':'Separate from K and nonlinear iteration timers'},
 'native_full_rebuild':native,'middle_case_K_speedup':speed,
 'middle_case_K_plus_iteration':{'before_seconds':before_compute,'after_seconds':after_compute,
         'ratio':before_compute/after_compute,'scope':'Sum of separately timed stages, not continuous end-to-end wall time'},
 'cases':cases,
 'limitations':['GPU timing is device-specific; fresh CPU baseline is one run.',
       'The native baseline overlapped brief development probes; GPU measurement starts after the native process exits.',
       'Existing solved Maxwell fields are reused; no new Maxwell solve was submitted.',
       'Original mesh/material/quadrature approximations and prior failed mesh-refinement conclusion remain.',
       'Geometry/quadrature/near-list preparation, native verification and air fields retain CPU work.',
       'Dense O(N^2) storage and work remain; no FMM or selective-block update in the packaged CUDA builder.',
       'Real-material comparisons use FP64; FP32 CUDA/native agreement was tested only on small synthetic geometry.']}
cli_path=HERE/'cli_Shield_T020/cli_record.json'
if cli_path.exists():summary['real_model_CUDA_CLI']=json.loads(cli_path.read_text())
(HERE/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
with (HERE/'timing_accuracy.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)

fig,axes=plt.subplots(1,3,figsize=(16,5.2))
labels=['Far integral','Near integral','Whole K']
keys=['K_far_seconds','K_near_seconds','K_total_seconds']
x=np.arange(3)
axes[0].bar(x-.18,[native[k] for k in keys],.36,label='Native CPU, 4 threads',color='#587a9d')
axes[0].bar(x+.18,[gpu[k] for k in keys],.36,label='CUDA FP64',color='#dd8845')
axes[0].set_yscale('log');axes[0].set_xticks(x,labels);axes[0].set_ylabel('Elapsed seconds (log scale)')
axes[0].set_title('0.20 x 8.8 x 3.4 mm: identical full K')
for i,k in enumerate(keys):
    axes[0].text(i-.18,native[k]*1.08,f'{native[k]:.2f}',ha='center',fontsize=9)
    axes[0].text(i+.18,gpu[k]*1.08,f'{gpu[k]:.2f}',ha='center',fontsize=9)
axes[0].legend(fontsize=8);axes[0].grid(axis='y',alpha=.2)

short=['No shield','0.10 mm','0.20 mm','Large 0.20 mm']
y=np.arange(4);left=np.zeros(4)
for key,label,color in [('K_far_seconds','Far','#587a9d'),('K_near_seconds','Near','#dd8845')]:
    values=np.array([cases[n]['K_warm_JIT_no_matrix_cache_median_seconds'][key] for n in names])
    axes[1].barh(y,values,left=left,label=label,color=color);left+=values
other=np.array([cases[n]['K_warm_JIT_no_matrix_cache_median_seconds']['K_total_seconds'] for n in names])-left
axes[1].barh(y,other,left=left,label='Preparation / other',color='#90a18a')
axes[1].set_yticks(y,short);axes[1].invert_yaxis();axes[1].set_xlabel('Seconds')
axes[1].set_title('CUDA full rebuild; no K cache')
axes[1].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.17),ncol=3)
axes[1].grid(axis='x',alpha=.2)

x=np.arange(4)
before=[cases[n]['Maxwell_error_before']['relative_L2_percent'] for n in names]
after=[cases[n]['Maxwell_error_after']['relative_L2_percent'] for n in names]
axes[2].bar(x-.18,before,.36,label='CPU-built K',color='#587a9d')
axes[2].bar(x+.18,after,.36,label='CUDA-built K',color='#dd8845')
axes[2].set_xticks(x,['None','0.10','0.20','Large']);axes[2].set_xlabel('Shield case')
axes[2].set_ylabel('B-vector L2 error vs Maxwell (%)');axes[2].set_ylim(0,5)
axes[2].set_title('Same existing Maxwell reference / air probes')
for i,v in enumerate(after):axes[2].text(i,v+.1,f'{v:.5f}%',ha='center',fontsize=9)
axes[2].legend(fontsize=8);axes[2].grid(axis='y',alpha=.2)
fig.suptitle('CUDA tetrahedral K: time and accuracy, RTX 3090 Ti / FP64',fontsize=14)
fig.tight_layout();fig.savefig(HERE/'gpu_k_comparison.png',dpi=180);plt.close(fig)

lines=['GPU K 构建优化与精度对比，2026-11-07','',
 '实现：独立三角面磁荷解析式，CUDA 批量积分，近邻每 warp 对一个源/接收体对归约。',
 '保持原网格、材料、近邻集合及四点/256 点积分不变；所有四种模型完整重建 K，没有复用 camera 块。',
 'CUDA K 矩阵缓存关闭；报告后三次构建的中位数。首次单独空编译缓存的成本另见 summary.json。',
 'CPU 基线为中间尺寸完整重建一次，4 线程；GPU 正式计时在 CPU 进程结束后开始。',
 'CPU 基线期间有简短开发探针，不是严格独占机器的统计基准。','',
 f'中间尺寸 CPU 远场 {native["K_far_seconds"]:.6f}s；GPU {gpu["K_far_seconds"]:.6f}s。',
 f'中间尺寸 CPU 近邻 {native["K_near_seconds"]:.6f}s；GPU {gpu["K_near_seconds"]:.6f}s；比值 {speed["K_near_seconds"]:.2f}。',
 f'中间尺寸 CPU K 总计 {native["K_total_seconds"]:.6f}s；GPU {gpu["K_total_seconds"]:.6f}s；比值 {speed["K_total_seconds"]:.2f}。',
 f'同模型构 K + 非线性迭代阶段合计：{before_compute:.6f}s -> {after_compute:.6f}s，比例 {before_compute/after_compute:.2f}。该合计不是连续端到端计时。','']
lines.extend([f'同一次对比中，非线性迭代由 {middle["fresh_native_K_solve"]["iterations"]} 步 / {middle["fresh_native_K_solve"]["solve_seconds"]:.6f}s 变为 {middle["solve"]["iterations"]} 步 / {middle["solve"]["solve_seconds"]:.6f}s。',
              '解析式与归约顺序的微小舍入差异会改变 Anderson 迭代轨迹；GPU K 优化并未加速本构迭代本身。',''])
for row in rows:
    lines.extend([row['case'],f'  完整 GPU K {row["GPU_K_total_s"]:.6f}s；GPU 迭代 {row["GPU_solve_s"]:.6f}s / {row["iterations"]}步。',
       f'  K 相对 Frobenius 差 {row["K_relative_Frobenius"]:.6g}；独立 CPU 本构残差 {row["CPU_material_residual"]:.6g}。',
       f'  Maxwell B L2 差 {row["Maxwell_L2_before_percent"]:.10f}% -> {row["Maxwell_L2_after_percent"]:.10f}%。',
       f'  相对原 CPU K 解的 B RMS 差 {row["B_GPU_vs_CPU_RMS_mT"]:.10g}mT，最大差 {row["B_GPU_vs_CPU_max_mT"]:.10g}mT。',''])
lines.extend(['全部场比较复用原先已收敛的 Maxwell 结果，没有重新求解或改变参考网格。',
 '首个 NoShield 迭代计时包含独立空 CuPy 缓存下的首次材料/迭代核编译；不同算例间不可只按步数比耗时。',
 '后端等价不提高原来的空间离散精度；原 0.4mm 加密未通过的结论不变。',
 '几何、积分坐标、近邻枚举仍在 CPU，原生校验及最终空气场仍用 CPU。存储仍为完整 3N×3N。',
 '详细阶段和误差：summary.json；精简表：timing_accuracy.csv；图：gpu_k_comparison.png。'])
(HERE/'FINAL_REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')

# Portable skill evidence excludes local paths and large convergence histories.
portable={k:v for k,v in summary.items() if k not in ['native_full_rebuild','cases']}
portable['native_full_rebuild']={k:v for k,v in native.items() if isinstance(v,(int,float,bool))}
portable['cases']={name:{k:v for k,v in d.items() if k not in ['K_first_trial','air_field_regularization']}
                   for name,d in cases.items()}
(SKILL/'references/validation-gpu-k-20261107.json').write_text(json.dumps(portable,indent=2),encoding='utf-8')
print(json.dumps({'CPU_K_seconds':native['K_total_seconds'],'GPU_K_seconds':gpu['K_total_seconds'],
                  'K_speedup':speed,'cases':rows},indent=2))
