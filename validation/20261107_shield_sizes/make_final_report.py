"""Create scientific figures and a Chinese report from all completed references."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT=Path(__file__).resolve().parent
summary=json.loads((ROOT/'comparison_summary.json').read_text())
assert summary['all_requested_results_complete']
prepared=json.loads((ROOT/'prepared_cases.json').read_text())
session=json.loads((ROOT/'group_session.json').read_text())
cases=summary['cases']
names=['Shield_T010','Shield_T020','Shield_Large_T020']
titles=['0.10×8.8×3.4','0.20×8.8×3.4','0.20×10×5']
colors=['#246a8a','#d36b30','#397c50']
labels=np.asarray(json.loads((ROOT/'point_groups.json').read_text())['labels'])
baseline_g=np.loadtxt(ROOT/'baseline_gpu_B.csv',delimiter=',',skiprows=1)
baseline_r=np.loadtxt(ROOT/'baseline_reference/maxwell_B.csv',delimiter=',',skiprows=1)
xyz=baseline_r[:,:3]*1000
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False})

fig,axes=plt.subplots(2,2,figsize=(12.5,9.3),constrained_layout=True)
fig.suptitle('3 mm 双 camera 隔磁片尺寸验证 · 零线圈电流\n566 个共同空气点；尺寸为厚×宽×高(mm)，差分结合固定网格诊断',fontsize=15)
ax=axes[0,0]
values=[summary['baseline']['all']['relative_L2_percent']]+[cases[n]['total_B']['all']['relative_L2_percent'] for n in names]
bars=ax.bar(np.arange(4),values,color=['#8a98a8',*colors],width=.6)
ax.set(xticks=np.arange(4),xticklabels=['无片',*titles],ylabel='B 向量相对 L2 差 (%)',title='总 B 场：CuPy 对各自 Maxwell 细参考')
for bar,value in zip(bars,values):
    ax.text(bar.get_x()+bar.get_width()/2,value,f'{value:.2f}%',ha='center',va='bottom')
ax.set_ylim(0,max(values)*1.22);ax.tick_params(axis='x',labelsize=9)
ax=axes[0,1];x=np.arange(3)
signal=[cases[n]['delta_B']['all']['reference_RMS_mT'] for n in names]
error=[cases[n]['delta_B']['all']['RMS_vector_error_mT'] for n in names]
ax.bar(x-.17,signal,width=.34,color='#637e9c',label='Maxwell 场变化量 RMS')
ax.bar(x+.17,error,width=.34,color='#d36b30',label='CuPy 场变化量误差 RMS')
ax.set(xticks=x,xticklabels=titles,ylabel='mT',title='不同 FEM 网格直接作差：含参考离散影响')
ax.tick_params(axis='x',labelsize=9);ax.legend(fontsize=9)
for ax,group,caption in [(axes[1,0],'gap_A_side','A 侧空气线 x=5.2 mm'),
                          (axes[1,1],'gap_B_side','B 侧空气线 x=6.6 mm')]:
    mask=labels==group
    for name,title,color in zip(names,titles,colors):
        gpu=np.loadtxt(ROOT/name/'gpu_B.csv',delimiter=',',skiprows=1)
        ref=np.loadtxt(ROOT/f'{name}_reference/maxwell_B.csv',delimiter=',',skiprows=1)
        ax.plot(xyz[mask,2],(ref[mask,5]-baseline_r[mask,5])*1000,color=color,lw=1.8)
        ax.plot(xyz[mask,2],(gpu[mask,5]-baseline_g[mask,5])*1000,color=color,ls='--',lw=1.3)
    handles=[Line2D([0],[0],color=c,label=t) for c,t in zip(colors,titles)]
    handles += [Line2D([0],[0],color='#333333',label='Maxwell'),
                Line2D([0],[0],color='#333333',ls='--',label='MagTense/CuPy')]
    ax.legend(handles=handles,fontsize=8,ncol=2)
    ax.set(xlabel='z (mm)',ylabel='ΔBz (mT)',title=caption)
for ax in axes.flat:ax.grid(axis='y',alpha=.16)
fig.savefig(ROOT/'shield_accuracy_comparison.png',dpi=180);plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(12,8.8),constrained_layout=True)
fig.suptitle('场分量与隔磁片网格检查：总场误差与本构残差分开评估',fontsize=14)
mask=(labels=='near_top_0.5mm') & np.isclose(xyz[:,1],0)
indices=np.flatnonzero(mask);indices=indices[np.argsort(xyz[indices,0])]
for ax,name,title in zip(axes.flat[:3],names,titles):
    gpu=np.loadtxt(ROOT/name/'gpu_B.csv',delimiter=',',skiprows=1)
    ref=np.loadtxt(ROOT/f'{name}_reference/maxwell_B.csv',delimiter=',',skiprows=1)
    ax.plot(xyz[indices,0],ref[indices,5]*1000,label='Maxwell',color='#24394c',lw=2)
    ax.plot(xyz[indices,0],gpu[indices,5]*1000,label='MagTense/CuPy',color='#d36b30',ls='--')
    ax.set(xlabel='x (mm)',ylabel='Bz (mT)',title=f'{title} mm，顶面上方原 z=5.3 mm')
    ax.legend(fontsize=9)
ax=axes[1,1]
for name,title,color in zip(names+['Shield_T020_refined'],titles+['中间尺寸：0.4 mm 网格'],colors+['#87577e']):
    history=cases[name]['gpu']['iteration_history'] if name in cases else summary['middle_case_mesh_refinement']['history']
    ax.semilogy([p['iteration'] for p in history],[p['residual'] for p in history],label=title,color=color)
ax.axhline(1e-8,color='#555555',ls=':')
ax.set(xlabel='GPU 迭代次数',ylabel='真实相对本构残差',title='验收线 1e-8；0.4 mm 加密模型未通过')
ax.legend(fontsize=8)
for ax in axes.flat:ax.grid(alpha=.16)
fig.savefig(ROOT/'shield_fields_and_convergence.png',dpi=170);plt.close(fig)

def line(name,case):
    a,b=case['total_B']['all'],case['delta_B']['all']
    return (f"{name}: 总场L2差 {a['relative_L2_percent']:.4f}%，RMS差 {a['RMS_vector_error_mT']:.5f}mT，最大差 {a['max_vector_error_mT']:.5f}mT；"
            f"Delta B L2差 {b['relative_L2_percent']:.4f}%，RMS差 {b['RMS_vector_error_mT']:.5f}mT，"
            f"Maxwell变化量RMS {b['reference_RMS_mT']:.5f}mT。")
table='\n'.join(line(title,cases[name]) for name,title in zip(names,titles))
detail=[]
for name,title in zip(names,titles):
    case=cases[name];reference=case['maxwell'];gpu=case['gpu'];audit=case['independent_audit']
    native=json.loads((ROOT/f'{name}_native_record.json').read_text())
    shield=audit['per_object']['MID_SHIELD']
    detail.append(f"{title} mm\nMaxwell：{reference['passes']}轮，{reference['total_tetrahedra']}总四面体，能量误差{reference['energy_error_percent']:.6g}%，本次原生阶段{native['seconds']:.2f}s。\n"
       f"CuPy：{gpu['tiles']}磁性单元，{gpu['iterations']}步，残差{gpu['residual']:.9e}，独立CPU残差{audit['independent_CPU_material_residual']:.9e}，原生K/H差{audit['native_K_H_relative_error']:.4e}。\n"
       f"新增含片作用块构建{audit['new_operator_blocks']['new_block_seconds']:.2f}s，GPU本构迭代{gpu['solve_seconds']:.2f}s。\n"
       f"片内体积平均工作H：{shield['H_min_Apm']:.6g}–{shield['H_max_Apm']:.6g} A/m，B：{shield['B_min_T']:.6g}–{shield['B_max_T']:.6g} T，"
       f"超出原图最高H节点的单元{shield['above_last_measured_knot']}/{shield['tiles']}。\n"
       +'\n'.join(f"{group}: 总场L2差{case['total_B'][group]['relative_L2_percent']:.4f}%，Delta B L2差{case['delta_B'][group]['relative_L2_percent']:.4f}%，Delta B绝对RMS差{case['delta_B'][group]['RMS_vector_error_mT']:.6f}mT。" for group in ['near_top_0.5mm','far_top_3mm','gap_A_side','gap_B_side']))
refine=summary['middle_case_mesh_refinement']
if 'Shield_T020_refined' in cases:
    coarse=cases['Shield_T020'];fine=cases['Shield_T020_refined']
    refine_text=(f"粗细片网格的空气场彼此L2差{refine['field_difference_between_meshes']['all']['relative_L2_percent']:.5f}%，RMS差{refine['field_difference_between_meshes']['all']['RMS_vector_error_mT']:.6f}mT。\n"
       f"对Maxwell总场L2差：{coarse['total_B']['all']['relative_L2_percent']:.5f}% → {fine['total_B']['all']['relative_L2_percent']:.5f}%；Delta B L2差：{coarse['delta_B']['all']['relative_L2_percent']:.5f}% → {fine['delta_B']['all']['relative_L2_percent']:.5f}%。")
else:
    refine_text=(f"加密试验未收敛：{refine['iterations']}步后的真实残差{refine['last_true_residual']:.9e}，要求1e-8。未输出可接受的M或B场，未降低阈值，也没有再次提交该加密试验。\n"
                 '因此不能报告加密后的场误差或宣称网格收敛；本轮只能给三种主尺寸的有效解，并明确记录加密检查失败。')
original=Path(prepared['original_project']);source=Path(prepared['source_project'])
integrity={'original_project_sha256_before':prepared['original_sha256'],
           'original_project_sha256_after':hashlib.sha256(original.read_bytes()).hexdigest(),
           'previous_fine_reference_sha256_before':prepared['source_sha256'],
           'previous_fine_reference_sha256_after':hashlib.sha256(source.read_bytes()).hexdigest()}
assert integrity['original_project_sha256_before']==integrity['original_project_sha256_after']
assert integrity['previous_fine_reference_sha256_before']==integrity['previous_fine_reference_sha256_after']
(ROOT/'source_integrity_final.json').write_text(json.dumps(integrity,indent=2),encoding='utf-8')
elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(session['started_UTC'])).total_seconds()
paired_path=ROOT/'matched_mesh_delta_metrics.json'
paired=json.loads(paired_path.read_text()) if paired_path.exists() else None
if paired:
    pm=paired['all'];shift=paired['two_no_shield_Maxwell_references_difference']
    paired_text=(f"0.10×8.8×3.4mm额外去片对照：源与对照351031个四面体的角点坐标集合逐一一致（1e-12m取整指纹），总单元数相同；二次边中节点未单独比对。冻结源场重新导出与原先结果一致。\n"
      f"相同角点网格下，Delta B相对L2差{pm['relative_L2_percent']:.5f}%，绝对RMS差{pm['RMS_vector_error_mT']:.6f}mT，最大差{pm['max_vector_error_mT']:.6f}mT，参考变化量RMS{pm['reference_change_RMS_mT']:.6f}mT。\n"
      f"两份不同网格的无片Maxwell场彼此RMS差{shift['RMS_vector_error_mT']:.6f}mT。这直接说明首例76%的表观差分差异主要受参考解离散/作差敏感性影响。其他尺寸仍只给不同网格差分指标，不能继承此组9.08%的数值。\n"
      + '\n'.join(f"{g}: 同网格Delta B L2差{m['relative_L2_percent']:.4f}%，绝对RMS差{m['RMS_vector_error_mT']:.6f}mT，参考变化量RMS{m['reference_change_RMS_mT']:.6f}mT。" for g,m in paired['groups'].items())
      + '\n对照仅做一轮固定网格，能量误差0.3417%，不是独立自适应收敛序列。8min时已请求原生中止；原生收到指令后仍完成当前整轮，记录Normal Completion，ComEngine实测645s、批处理约667s。初始监控因中止标记将其拒绝；后续只读核查确认完整场与网格一致后，作为探索性诊断采用，没有再次求解。它不是严格Delta B误差界或实物精度证明。')
else:
    paired_text='未得到可接受的同网格去片对照，不能隔离参考网格差异对Delta B的影响。'
abort=json.loads((ROOT/'failure_snapshot_default6000/Shield_T010_native_record.json').read_text())
refinement_abort=json.loads((ROOT/'failure_snapshot_refinement24000/Shield_T020_native_record.json').read_text())
report=f'''3 mm 双 camera 隔磁片尺寸精度验证
日期：2026-11-07

本轮完成范围
三种隔磁片尺寸与无片基线，在566个相同空气点比较B向量；中间尺寸额外执行隔磁片面内网格加密检查，但加密模型未达到本构残差要求。两路线圈始终为零电流；只评估B场，不评估力或带电线圈。

结果（厚×宽×高，mm）
无片在本次566点上的总场L2差：{summary['baseline']['all']['relative_L2_percent']:.4f}%，RMS差{summary['baseline']['all']['RMS_vector_error_mT']:.5f}mT。
{table}
此处三组Delta B使用各自独立生成的FEM网格与无片参考作差，包含参考网格敏感性，不能直接等同于隔磁效应预测的真实误差。

指标定义
总场相对L2 = ||B_CuPy-B_Maxwell||_2 / ||B_Maxwell||_2。
Delta B分别在每个求解器内计算：B_有片-B_无片，再比较两个Delta B。相对误差以Maxwell Delta B为分母，同时报告绝对mT误差和参考变化量RMS。
RMS和最大差均为B向量的误差范数。相对L2不是逐点相对误差的平均值，不用近零场点制造夸大百分比。
两camera都含磁源；此处总场变化不是单个camera对另一个的严格串扰衰减率，也不等于隔磁片对磁力的衰减。

同网格差分诊断（仅0.10mm组）
{paired_text}

各尺寸详细结果
{chr(10).join(detail)}

中间尺寸的网格检查
片面内目标间距0.8→0.4mm，片单元330→1188，总磁单元3959→4817，camera网格与同几何Maxwell参考均保持不变。
{refine_text}
这只检查面内细化，两者厚度方向均为一层；不是严格误差上界，也未验证厚度方向收敛。

材料与物理条件
隔磁片采用VAC MUMETALL，2002 PHT-001公开图7（PDF第20页、印刷第18页）的静态磁化曲线。原图为1mm厚冲制圆环测试，非0.1/0.2mm薄箔实物测试。读取45个有效图示节点，H约0.106–714 A/m，B约0.0053–0.689T；读图误差估计约2–3%，不是经校准的不确定度。
加原点，H-B分段线性插值；超出最后图示节点后按恒定M延伸至H=200000 A/m。这个高场尾部是模型假设，不能把图示末端称为材料完整饱和值。
读图质量复核：原图多条曲线在约50–100 A/m附近交叠/交叉，不能始终按上下顺序认定材料。当前输入在30 A/m以上应视作未认证的高场近似，不能把全部45个图示节点称为精确厂家表。输入CSV和两个求解器中的节点未在求解后更改。三组CuPy隔磁片工作点都低于此范围；另核查三组Maxwell片内全域最大H分别21.2194、2.16591、6.26254 A/m，均位于可辨低场段，最大B分别0.63934、0.49868、0.59344T。200点中心采样的最大值不是全域最大值。该核查不等于整条高场曲线认证。
源：https://vacuumschmelze.com/03_Documents/Brochures/Product%20Information%20Soft%20Magnetic%20Materials%20and%20Semi-Finished%20Products.pdf
camera轭铁仍为上一轮VACOFLUX50厂家稀疏表；永磁Br=1.2T、recoil μr=1.05及方向保持；铜按μr=1。两种软磁材料的B-H曲线独立，未混用或按场结果拟合。
两求解器均采用单值、各向同性、无磁滞近似。厂家不同厚度、加工和退火状态不由本次模型覆盖，数值一致性不代表实物准确度。

几何、网格与求解设置
隔磁片中心(5.9,0,3.1)mm，法向X，与两侧camera不接触。外空气域固定为x[-24.4,36.2]、y[-24.4,24.4]、z[-18.6,24.8]mm，避免大片扩大自动Region而混淆比较。
原3条近场线与远场线共404点中，3个落入大片的点移到x=5.2mm；原中央间隙线替换为片两侧x=5.2/6.6mm各81点，共566点。所有情况均通过空气内检查。
Maxwell 2022 R2：0.5%能量误差和变化目标，3–8轮，连续2轮满足，NonLinearResidual=1e-5，B-H smoothing关闭；保存真实原生Normal Completion、网格、profile及收敛历史。
MagTense3.0.1 + CuPy FP64，RTX3090Ti。camera沿用3629个TET10角点线性化后的磁性四面体；规则隔磁片TET4严格覆盖对应CAD长方体，体积校验通过，每单元均匀磁化。
接收体积算子：远源4点，近邻/自作用256点，近邻因子1.5。隐式本构+有保护Anderson混合，保留原始本构残差验收和原生平均场自检；未裁剪特征值或降低材料磁导率。
只复用完全不变的camera-camera K块，其余作用块重新计算。几何、精度、积分配置及MagTense版本核对后才使用缓存；小模型完整重建与复用矩阵逐元素相同，原生K/H差3.75e-16。每个正式案例另通过原生随机磁化平均场自检和独立CPU本构复算。
该作用块复用由本验证目录的独立辅助脚本实现；不是宣称既有skill CLI自动支持任意几何局部更新。

计时、失败与预算
用户批准整组75min上限。报告生成时总墙钟{elapsed/60:.2f}min；Native与K计算有并行，不能直接累加各阶段时间当墙钟。
初次0.10mm CuPy在默认6000步停于4.095e-8，尚未满足1e-8；监控随后原生中止Maxwell，耗时{abort['seconds']:.2f}s，返回码{abort['returncode']}，中止请求返回码{abort['abort']['request_returncode']}。这些成本计入本组预算，完整失败记录和当时的原生工程/results在failure_snapshot_default6000。
同矩阵和材料的有界短检查在6508步达到9.9153e-9，CPU复算9.9152e-9，确认是默认步数上限不足。未放宽误差阈值。后续上限调整为24000步，并受同一总截止时间限制；首例采用该短检查的有效结果，未重复构建K。
Maxwell后续从保留的适应网格继续，第一次恢复实际从Pass3开始。额外加密试验未达阈值时，跨worker停止监控还中止了第二组原生参考（当时原生阶段{refinement_abort['seconds']:.2f}s，退出码{refinement_abort['returncode']}）；其工程/results在failure_snapshot_refinement24000。停止该加密试验，不使用其场；随后只继续独立且已有合格CuPy解的主尺寸参考。原生重启、导出、GPU失败与参数检查均保留，不能把恢复后的单阶段计时冒充整个案例耗时。
无片fine参考与其M、camera K来自已完成的前一轮；其首次Maxwell615s和K782s不计入本轮新构建，但明确作为复用前置成本。无片566点重新导出/求场的成本在本轮内。
GPU迭代时间与CPU K构建分开报告；两求解器的空间离散不同，本轮没有证明同精度整体加速。

精度边界与解释
1e-8本构残差不是1e-8场误差。Maxwell能量收敛也不是566个局部B点的严格误差界。
有片与无片使用不同FEM网格，作差会包含两个参考场的有限网格误差；小Delta B的相对误差可能明显大于总场误差。
隔磁片只做了一组面内细化；固定camera粗网格、近场积分误差、厚度方向离散和空气边界影响尚未独立穷尽。材料数据的读图、厚度和高场尾部假设也限制物理解释。
不得仅凭总场接近就宣称能准确预测隔磁片优化收益；应同时看Delta B的绝对/相对误差与细化后变化。

文件
comparison_summary.json：完整数值与分组指标。
shield_accuracy_comparison.png：总场与Delta B对比。
shield_fields_and_convergence.png：场分量与真实本构残差。
各Shield_*目录：模型、K缓存、M、空气B、逐点比较、独立复算与CLI日志。
各Shield_*_reference目录：真实Maxwell B/H、profile、收敛、材料回读审计。
source_integrity_final.json：用户原始工程及前一轮fine参考的SHA256前后一致。
paired_mesh_delta_check.png / matched_mesh_delta_metrics.json：0.10mm固定角点网格的额外差分诊断与限制。
curve_quality_audit.json：低场读图有效范围和高场曲线身份歧义，不改变本轮实际输入。
'''
(ROOT/'FINAL_REPORT.txt').write_text(report,encoding='utf-8')
print(table,flush=True)
print('REPORT:',ROOT/'FINAL_REPORT.txt',flush=True)
