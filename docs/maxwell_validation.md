# Maxwell 参考验证

参考使用本机 Ansys Electronics Desktop / Maxwell 3D 2022 R2，执行文件为 `C:\Program Files\AnsysEM\v222\Win64\ansysedt.exe`。Python 粗算法的结果没有用 Maxwell 力拟合、缩放或替换。

本轮实际结果：五点总轴向力误差为 3.34%–3.55%，2 mm 同网格增量误差为 42.28%。当前 MVP 的距离增量未通过一致性验证。[完整实算报告](../outputs/maxwell_verified/validation_report.md)保留全部数值和来源。

## 对齐条件

- 两个相同 STEP 模组同向沿 +X 排列，间隙为 8.8 mm EM 包围盒的边缘间距。
- 每个线圈 0.1 A、100 匝，从 +Z 看逆时针。Maxwell 采用 stranded Current 的均匀化 10 A 安匝激励，直接从 J 导出反算安匝数。
- 永磁体 Br=1.2 T、回线 μr=1.05，四块磁铁朝向模组中心；Maxwell 矫顽力为 `-Br/(μ0 μr)`。
- 轭铁采用同一示例 B-H 表，在 H≤8000 A/m 范围密集采样粗算法的 PCHIP 曲线；最后原始节点之间使用表的弦线，末端以 μ0 斜率延伸。导出的材料表保存到各案例 `maxwell_bh.csv`。
- Maxwell 导入完整实体；粗算法使用高斯体积偶极子与矩形闭合绕线路径。两种离散方式存在误差，不能预设数值相同。

## 实际检查

`scripts/run_maxwell_case.py` 使用 PyAEDT 0.8.11 建模和导出。正式求解由 AEDT 自己的 `-ng -BatchSolve` 执行，批日志和每个设计的 profile 必须记录正常完成。参考采用 0.5% 能量误差门限、至少三轮、连续两轮满足收敛条件，材料非线性残差门限 1e-4。每个案例独立保留每轮的力，能量收敛与受力收敛分开记录。

导出检查包括 STEP 内容 SHA-256、场采样坐标的 SI 数值、线圈 J 的方向与安匝数，以及轭铁 B、H 与指定曲线的关系。Maxwell 的场插值会在材料边界和非线性区产生 B-H 偏差，因此记录中位数、95 分位和最大值，要求中位偏差小于 5%，并保留原始场。

开发时发现本机 AEDT 2022 R2 的 B-H 坐标对按 `[H, B]` 读取；旧 PyAEDT 示例所示 `[B, H]` 在这里导致极大磁导率。通过实测 B/H 发现并修正了此问题。旧开发目录有明确的无效标记，正式参考使用新的项目和结果目录。

## 力的比较

总力误差为 `(Fz_coarse / Fz_Maxwell - 1) × 100%`。另外计算 `ΔFz = Fz_pair - Fz_isolated`，单列其绝对误差和相对误差。由于 ΔFz 只有总力的不到 1%，总力接近不能单独证明距离效应准确。横向力也单独保留，避免用较大的 Fz 掩盖小分量误差。

2 mm 还建立 `Gap_2mm_OnlyA` 对照：保持相同几何，把 B 模组磁铁与轭铁材料置为 vacuum，B 线圈电流置零，导入双摄最终网格，关闭后续网格细化。在完全相同网格上比较 A 线圈的两个解，并核对四面体数量一致，减少做差时的网格偏差。

空气域采用实体外绝对 20 mm 留白；额外以 40 mm 留白重算 2 mm。边界检查的力差包含不同网格误差，应与自适应历史一起理解。粗算法另有 0.35 mm、截面 5×5 路径、每边 40 段的结果，用于检查其离散敏感性。

## 复现

在项目目录执行（AEDT 批求解完成后再导出）：

```powershell
uv sync --extra test --extra maxwell
.\.venv\Scripts\python.exe scripts\run_maxwell_case.py --phase build `
  --project outputs\my_reference.aedt --design Gap_2mm_CCW `
  --mesh-mm 0.4 --percent-error 0.5 --distance-mm 2

$taskSolve = Start-Process -FilePath 'C:\Program Files\AnsysEM\v222\Win64\ansysedt.exe' `
  -WindowStyle Hidden -PassThru -ArgumentList @('-ng', '-BatchSolve', `
  'Gap_2mm_CCW:Nominal:Reference', '-machinelist', 'list=localhost:1:4:60%:0', `
  '-Monitor', 'C:\Projects\magrun\outputs\my_reference.aedt')
$taskSolve.WaitForExit()

.\.venv\Scripts\python.exe scripts\run_maxwell_case.py --phase extract `
  --project outputs\my_reference.aedt --design Gap_2mm_CCW --distance-mm 2
```

五点扫距的建模脚本为 `scripts/build_maxwell_sweep.py`，正式对比脚本为 `scripts/compare_maxwell.py`。每个已接受案例有 `provenance.json`、`forces.json/csv`、`force_adaptive_history.json/csv`、`convergence.txt`、`mesh_stats.txt`、`profile.txt`、`current_check.json` 和 `constitutive_check.json`，并保留可在 Maxwell 打开的 `.aedt` 项目与 `.aedtresults` 场数据。
