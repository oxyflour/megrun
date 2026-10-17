# 摄像头磁场与线圈力 MVP

新增 JSON 驱动的多 camera 工作流（N 默认 3），支持空间 B 监视器和指定 camera 线圈的 Lorentz F 监视器，
以及低保真源贡献分析、可选网格预加密、gpt-6-luna 位置选择和 JSON 迭代。
入口与预算约束见 [多 camera 工作流说明](docs/camera_array_workflow_zh.md)，输入示例在 `examples/camera_array_B.json` 和 `examples/camera_array_F.json`。
多 camera 的几何与材料现可直接从 AEDT 提取，推荐入口为 [AEDT 输入示例](examples/camera_array_aedt_B.json)。
实际实体、材料赋值、完整 B-H 表和带符号的矫顽力冻结到可校验源包，低保真与重建工程共用该源包。
STEP 示例显式保留为旧基准输入，不再隐式指定默认 STEP 或材料。
2026-10-17 的三 camera 原生对比已完成：两组均原生收敛，Luna 冻结布局使最大监视器 |B| 减少 15.92%；
原生工程、场解快照、完整计时和适用范围见 [验证报告](outputs/camera_array_native_20261017/validation_report_zh.md)。

已用 `B0500_scale_AF_VCM_EM_only.step` 实际运行。默认两个相同模组同向沿 +X 并排，线圈均为 **0.1 A、100 匝**，正电流从 +Z 看为逆时针。计算三个方向的线圈力、相对单摄的力变化，以及冷热启动复用性能。

这是用于物理特征和网格建议的低保真求解器。粗算法与 Maxwell 是两个独立的计算流程。本机 AEDT 2022 R2 位于 `C:\Program Files\AnsysEM\v222\Win64\ansysedt.exe`，已实际运行磁静态参考求解。正式参考输出在 `outputs/maxwell_verified/`；对比方法见 [docs/maxwell_validation.md](docs/maxwell_validation.md)。旧 `outputs/maxwell/` 含材料坐标次序错误等开发试算，不能用于精度结论。

**实际比较未通过距离增量验证。** 1–3 mm 五点扫距的总轴向力误差为 3.34%–3.55%；2 mm 同一网格上的力增量为 Maxwell −0.09743 mN、粗算法 −0.13863 mN，差 42.28%。粗网格细化后该增量误差仍为 44.19%。完整数据、收敛证据和工程链接见 [Maxwell 对比报告](outputs/maxwell_verified/validation_report.md)。

## 低保真引导 Maxwell 网格

当前重点是让低保真结果分配局部网格预算，线圈力由 Maxwell 求解。
`camera_mag/mesh_seeding.py` 用对象内部的 B 变化、面法向磁通、轭铁饱和程度和距线圈的距离选择加密面，设置 0.3/0.4 mm 目标长度和新增单元上限；其余区域保留较粗初始网格，再由 Maxwell 自适应检查。
这是未训练的规则，未使用 Maxwell 场或力作为网格选择输入。旧 `mesh_recommendations.json` 是早期全对象启发式输出，不是本次实际应用的网格规则；实际规则记录在各算例的 `mesh_plan.json`。

2 mm 间隙、同一 4 核配置的首次顺序对照：默认自适应 25.9 分钟/22 轮，统一 0.5 mm 加密 28.8 分钟/7 轮，低保真引导 21.8 分钟/6 轮。低保真引导相对前两组分别加速 1.18×、1.32×，未达到 1.5× 目标。耗时包含建模、求解与提取，低保真规则生成约 2.3 秒，使用已缓存的自作用算子。完整精度检查、原始日志与工程见 [网格对照报告](outputs/mesh_seeding_2mm/report.md)，实验条件见 [对照方法](docs/mesh_seeding_benchmark.md)。单模型单间隙的结果不代表跨模型收益。

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_mesh_seeding.py --output outputs/mesh_seeding_repeat --distance-mm 2
```

三个策略共用 0.5% 能量误差和每只线圈 20 µN 的逐轮 Fz 收敛条件。
更细的 Maxwell 参考解为 81.9 万个单元，能量误差目标 0.25%、逐轮 Fz 变化目标 10 µN。三组每只线圈的总 Fz 最大偏差分别为 0.0347%、0.0390%、0.0643%，均通过 0.2% 目标；该检查不验证微小横向力或距离干扰增量的相对精度。中文结果见 [网格试验总结](outputs/mesh_seeding_2mm/report_zh.md)。

后续激进加密试验将初始网格提高到 54.6 万单元，并显式加密邻近空气。首轮总 Fz 最大参考误差 0.1305%，已经低于 0.2% 目标；但能量误差 0.6166% 未过 0.5% 目标。第二轮增至 71.0 万单元，力误差 0.0471%、能量误差 0.3333%，跨轮力变化 28.1 µN 仍超过 20 µN，Maxwell 明确未自适应收敛。总耗时 26.9 分钟，未比默认 25.9 分钟更快。首轮阶段约 8.5 分钟，建模及初始网格另计约 2.6 分钟，尚未单独测量一轮完整流程。当前证据支持继续验证一轮总力精度，但不支持声称已实现一轮原生收敛或稳定加速。见 [激进试验报告](outputs/mesh_seeding_aggressive_2mm_v3/report_zh.md)。

现已单独实测同一激进规则的固定网格一轮流程：2 mm、546,205 个四面体，单线圈总 Fz 最大参考误差 **0.13050%**，建模、初始网格、求解和提取合计 **12.11 分钟**。与原默认流程的时间比为 **2.135×**，与原低保真六轮流程为 **1.803×**。这是相对更细 Maxwell 的总力精度验收；原生能量误差仍为 0.61659%，没有跨轮收敛检查，不能称为 Maxwell 自适应收敛。该流程也改变了停止规则，尚未隔离低保真排序本身的加速贡献。见 [一轮实测报告](outputs/mesh_seeding_onepass_2mm/report_zh.md)。

两种更省网格的局部盒方案都未通过：30.2 万单元 / 7.07 分钟的最大 Fz 误差为 0.579%，48.0 万单元 / 11.35 分钟为 0.528%。失败结果和各版源码快照均保留。

冻结激进规则后的跨距离验收也已完成：1 mm 一轮 15.13 min、最大总 Fz 参考误差 **0.12312%**；3 mm 一轮 11.91 min、误差 **0.10719%**。两者均未用本次 Maxwell 输出回调规则。各自细参考经过 5 轮、90.8 / 95.3 万单元，最终能量误差 0.16363% / 0.16224%，连续两次力变化 ≤10 µN。参考验证额外用了 93.66 min，单独记录。1 / 3 mm 尚无相同停止规则的完整时间基线，不能据此宣称同样加速比。

与 2 mm 相比的距离差分力，1 mm / 3 mm 最大绝对参考偏差为 **5.00 / 10.09 µN**。按当前参考差分值计算，最大相对偏差约 29%；参考自身的逐轮变化也不是严格误差上界。当前结果支持这三个点的一轮总 Fz 计算，仍不能把 0.2% 精度承诺扩展到微小差分力。完整数据和复现条件见 [三个距离的一轮报告](outputs/mesh_seeding_onepass_holdouts/report_zh.md)。

## 单摄 Maxwell 结果预测双摄距离效应

已做第二点的原型与对照：保存单摄 Maxwell 的 J、空气域 B 和磁性体 H，平移查询邻居场，再加入轭铁磁化的增量响应。
增量算子仍是近似体积模型，未使用双摄力拟合，也未建立 Maxwell 外场扰动响应库。

1 / 2 / 3 mm 相对 2 mm 的 Fz 变化，直接叠加单摄场的最大参考差 **35.25 μN**，加入响应修正后为 **15.84 μN**。
总 Fz 最大参考误差 **0.08694%**，但距离效应最大相对差约 **31%**，**尚未通过 10 μN 差分目标**。
响应离散从 0.7 mm 细化到 0.5 mm，只改变最多 0.70 μN；空气域插值另引入不到 0.05 μN 的差，不能把主要偏差归因于积分或插值。

新间隙查询可完全离线运行；本次单摄场采用正常中止后保留、经过审计的第四轮解，未达到原设定的完整自适应停止条件。
仅验证同一 B0500 模组、同向 +X 放置、0.1 A / 100 匝。完整结果、成本、原生文件与局限见 [实验报告](outputs/single_camera_reuse/report_zh.md) 和 [方法说明](docs/single_camera_reuse.md)。

```powershell
.\.venv\Scripts\python.exe scripts\predict_camera_pair_from_library.py `
  --gaps 1.25 1.75 2.25 2.75 --output outputs\single_camera_reuse\extra_gaps.json
```

## 运行

本目录已建立 `.venv`。重新安装可使用 `uv sync --extra test --extra maxwell`，依赖版本保存在 `uv.lock`。

```powershell
cd C:\Projects\magrun
.\.venv\Scripts\python.exe scripts\run_low_fidelity.py `
  --step "C:\Users\oxyfl\Downloads\B0500_scale_AF_VCM_EM_only.step" `
  --distance-mm 2.0

.\.venv\Scripts\python.exe scripts\benchmark_reuse.py `
  --step "C:\Users\oxyfl\Downloads\B0500_scale_AF_VCM_EM_only.step" `
  --distances-mm 1.0 1.5 2.0 2.5 3.0

.\.venv\Scripts\python.exe -m pytest -q
```

也可用 `uv run python ...`；仓库中 `models/` 保存了输入 STEP 的副本，可以替换上述绝对路径。

## 结果

单距离输出到 `outputs/single/`：

- `summary.json`：力向量、单摄基线、邻近模组引起的增量、收敛残差、时间、材料和几何参数。
- `state.npz`：两模组的 M、H、B、坐标和迭代历史。
- `features.parquet`、`features.csv`：逐单元物理特征。
- `field.vtk`：供 ParaView 使用的粗单元采样点，坐标单位 m，不是 FEM 网格。
- `field.png`：XY 投影与中心 XZ 切片。
- `mesh_recommendations.json`：按对象聚合的长度建议，来自未训练的物理启发式，不是 ML 预测。

扫距输出到 `outputs/sweep/`：`coil_force_sweep.csv/png`、`reuse_benchmark.csv/png`、`benchmark_summary.json`，以及每个间隙完整的单案例结果。

`total_N` 为线圈总力；`isolated_N` 为相同电流下单摄线圈力；`interaction_delta_N = total_N - isolated_N` 包含另一模组直接磁场、另一线圈磁场及本模组磁化响应变化。`direct_other_module_N` 单列直接来自另一模组和线圈的部分。导出的 CSV 力单位 N，曲线单位 mN。

## 距离与几何

`--distance-mm` 指**导入 EM 几何包围盒之间的 +X 间隙**。此文件实际包围盒为 **8.8 × 8.8 × 3.4 mm**，因此 2 mm 间隙对应 10.8 mm 中心距。完整相机外壳未包含在该 STEP 中；若实际外壳宽度不同，需要相应换算间隙。

通过 CadQuery/OCCT 保留六个实体名称并用实体内点判定离散，不以包围盒代替真实实体。边界单元使用 3³ 子采样，按 CAD 精确体积校正各对象总体积。默认每模组 520 单元。仅支持这类六实体、沿坐标轴的矩形环绕组基准；其他形状会明确报错，避免静默猜测。

## 数值模型与复用

永久磁铁 `Br=1.2 T`、回线相对磁导率 `1.05`；四块磁铁分别沿 -X、+X、-Y、+Y 磁化。轭铁使用示例单值 B-H 表和 PCHIP，没有滞回。铜默认 μr=1；材料 API 也支持 μr 小于 1 的线性材料。使用 `--br-t`、`--recoil-mu-r` 或 `--bh-csv` 替换基准参数；CSV 表头为 `H_A_per_m,B_T`。

粗单元使用有限体积高斯正则化偶极子：近邻作用平滑，远场恢复标准偶极子，自退磁仍为 N=1/3。裸点偶极子在薄轭铁中出现了数值伪磁化，因此不作为相机默认内核。数学细节与限制见 [docs/numerics.md](docs/numerics.md)。

自作用矩阵和几何缓存到 `.cache/`，键由 STEP 内容、离散设置、内核和精度决定，**不包含间隙**。两个相同模组共享同一矩阵。距离变化只更新跨模组作用和电流源场；热启动使用上个距离的磁化状态。独立计算单摄状态，用于受力基线和磁化变化对比。

非线性算法是局部退磁预处理后的阻尼固定点，默认加 Anderson 历史加速；可用 `--method fixed` 对比。收敛报告同时列出预处理残差和真实 B-H 本构残差。`--tol` 控制预处理残差，默认 1e-7。未收敛时不会宣称成功，CLI 返回非零状态。

绕组由铜实体的内外矩形及高度提取，以均匀截面分布的闭合绕线路径积分 `I dℓ × B`。磁化求解包含两个线圈的 Biot–Savart 源场；受力排除线圈自身电流场。默认截面 3×3 路径、每边 20 段，可用 `--coil-cross-samples` 和 `--coil-segments` 检查积分敏感性。

## 实际验证与下一步

解析偶极子场、两偶极子受力方向/对称性/d⁻⁴趋势、正负线性磁化率、B-H 插值、高斯算子能量稳定性、绕组解析场、力方向、矩阵缓存、热启动一致性和真实 STEP 导入/输出均有测试。

`outputs/verification.json` 记录本机实测的网格、绕组积分和 float32 敏感性。敏感性比较不代表对 Maxwell 的误差估计。全局场和线圈局部磁场仍受粗离散、材料近似和绕组假设影响。

Maxwell 接口实际导入相同 STEP，使用相同材料参数、安匝数和方向，导出 Lorentz 力、自适应力历史、收敛记录和场采样。SI 坐标、电流方向、安匝数和 B-H 映射均有实测检查。线圈力精度需同时考察总力与相对单摄的增量。网格加速已有单点对照；跨模型、扫距泛化、训练模型和 ROM 仍需后续独立验证。
