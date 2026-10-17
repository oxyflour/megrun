# 多 camera 干扰工作流

入口是 `scripts/camera_interference.py`，支持 `prepare`、`propose`、`build`、`run`。
每轮冻结输入 JSON，从 AEDT 提取 camera 模板的实际几何与材料，按坐标装配 N 个 camera，用完整双向耦合低保真模型进行源贡献分析，
为贡献较大的可移动 camera 生成有界平移候选，调用 **gpt-6-luna** 选择候选，再生成可直接重新输入的 `next.json`。
`run` 重复上述过程；默认使用 Maxwell 验证每轮，`--surrogate-only` 用于离线预演。

当前支持一个六实体 camera 模板、任意 N、各 camera 独立电流和匝数、XYZ 平移。
N 省略时为 3；提供其他数量的坐标时必须显式设置 N。旋转和不同 STEP 混装会明确拒绝。
原有双摄脚本和冻结实验数据不变；本流程不使用失败的原生邻居场加载/响应库，也不恢复第二档参考加密。

## 输入与监视器

推荐示例：[从 AEDT 读取的 B 监视](../examples/camera_array_aedt_B.json)。
旧基准示例：[STEP B 点位监视](../examples/camera_array_B.json)、[STEP 线圈 F 监视](../examples/camera_array_F.json)。

### AEDT 作为几何和材料来源

输入必须明确选择 `aedt`、冻结的 `aedt_source` 或旧基准 `step` 三者之一；没有隐式 STEP 回退。
`aedt` 指定 `path`、已有磁静态 `design`、`template_prefix` 和 `origin_mm`。
例如示例从已验证三 camera 工程选择 `C1_` 六实体，减去 `[0,0,0]` 作为模板，随后用 JSON 坐标复制 N 份。
也可用 `objects` 字典显式映射六个逻辑实体名到实际 AEDT 对象名。
其他 camera 和空气域会保留在源工程实体清单中；当前不会自动把不同 camera 几何混装成共用模板。

读取流程只打开原工程的副本，不提交求解。每个实体从 AEDT 导出精确 CAD，保留原始导出件，
再组成内部几何中间文件 `template.step`；此文件来自 AEDT，而不是原有 STEP 输入。
导出与体素化后的实体包围盒、体积均对照 AEDT 核验。
材料按实际赋值读取相对磁导率、完整 B-H 采样与单位、带符号的矫顽力及全局方向，
保留原始材料属性。永磁源由 `M_r = -mu_r * Hc * direction` 计算，不固定 Br 或磁化方向。
当前原生设置 `SmoothBHCurve=false`，低保真保留分段线性 B-H；表外延伸采用末段斜率，属于粗模型近似。

当前适配器支持 mm、全局 Cartesian 坐标、各向同性线性材料或单值正常 B-H 曲线，以及矩形环绕组。
局部坐标、各向异性、材料表达式、温度依赖、平滑 B-H 和未适配几何会明确报错。
原生电流边界与端面信息会一并保存供审计；`current_A`、`turns` 在 AEDT 模式下由 JSON 显式给定，
因为一个 `Current=10A` 安匝边界无法唯一确定电流与匝数。新建电流保持已有 CCW +Z 约定。
原工程的其他激励与边界不自动复制为新的物理模型。

可直接从 AEDT 提取并准备候选（会启动 AEDT 读取，但不求解）：

```powershell
.\.venv\Scripts\python.exe scripts/camera_interference.py prepare `
  --config examples/camera_array_aedt_B.json --output outputs/my_aedt_prepare --budget-minutes 5
```

也可先单独提取，再在输入 JSON 中使用 `aedt_source` 指向 `source.json`：

```powershell
.\.venv\Scripts\python.exe scripts/extract_camera_aedt.py `
  --project path/to/input.aedt --design CameraArray --template-prefix C1_ `
  --output outputs/my_aedt_source --budget-minutes 5
```

源包包含来源工程哈希、原始 CAD、实体材料表、原生 B-H 点、激励记录、体积核验和完整计时。
后续 `next.json` 使用冻结源包及其哈希，不重复启动 AEDT 提取；CAD 或源包被修改会停止。
低保真和新建 Maxwell 工程共用相同材料与几何；重建后还解析保存的 AEDT 材料表独立核验，
不依赖 PyAEDT 内存缓存。切换来源只验证数据链，不能据此宣称低保真达到 Maxwell 精度。

`position_mm` 是 **camera 模板局部原点在世界坐标中的位置**，即导入几何的平移量，不能混用外壳边缘间隙或包围盒中心。
本 STEP 的 EM 包围盒尺寸为 8.8 × 8.8 × 3.4 mm；例如 `[0,0,0]` 与 `[10.8,0,0]` 对应沿 X 的 2 mm 边缘间隙。
`step` 相对输入 JSON 所在目录解析，冻结输出改用绝对路径，因此 `next.json` 可以直接重新加载。

|字段|定义|
|---|---|
|`cameras`|对象列表，或简写为 XYZ 坐标列表；每个 camera 有稳定的唯一 `id`|
|`current_A` / `turns`|AEDT 来源须显式给定；旧 STEP 模式默认 0.1 A / 100 匝，正电流从 +Z 看逆时针|
|`movable`|默认 true；示例固定 C1 作为装配基准|
|`move_axes`|默认 `["x","y"]`；可选择 XYZ 子集|
|`bounds_mm`|绝对坐标下限/上限，省略时初始位置 ±2 mm；写入下一轮后不重新居中|
|`search.step_mm`|每次平移的最大长度，默认 0.5 mm|
|`search.clearance_mm`|EM 包围盒最小分离量，默认 0.2 mm，保守拒绝相交包围盒|
|`search.top_cameras`|按源贡献选择的可移动 camera 数量，默认 2|
|`search.max_candidates`|每轮最多评估的轴向单 camera 平移候选，默认 12|
|`search.min_relative_improvement`|候选 score 最低相对降幅，默认 0.001|
|`search.stop_score`|达到即停止，默认 0；另受轮数与累计时间限制|

监视器有两种类型：

- **B**：`quantity="B"`，给出世界坐标 `position_mm`，单位 T，默认 `component="magnitude"`。
  可指定 `x/y/z`；点必须在所有 EM 包围盒之外，Maxwell 空气域会包含这些点。
- **F**：`quantity="F"`，给出 `camera_id`，表示该 camera **整个线圈的 Lorentz 力**，单位 N，默认 `component="z"`。
  F 没有独立空间点，不能只凭点坐标定义力；结果中的 `position_mm` 是随 camera 平移的线圈中心，仅用于定位显示。

每个监视器可以设置 `target`、`scale` 和 `weight`，目标函数为
`max_i(weight_i * abs(value_i - target_i) / scale_i)`。
`value` 是所选分量或矢量模；`target` 默认 0。负分量不会被误当成无限改进。
混合 B/F 时必须分别显式指定带本单位的 `scale`，避免直接比较 T 与 N。

**F 示例明确优化总 Fz 向零减少。** 如果任务是保持驱动力或减小相对某个基准的偏离，应把 `target` 设为相应的力值。
本流程没有自动把总力改成单摄增量，也没有把总力改善称为距离差分精度改善。
`Fz(g)-Fz(2 mm)` 的历史 10 μN / 5 μN 目标和尚未通过的结论仍保留。

## 运行

离线准备候选和子 agent 提示，不启动 AEDT：

```powershell
.\.venv\Scripts\python.exe scripts/camera_interference.py prepare `
  --config examples/camera_array_B.json --output outputs/my_array_prepare --budget-minutes 3
```

调用 Luna 选择已复算的改进候选，并写入 `next.json`：

```powershell
.\.venv\Scripts\python.exe scripts/camera_interference.py propose `
  --output outputs/my_array_prepare --budget-minutes 2
```

默认使用已登录的 Codex CLI，固定 `--model gpt-6-luna`、只读沙箱和 JSON Schema，提示经 stdin 传递。
不静默更换模型、不在失败后重试；提示、schema、JSONL 事件、stderr、模型请求和计时全部保存。
也可以把 `agent/prompt.txt` 交给当前会话的 Luna 子 agent，保存其 `{candidate_id, reason}` 响应，
再用 `--proposal response.json` 消费；外部响应的模型来源由调用者记录。
调用方式依据 [OpenAI 官方非交互文档](https://learn.chatgpt.com/docs/non-interactive-mode)。

完整离线闭环，最多移动两次：

```powershell
.\.venv\Scripts\python.exe scripts/camera_interference.py run --surrogate-only `
  --config examples/camera_array_B.json --output outputs/my_array_preview `
  --max-iterations 2 --budget-minutes 3
```

仅构建新 AEDT 工程、不求解：

```powershell
.\.venv\Scripts\python.exe scripts/camera_interference.py build `
  --config examples/camera_array_B.json --output outputs/my_array_build --budget-minutes 5
```

Maxwell 全闭环使用 `run` 并省略 `--surrogate-only`。
必须提供本次**全部串行工作**的 `--estimated-total-minutes`、`--estimate-basis` 和 `--budget-minutes`。
`--max-iterations K` 最多提交 **K+1 次 N-camera 求解**，估计必须包括全部建模、低保真、网格、求解、快照、提取和 Luna 等待。
已有本模型三 camera 两布局各约 25–26 min 的实测；改变模型或网格后仍需重新评估完整耗时。
估计不可靠且可能超过 30 min 时先向用户确认，不能试着提交。
预计或预算超过 30 min 时，只有用户明确批准本次方案后才能使用 `--confirm-long-simulation`。
该确认只对用户批准的具体批次有效。2026-10-16 的离线重构没有提交 Maxwell 求解；
2026-10-17 另行批准的两布局原生比较已完成，结果见下文。

## 原生装配与停止规则

`run_camera_array_native.py` 是内部的 build/audit/extract worker，没有独立 solve 选项。
每个 camera 导入完整六实体 STEP，平移所有实体和电流端面，保持原磁化方向和各自的 N·I 激励。
统一创建空气域、Lorentz 力参数和磁静态自适应设置。
`maxwell.mesh_policy="low-fidelity"` 时，使用本轮 N-camera 的 B 场按对象内部变化、法向磁通、轭铁饱和度排序表面加密，
保留实体和线圈长度约束；这是新规则，尚无多 camera 加速倍数或监视器精度保证。
AEDT 模式使用冻结源包中的原始材料属性；只有显式旧 STEP 模式使用项目基准 Br=1.2 T、μr=1.05 和示例单值 B-H 表。

原生构建现在逐实体检查 JSON 平移后的包围盒和 CAD 体积，并检查保存工程中的安匝、电流端面方向与 Lorentz 力参数。
Maxwell 磁静态设置显式使用 `SolveFieldOnly=False`；不能传入该版本不支持的 `SaveFields`，
因为 PyAEDT 0.8.11 会把它模糊匹配成另一个设置。
已有未求解工程可在保留原件后对副本执行内部 `--phase audit`，结果保存在 `build_audit.json`。

固定的两布局原生比较使用 `scripts/validate_camera_array_native.py`，它在一个累计预算内依次求解初始与冻结候选。
两个输入必须具有相同的冻结 AEDT 源包（或旧 STEP）、camera 数量与激励、监视器和 Maxwell 设置；首组失败或未原生收敛时，不提交第二组。
超过 30 min 的比较仍须先取得用户对该批任务的明确确认。2026-10-17 本次两组比较已另行获得 120 min 预算批准，
授权记录与运行数据在 `outputs/camera_array_native_20261017/`；这次批准不扩展到其他任务。
本次两组均在 5 轮原生自适应后正常完成，最大监视器 |B| 从 4.626867 mT 降至 3.890486 mT（15.9153%），
整组建模、求解、快照和提取累计 51.29 min。低保真预估降幅为 17.7511%，不能把它当作精确场解。
完整数据与限度见 [原生验证报告](../outputs/camera_array_native_20261017/validation_report_zh.md)。

整轮使用同一个墙钟截止时间。原生阶段预留 90 s 清理时间；预算触发即停止，不追加阶段。
求解中用 `desktopproxy -abort` 请求原生中止并记录返回码，未退出时明确记录可能仍在运行，不强行修改原生状态。
构建/提取异常保留日志和工程；新启动器单次等待 gRPC 最多 60 s，记录独立进程 PID、端口、退出码和 stdout/stderr，避免 PyAEDT 内置启动重试。
只有本轮创建、PID 与端口匹配的 desktop 可以被清理，不触碰其他工程的会话。

批处理返回后，**先复制并逐文件校验 `.aedt`、整个 `.aedtresults` 和 batchinfo，再重新打开工程提取**。
完整副本位于 `pre_extract_snapshot`，防止提取失败丢失场解证据；提取不得修改这个快照。
B 导出检查 SI 坐标与有限值，F 报告检查单位 N；原生 profile、convergence 和网格统计独立保存。
未达到原生自适应收敛时停止闭环并保留真实状态；若下一轮 Maxwell 指标没有改善，则停止并保留之前的 `best.json`。
原生自适应完成不构成局部场、总力或微小差分力的严格误差界。

## 输出与验收

`prepare` 目录包含 `config.json`、`analysis.json`、`low_fidelity_state.npz`、`candidates.json`、`agent/`、`run_record.json`。
`propose` 增加 `proposal_audit.json` 和 `next.json`。
`run` 为每轮保留 `iteration_NNN/`，并写出该保真度下已复算的 `best.json` 和累计 `run_record.json`。
新任务必须使用新输出目录，脚本不会覆盖原生工程或自动续跑失败任务。

源贡献使用**已耦合磁化状态**按永磁体、轭铁和绕组分解；矢量贡献之和等于该模型的监视量。
`frozen_removal_score_change` 只表示冻结其他源时移去一项的 score 变化，不是实际拆掉磁铁后的物理重求解。
候选位置全部重新求解 N-camera 磁化，因此保留两两双向反馈。
候选搜索是有界坐标轮询，没有全局最优保证；没有改进候选、未收敛、失败、超时或迭代次数到限均停止。

回归测试验证 N=2 对既有双摄磁化和线圈力的一致性、N=3 平移协变性、源贡献求和、
机械边界、B/F 单位和目标、预算门禁以及提取前完整原生快照。
离线重构与原生启动失败记录见 `outputs/camera_array_refactor_20261016/`；
后续原生构建修正、两布局比较与完整结果快照见 `outputs/camera_array_native_20261017/`。
