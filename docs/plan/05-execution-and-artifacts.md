# 执行步骤、工具与结果留档

返回 [计划提纲](accuracy-study.md)。本文件描述后续执行方式；本次文档编写不启动新 Maxwell 仿真。

## 1. M0：输入与能力盘点

1. 确认本机 Maxwell 可用版本与许可证、兼容的 PyAEDT 导出环境，以及独立的 MagTense/CuPy 求解环境。记录解释器绝对路径和依赖版本，不为复核旧结果盲目升级依赖。
2. 确认原始工程、历史细参考、网格、曲线、点集和结果的实际路径。缺失项标为待生成，不能假定 Git 中包含被忽略的 AEDT 数据库。
3. 对原工程计算 SHA256，在新工作目录建立派生副本，后续只修改派生副本。
4. 按 [共同材料模型](01-scope-and-materials.md) 冻结输入；建立 C00–C06 状态清单、统一点集、区域和霍尔轴。
5. 将 [验收定义](04-metrics-and-acceptance.md) 写入 `acceptance_config.json`，登记版本与 SHA256。
6. 读取可用显存和主存，确定分批顺序及每次运行的时间/迭代上限。计划不沿用历史任务的 75 分钟上限。

## 2. 先补齐最小验证工具

现有 CLI 已支持磁性网格/材料导出、GPU 求解和基础同点 B 对比。以下能力需要按需补充，不能把计划文件名或字段当作已经存在的命令：

| 能力 | 当前基础 | 待完成工作 |
| --- | --- | --- |
| 基础场比较 | `bridge.py compare` | 保留原指标，增加按 `point_id` 核对、分区和混合容差 |
| 霍尔场 | `gpu_B.csv` 的三分量 | 按 `probes.json` 投影，输出总场和隔磁增量 |
| 收敛汇总 | 原生日志、场数据、`gpu_solve_info.json` | 汇总细化参数、真实残差、场变化与预算 |
| 成对网格核对 | 历史角点集合诊断 | 补充连接关系、单元阶次和全部高阶节点映射 |
| C00 解析参考 | 既有离线 demo 仅为测试夹具 | 建立独立解析实现并记录来源；demo 不能当 Maxwell 参考 |
| 完整验收 | 历史对比 JSON | 生成带前置条件检查的 `acceptance.json`，区分失败类型 |

新工具放在本轮 `validation/maxwell_alignment/tools/` 或合适的求解器测试目录，复用既有读取和单位逻辑。只对新增的解析核、坐标匹配、投影/差分公式及零场处理补充有意义的测试。

## 3. 执行顺序

### M1：C00–C03

先运行 C00 检查解析场、方向、坐标和单位，再执行回弹永磁、线性软磁和多条非线性曲线。对代表用例核查原生/CUDA K 一致性。基础用例未通过时，停止依赖它的完整工程验收，先定位材料、几何、核函数或收敛问题。

### M2：C04

先重现历史无片结果，再建立新的 Maxwell 网格/边界收敛序列和当前求解器网格/积分序列。测点保持冻结，报告旧结果与新细化结果的关系。采用最终合格参考后，再判定 1% 总场和霍尔轴向场目标。

### M3：C05–C06

优先处理 0.20 × 8.8 × 3.4 mm 的已知加密未收敛问题；达到真实残差要求后，独立检查面内和厚度方向。为全部三种尺寸建立完整网格一致的 Maxwell 成对对照，并逐级收敛 Delta B。每种尺寸单独给出总场、关键区域、霍尔投影和增量结论。

### M4：报告

汇总必需状态，检查原工程哈希未改变，输出机器可读指标和可复核报告。达到时间/内存/迭代上限的案例保留状态和日志，不输出通过标记。C07 扩展扫描另行登记，不在首轮必需清单中。

## 4. 现有 CLI 命令模板

以下路径中的 `C:/path/`、设计名、求解设置名和运行目录均为待替换模板。应分别使用已确认的 AEDT 导出解释器和求解器解释器；命令不负责创建或收敛 Maxwell 参考设计。

从求解器脚本目录执行：

```powershell
Set-Location 'C:/Projects/magtense-gpu/skills/magtense-cupy-magnetostatics/scripts/aedt_magtense_bridge'

$aedtPython = 'C:/path/aedt-env/Scripts/python.exe'
$solverPython = 'C:/path/solver-env/Scripts/python.exe'
$caseFolder = 'C:/Projects/magtense-gpu/validation/maxwell_alignment/run_YYYYMMDD_HHMMSS/C04/magtense/grid_medium'
$pointFile = 'C:/Projects/magtense-gpu/validation/maxwell_alignment/run_YYYYMMDD_HHMMSS/inputs/points.csv'
$referenceProject = 'C:/path/derived_reference_copy.aedt'

& $aedtPython bridge.py export --project $referenceProject --design 'DESIGN_NAME' --setup 'SETUP_NAME : LastAdaptive' --out $caseFolder --points-csv $pointFile --max-tiles 10000

& $solverPython bridge.py inspect --out $caseFolder

& $solverPython gpu_solver.py --out $caseFolder --backend cupy --dtype float64 --operator volume --k-builder cuda --method proximal --tol 1e-8 --maxiter 24000 --near-levels 2 --near-factor 1.5 --max-tiles 10000 --cache-k

& $solverPython bridge.py compare --out $caseFolder --prediction gpu_B.csv --prefix gpu_
```

导出前已在派生工程中完成目标状态的原生求解，并核对材料和激励。首次端到端检查可从一个小用例开始。`bridge.py export` 的电流检查不会替代原生零电流回读；如软件仍保留零电流绕组对象导致检查阻止导出，先核查其值，再决定是否使用已有的 `--ignore-current-check`。

上述比较默认读取同目录的 Maxwell 参考场。正式验收必须确保其来自最终合格细参考，而非仅用于提取磁性离散的粗 Maxwell 解。若导出的粗网格和细参考分开维护，可用已有 `--maxwell-csv` 指向同点细参考：

```powershell
$referenceField = 'C:/path/converged_reference_B.csv'
& $solverPython bridge.py compare --out $caseFolder --prediction gpu_B.csv --prefix fine_ --maxwell-csv $referenceField
```

优先保存带 SI 坐标的六列 `x,y,z,Bx,By,Bz` 参考，并在比较前校验点 ID/坐标。当前命令输出基础向量指标；它尚不自动执行本计划的分区、霍尔、差分、完整网格审计和综合验收。

原生 K 对照只将 `--k-builder cuda` 改为 `--k-builder native`，其他设置保持相同。使用独立输出目录保存两路线最终场，避免后一次运行覆盖前一次结果。积分检查分别改变 `--near-levels` 或 `--near-factor`，网格检查使用不同输入目录。

## 5. 资源与缓存

FP64 完整 K 约占 `72*N^2` 字节，以下为十进制 GB，仅包括矩阵本体：

| 磁性四面体 N | K 大小 |
| --- | ---: |
| 5,000 | 1.80 GB |
| 10,000 | 7.20 GB |
| 15,000 | 16.20 GB |
| 20,000 | 28.80 GB |

还需预留 GPU 工作区、构建临时数组和主存。根据实测峰值选择网格批次；不能为了塞进显存就降低本轮 FP64 验收要求或修改材料。如果单一矩阵规模成为阻碍，先量化所需局部细化，再另立数值实现改进项，不能报告尚未获得的精度。

K 缓存只在几何、单元顺序、积分、数据精度及实现版本匹配时复用。材料改变且几何完全相同时可复用几何算子，但必须重新求解 M。原生/CUDA、不同细化等级和不同积分设置的结果分开留档。每次运行使用新目录，避免未收敛运行留下的旧场文件被误当成新输出。

历史验证辅助脚本曾实现不变块复用，但当前通用 CLI 没有任意几何局部块更新功能，不能在计划计时中假定已经支持。

## 6. 推荐产物结构

以下为执行时拟建立的目录，不表示本次已生成数据：

```text
validation/maxwell_alignment/
  tools/                       # 本轮补充的审计与汇总工具
  run_YYYYMMDD_HHMMSS/
    manifest.json              # 提交、依赖、工程/输入哈希、用例状态清单
    acceptance_config.json     # 冻结的目标及绝对/相对切换规则
    inputs/
      materials/               # 原始/标准化曲线与来源
      points.csv               # x_m,y_m,z_m，与 point_ids.json 对应
      point_ids.json
      regions.json
      probes.json
    C00/ ... C06/              # 每个用例按材料/方向/尺寸继续分状态
      maxwell/
        reference_series/      # 网格、边界和非线性容差序列
        paired_meshes/         # 有片/无片完整网格审计
      magtense/
        grid_coarse/ ...       # 每个目录保存模型、M、场和求解信息
      comparison/
        pointwise.csv
        regional_metrics.json
        hall_metrics.json
        delta_metrics.json
        convergence.json
        acceptance.json
      failure_records/
    summary.json
    report.md
```

CSV/JSON、脚本、关键网格、材料和报告按仓库既有约定保存；大型 K、原生结果数据库等遵循 `.gitignore`，在清单中记录本地路径、体积及哈希，不能把引用文件写成已经纳入 Git。每份最终场附带产生它的输入哈希和求解状态。

## 7. 失败定位与报告要求

| 现象 | 优先排查 |
| --- | --- |
| 单磁体符号/方向异常 | 单位、局部坐标、Br/Hc 符号、场分量顺序 |
| 材料响应差异 | 曲线分配、B/M 换算、smoothing、尾部和温度设置 |
| Maxwell 场随网格/空气域变化 | 参考网格、外边界、局部输出收敛 |
| 真实本构残差不达标 | 积分、单元质量、初值、步数及迭代算法 |
| 方程已收敛但场不稳定 | 磁性网格、薄片厚度、近邻/远源积分和几何线性化 |
| 总场接近但 Delta B 不接近 | 有片/无片完整网格、差分参考稳定性及低幅值分母 |

最终报告逐用例写明：共同物理模型、采用的参考和离散、全部收敛证据、总场/分区/霍尔/增量指标、最差点、失败与限制，以及通过的具体范围。精度是主结论；耗时只作为附带数据，注明首次构建、缓存、求解及导出的边界，不用数秒迭代声称同精度端到端加速。
