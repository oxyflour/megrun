# MagTense + CuPy 非线性 GPU 静磁求解器（原型）

## 体积积分与材料隐式模式

对于高磁导 B–H 材料，可使用修复版模式：

```powershell
python gpu_solver.py --out out --backend cupy --dtype float64 --operator volume --method proximal --tol 1e-8 --cache-k
```

`volume` 在接收四面体内积分原生解析场：远源采用正权重 4 点公式，近邻及自作用默认均匀八分细化两层，得到 256 个接收点。近邻判据为中心距离不超过 `--near-factor`（默认 1.5）乘两体包围球半径之和。`--near-levels 0/1/2/3` 分别对应 4/32/256/2048 点。几何仍为原四面体，没有裁剪正特征值或修改材料。

`proximal` 隐式处理局部本构并以带保护的 Anderson 混合加速。最终验收始终复算原始 `M-F(H_external+K@M)` 残差，而不是仅看迭代更新量；默认上限 6000 步，可用 `--maxiter` 指定。支持相同几何热启动及匀强外场。该模式解决强本构刚性，不能代替网格/积分精度验证。

兼容模式 `--operator centroid --method newton` 仍为 CLI 默认值，便于重现旧结果；Newton 的步长判据已改为实际残差的 Armijo 下降，避免变化的归一化分母误导线搜索。原有 Picard 模式保留。

体积 K 和重心 K 分开缓存；体积缓存还核对积分设置及 MagTense 版本。体积模式保留原生平均场校验，重心模式保留原生重心场校验。两种校验都不证明对 Maxwell 的物理精度。

兼容默认 `--k-builder native` 仍在 CPU 构 K。新增 `--k-builder cuda --backend cupy` 将四面体的三角面磁荷解析积分和接收点平均搬到 GPU，保持相同的四点/256 点积分、近邻判定和材料方程。近邻核以一个 warp 合作计算一个源/接收体对，不再按接收体逐次调用 Fortran。坐标、解析函数及归约始终使用 FP64；`--dtype float32` 只改变矩阵存储。没有启用 fast-math、偶极近似或谱裁剪。几何准备和近邻表仍在 CPU，完整 `3N×3N` 存储规模不变。

```powershell
python gpu_solver.py --out out --backend cupy --dtype float64 --operator volume --k-builder cuda --method proximal --tol 1e-8 --maxiter 24000 --cache-k
```

CUDA 体积模式通过 CuPy NVRTC 编译随包提供的 `tetra_cuda.cu`，无需修改或重新编译 MagTense。`--k-builder cuda` 要求 `--operator volume` 和 CuPy；不可用时报错，不会悄悄切回 CPU。首次编译、磁盘编译缓存、K 矩阵缓存是不同成本。CUDA 矩阵采用独立 `_cuda.npy` 文件，元数据核对 CUDA 源码哈希、几何和积分设置；原 CPU K 缓存不会冒充 GPU 构建结果。

历史 CPU 路径的 3629 个磁性四面体构 K 约 13 分钟，GPU 材料迭代约 3.75 秒。新 GPU K 路径的阶段计时、全矩阵比较及 Maxwell B 场对比见 skill 的 `references/validation-gpu-k-20261107.json`；不同离散精度之间的速度不能作为同精度加速比。

局部更新满足 `prox(z)+τ*H_material(prox(z))=z`，故固定点仍满足原始本构；饱和段以原曲线的恒定 M 边界处理。方法背景可参阅 [Parikh 与 Boyd 的 Proximal Algorithms](https://stanford.edu/~boyd/papers/prox_algs.html)。

本目录包含 **CPU 原生 / CUDA 解析四面体退磁张量 + CuPy GPU 非线性迭代** 实现。它直接使用之前 `bridge.py export` 生成的 `model.json`、`mesh/*.npz` 和 `points.csv`；无须重新在 Maxwell 划网格或修改数据格式。

**已验证**：原离线数学测试、真实 MagTense 原生库、CuPy FP64/FP32，以及 Maxwell 2022 R2 的实际网格/场导出。修复版的 3 mm、零电流、VACOFLUX 50 厂家表双模组基准使用 3629 个磁性四面体，对 485 个空气点的 B 向量相对 L2 差约 3.66%；这不是通用精度保证或实物验证。见 skill 的 `references/validation-20261107.json`。运行时保留与所选接收方式一致的原生 K/H 自检。

## 无矩阵 FMM 模式

`--operator volume --k-builder fmm` 不生成完整 K，入口是同目录的 `tetra_fmm.py`。兼容默认 `--fmm-farfield cpu` 调用 [FMM3D](https://fmm3d.readthedocs.io/en/latest/)。新增 `--fmm-farfield cuda --backend cupy` 使用当前源码的 Chebyshev 插值 FMM：GPU 构建紧凑 Morton 树并计算 P2M/M2M/M2L/L2L/L2P，邻近叶盒做直接点偶极子求和，固定网格复用树和交互索引；不把单个中心偶极子树算法当作 FMM。配合 GPU 近邻、解析近场修正和材料迭代，求解主计算都在 GPU，**原生 Windows 可运行，无须 WSL、JAX 或 FMM3D**。文件读写、几何准备和独立原生参考检查仍使用 CPU。

```powershell
python -m pip install "fmm3dpy>=2.1,<3"
python gpu_solver.py --out out --backend cupy --dtype float64 --operator volume --k-builder fmm --method proximal --tol 1e-8 --anderson-history 6 --max-tiles 3000000
# 原生 Windows GPU 主计算路径，不需要安装上面的 fmm3dpy：
python gpu_solver.py --out out --backend cupy --dtype float64 --operator volume --k-builder fmm --fmm-farfield cuda --fmm-neighbours cuda --fmm-eps 1e-6 --method proximal --tol 1e-8 --anderson-history 6 --max-tiles 3000000
```

此命令提高输入单元数限制，**不代表 300 万单元已完成容量或性能验收**。先用小模型确认近场数量、内存和误差，再扩展。依赖缺失时直接报错，不回退到稠密 K。FMM 不接受 `--cache-k`。

每个源四面体默认使用四点正权重体积积分形成磁偶极子。FMM 返回这些偶极子的 H 场，再对校正半径内的单元对减掉偶极近似、加回解析四面体积分。`--near-factor`、`--near-levels` 仍定义原体积算子的四点/256 点接收规则；扩大 FMM 校正半径只替换更多源积分近似，不改变这两种接收规则。CUDA 分批近场先计算面电荷，直接归约三分量 H，不逐对生成九分量张量；预条件器仍独立保留原解析自作用块。CPU 远场模式的空气测点走 FMM；CUDA 模式对少量空气测点分批直接计算 GPU 点偶极子场，再做 GPU 解析近场修正，记录 `cuda_direct_batched`，其成本为 O(N×测点数)。原生参考检查按源分批，避免分配 `N×测点数×9` 张量。

| 参数 | 默认值与作用 |
| --- | --- |
| `--fmm-eps` | `1e-10`，CPU 为 FMM3D 请求精度；CUDA 为每次场计算与独立直接点求和的抽样相对 L2 验收阈值，不是全域误差界 |
| `--fmm-farfield` | `cpu/cuda`，兼容默认 `cpu`；`cuda` 必须配合 CuPy，不静默回退 CPU |
| `--fmm-gpu-order` | CUDA 每轴插值阶数，默认 `ceil(-log10(eps))+4`，范围 3–16；提高阶数会增加显存和运算量 |
| `--fmm-gpu-gib` | `12`，CUDA 远场工作区预算；与近场缓存预算独立，超限报错 |
| `--fmm-gpu-leaf-size` | `512`，CUDA 叶盒平均源点数目标，平衡直接求和与 M2L 工作量 |
| `--fmm-neighbours` | `auto`，CuPy 有 KDTree 时使用 GPU 建树及紧凑计数/填充查询，否则用 CPU；`cpu/cuda` 显式选择，`cuda` 缺少依赖时直接报错 |
| `--fmm-source-levels` | `0`，每个源为 4 点；`1/2` 为 32/256 点，增加内存和计算量 |
| `--fmm-correction-factor` | `3`，校正距离为两体包围球半径和的倍数，必须不小于 `--near-factor` |
| `--fmm-near-cache-mib` | `512`，近场块缓存预算；超出后按接收体批次重算，`0` 强制分批模式 |
| `--fmm-index-gib` | `4`，主机端近场 CSR 索引的存储预算；超出则每次场计算分批查询近邻，只常驻 O(N) 行指针 |
| `--fmm-verify-rtol` | `1e-4`，求解前随机磁化、求解后实际磁化及空气场的原生抽样场误差阈值 |
| `--anderson-history` | `20`，大型模型可减小以控制历史向量内存；`0` 关闭 Anderson 混合 |

近场高阶接收点按 `--batch` 生成，不保留全体 `N×256×3` 坐标。GPU 查询复用 CuPy KDTree 的树布局，自定义 FP64 半径遍历先计数再填充，按最多 400 万对的查询批次分配，不使用接收体数乘源体数的稠密查询数组；超预算时缩小批次，单行仍超预算则报错，不截断邻居。算子记录 `fmm_neighbour_backend`。缓存近场索引需要 O(P) 存储，P 为校正单元对数；超出索引预算后，索引和近场块一起分批重算。极端网格尺寸差、重叠或过大的校正半径可能使 P 很大，导致很高的计算成本。空气点使用体校正系数的四倍作为解析修正半径，另记录 `fmm_air_correction_factor`；单元数乘测点数不超过 200 万时检查全部空气测点，大模型抽样检查，实际点数写入 `fmm_air_native_reference_points`。缓存预算**不是整个进程的峰值内存限制**。CPU FMM3D 每次场计算重建树；CUDA 路径复用树，并按偏移类别分批应用 M2L，不常驻全部平移矩阵。尚无分布式 FMM；不能从小模型外推百万单元的耗时。

`eps` 不包含源四面体积分近似误差。原生抽样不通过时，应加密源积分或扩大校正半径，并复查场结果，不能只降低 `eps`。`gpu_solve_info.json` 的 `residual` 是 **FMM 近似算子的本构残差**；`fmm_native_sample_constitutive_residual` 单独报告原生体积算子的抽样本构残差及是否达到 `--tol`。`full_native_constitutive_residual_verified=false` 表示没有进行全量 O(N²) 原生残差认证。抽样场检查也不是全域误差界或 Maxwell 精度认证。

CUDA 的原生体积场抽样检查复用同一源树、深度和插值阶数，只对选中的接收体建立目标子树，并按同一近场规则求值；不会为五个抽样接收体重算全模型近场。缓存块按选定行提取，分批模式按实际接收体编号计算。`fmm_verification_uses_full_matvec=false` 及 `fmm_verification_target_points` 记录此路径；独立原生参考和验收阈值保留。

解析源面共面的空气测点，即使原生核给出有限值，也强制进行双方向 Richardson 极限验证，防止有限的错误近场值被 FMM 远场掩盖。FMM 与独立原生参考均核对偏置点确实位于空气中；偏置求值按批执行，验收容差与原空气场求值器一致。

已有测试覆盖解析自作用、缓存/分批一致性、源积分细化、旋转/平移/缩放、两条 B–H 曲线、Newton/proximal、FP32/FP64 CUDA 修正、空气测点及命令行输出。4096 个源积分点另与原生抽样场核对。真实 3629 单元网格的随机磁化检查：默认四点源、校正因子 3、FP64、`eps=1e-10` 下，五个接收体抽样 H 相对差约 `3.45e-7`；32 点源约 `2.19e-8`。这是算子抽样检查，不是百万单元性能或非线性最终场验收。

2026-11-09 的 3629 单元非线性完整流程另通过 FMM 本构残差 `9.49e-9`、全部 485 个空气测点与原生场的相对差 `2.90e-6` 检查；最终 B 相对历史原生解的向量 L2 差为 `2.98e-6`，相对已有 Maxwell 参考的 L2 差为 `3.6550866%`。原生算子的抽样本构残差为 `4.11e-4`，**未达到原生 `1e-8` 残差等价验收**。参数为四点源、校正系数 3、`eps=1e-6`、原生解热启动；计时与 AEDT 大网格生成重叠，不作为性能基准。输入哈希、阈值、复现入口和场/磁化数据见 [FMM 验证记录](../../../../validation/20261109_fmm_refined/fmm_verification.json) 及其引用的 `fmm_small_model.npz`。

原生 Windows CUDA 远场另用相同真实网格的全部 14,516 个积分点，与 `eps=1e-12` 的 CPU FMM3D 点场对照：10 阶插值、512 点叶盒目标下，保存的物理磁化和随机磁化的全点相对 L2 差分别约 `4.08e-9`、`2.28e-9`，均通过 `1e-6` 阈值。复现入口为 `validation/20261109_fmm_refined/run_fmm.py --validate-cuda-small`，结果和源码哈希位于上述验证记录的 `cuda_point_validation`。这是 GPU 点远场检查，不是新的非线性或 Maxwell 精度认证。271.9 万磁性单元的当前实际尝试统一写入同目录 `large_fmm_run.json`；中断、失败与收敛分别标注。

2026-11-10 的 2,718,864 单元 GPU 非线性实跑因实际性能不实用而停止（`stopped_for_impractical_runtime`），不是内存耗尽或已证明的数值发散。约 7 小时 53 分钟完成 8 次完整场计算、4 次材料更新；残差 `2.952195 → 2.826249 → 2.710774 → 2.544746 → 2.459533`，未达 `1e-8`。最近一次场计算 3525.3 秒，其中远场 315.8 秒、近场 3209.5 秒（91.0%）。70.92 亿校正对的近场块超出缓存预算而逐次重算；近处每对使用 256 点接收积分。Anderson 候选检查还需要额外场计算。进程树采样峰值 RSS 4.41 GiB、提交内存 13.90 GiB，整卡显存峰值 11.37 GiB（含其他应用）。这只证明内存可控及重复求场能执行，**没有通过实用收敛验收，也没有证明相对 Maxwell 的端到端优势**；不能据此承诺百万单元生产求解能力。完整性能评估、停止原因及运行期间的源码哈希修订均保存在同目录 `large_fmm_run.json`。

近场压缩另做了有界局部检查：从该真实大网格的磁轭、永久磁体各取两块，每块 128 个四面体。六分量对称存储的差异为浮点舍入量级；六分量 FP32 对未压缩 FP64 局部近场校正的向量 L2 相对差为 `2.24e-8–2.87e-8`，FP16 为 `1.88e-4–2.23e-4`，逐对缩放的 int16 为 `1.37e-5–1.76e-5`。六分量 FP32 的全配对系数体积仍约 158.5 GiB，显式索引另约 26.4 GiB。四个局部 384×384 校正矩阵达到 `1e-4` 相对 Frobenius 误差所需 SVD 秩为 382–384，整体低秩存储不划算；这不代表空间分离块或分层表示的压缩能力。参数、原始网格和源码哈希见 `fmm_verification.json` 的 `near_compression_probe`。这里只检查局部系数及近场校正，尚无全模型压缩、最终场/非线性精度或压缩后吞吐验收。

固定网格的自适应分块实验复用当前 `tetra_fmm.AdaptiveNearCompression` 和同一大模型导出，入口为：

```powershell
C:/Projects/magtense-gpu/skill-test/.venv/Scripts/python.exe -u validation/20261109_fmm_refined/run_fmm.py --compress-near
```

默认在十个磁性部件各取四个接收块：两个固定种子的随机锚点，以及最小、最大包围球单元锚点；每块最多 128 个接收四面体，包含大模型中跨部件的**全部**近邻源。补零矩阵超过 256 MiB 或配对超过 200 万时缩小接收块，不删近邻。该实验不构建全模型缓存。压缩对象是 FMM 近场校正系数（解析四面体作用减去已包含的点偶极作用）；按空间中位数递归分块，自项所在叶块保持原值，其他块做自适应秩 SVD，不省空间则继续分裂或保留原始稀疏数据。默认 `--compress-admissibility weak` 允许测试接触但无共同四面体的块；`strong` 仅测试满足包围盒分离条件的块。

默认 `--compress-error-budget global` 将整个接收块的平方 Frobenius 误差预算按非零配对数分配给子块，保证各叶块误差平方之和不超预算；允许小修正块取秩零。`relative` 则要求每个子块独立满足相对误差，通常更保守。误差容差默认为 `1e-3 1e-4 1e-6`；构建后独立重建矩阵核对 Frobenius 误差，并对随机、匀强两组留出磁化核对近场向量 L2 差，超阈值以非零退出码失败。固定形状、采样参数、输入和源码哈希、逐块结果与资源记录写入现有 `fmm_verification.json` 的 `adaptive_near_compression_runs`。

离线 SVD 默认 `--compress-decomposition cpu` 使用四线程 LAPACK，亦可显式选 `cuda`；系数生成和复用乘法均在 GPU。非零低秩因子打包后用两个 CUDA 内核完成右乘及左乘累加。存储统计包含非零系数/因子、索引、块目录和打包内核工作区，比较基准为真实非零 FP64 BSR，而非补零后的稠密矩阵；另列六分量 FP32 BSR 基准。计时包含同步及 Python 调用开销，分别记录系数生成、压缩构建、预热后的乘法以及当前逐次积分路径。本实验的采样误差和吞吐不代表全模型、非线性残差或 Maxwell 验收，压缩表示尚未接入生产求解器。

2026-11-10 完成上述默认配置的 40 块实测，覆盖全部十个磁性部件、5099 个不同接收单元（大模型的 0.1875%），累计 13,374,243 个配对（包括块间重复）。所有块均保留 128 个接收单元及其全部全局近邻；总实验耗时 352.88 秒，FP64 系数生成合计 6.86 秒。以下存储量是这些采样块的合计，不能当作全量网格的压缩率保证：

| Frobenius 容差 | 相对 FP64 BSR 压缩倍数 | 最大系数相对 Frobenius 误差 | 两组磁化最大近场相对 L2 差 | 压缩构建合计 | 40 块预热乘法合计 |
| --- | --- | --- | --- | --- | --- |
| `1e-3` | 8.490 | `8.45e-4` | `1.032e-3` | 58.86 s | 11.98 ms |
| `1e-4` | 4.104 | `9.28e-5` | `1.2495e-4` | 62.20 s | 13.47 ms |
| `1e-6` | 1.224 | `8.97e-7` | `1.046e-6` | 172.96 s | 40.97 ms |

样本原始 FP64 BSR 合计 1,016,483,748 字节；`1e-4` 格式为 247,708,984 字节，其中 8.84% 的配对属于取秩零的块、4.30% 保留原系数，其余由非零低秩因子表示。相对六分量 FP32 BSR，三个容差档的压缩倍数仅为 3.128、1.512、0.451。未压缩 FP64 缓存的同口径乘法合计 10.79 ms，当前逐次积分路径为 6.08 s；因此该原型改善存储与复用，但压缩乘法没有快于现成的未压缩缓存。小块的重复预热计时不能外推整模型、远场和非线性总求解时间。进程峰值 RSS 1.22 GiB、提交内存 2.32 GiB，CuPy 最大预留池约 602 MiB（不是整卡显存峰值）。矩阵独立重建、两组留出磁化检查均通过；当前源码回归为 94 passed、1 skipped，跳过不计通过。

FEM–BEM 替代路线的全量网格统计可用同一入口的 `--assess-fem-bem` 复现：实际 271.9 万磁性四面体对应 522,922 个 P1 节点、7,242,130 个稀疏结构非零元（FP64 CSR 本体约 84.87 MiB），闭合材料表面有 235,652 个三角形。P1 势与 P0 边界通量合计 758,574 个未知量，但单块稠密 BEM 仍需 413.74 GiB，必须采用加速边界算子。计数和拓扑断言见同一验证 JSON 的 `fem_bem_feasibility`；[路线、参考原型与验收条件](../../references/architecture.md#fembem-replacement-topology-and-small-cpu-prototype) 区分全量统计、小模型耦合与加速求解实测。P1 节点数量和 CSR 本体大小不等于 Maxwell 高阶精度或完整求解显存。

### FEM–BEM 小模型参考原型

当前源码的 `fem_bem.py` 实现 FP64 的 P1 势 / P0 外侧法向导数耦合，复用现有 B(H) 及一致切线，采用阻尼 Newton–GMRES。外部衰减条件固定势基准，保留独立连通体的相对磁势。默认稠密 CPU 参考后端使用稀疏 FEM 加 Robin 项的 LU，以及边界 V 的 Cholesky；拒绝超过 4096 面或 256 MiB 预算数组的模型，该预算不含稀疏分解填充与积分临时空间。新增 `boundary_backend='cuda_hmatrix'` 采用全局 P0/P1 试探自由度的精确近块与 ACA 远块、对称单层算子及叶块边界预条件，FEM 可选 CPU PyAMG；[大模型命令、缓存及验证范围](../../references/architecture.md#cuda-hierarchical-boundary-and-sparse-amg)。这是混合 CPU/CUDA 实现，大模型成功仍须实测。

从仓库根目录运行已有验证入口：

```powershell
& .\skill-test\.venv\Scripts\python.exe -u validation/20261109_fmm_refined/run_fmm.py --prototype-fem-bem
```

该命令先做永磁球和非线性钢球的解析网格收敛、界面和预条件验收，再使用保留的 3629 单元真实模型及全部 485 个 Maxwell 空气参考点，检查积分阶数 8/12 的结果。可加 `--fem-bem-synthetic-only` 只做解析案例；这不能算 Maxwell 验证。运行会追加 `fmm_verification.json` 的 `fem_bem_prototype_runs`，保留失败证据并以非零退出码报错，不改写历史 FMM/Maxwell 结果。常规断言见 `tests/test_fem_bem.py`，另覆盖 CUDA 压缩/稠密对照、缓存损坏拒绝、细小平移面元积分及多连通体高磁导率 AMG。

此前真实案例两种积分阶数均在 3 次 Newton 更新后收敛，空气场相对 Maxwell 的 L2 误差约 3.93%。但粗 P1 网格的逐面原始法向通量差约 79%，还不能认证局部界面精度；积分阶数敏感性、耗时、内存和完整限制统一见上述架构说明。该历史结果保留原源码哈希，不代表后续细面元修复或压缩后端的精度。

直接读取实际 AEDT 案例并对照空气、材料内部 B/H 和界面两侧法向 B，可从仓库根目录运行：

```powershell
& .\skill-test\.venv\Scripts\python.exe -u validation/20261109_fmm_refined/run_fmm.py --validate-fem-bem-aedt
```

该入口不运行合成球体，不重新提交 Maxwell 求解；会启动已安装并授权的 AEDT 2022 R2，审计现有 `full_pair_reference.aedt` 的激励、材料、几何和收敛记录后重新导出查询场值，前后核对工程哈希。结果及全部比较向量追加到 `fem_bem_aedt_runs`；空气 B / 材料中心 B / 界面法向 B 的初始相对 L2 门槛为 5% / 10% / 10%，失败保留证据并以非零退出码结束。[完整范围与复现说明](../../references/architecture.md#direct-aedt-case-verification)。

此前粗网格直接 AEDT 验证**未通过**：空气 B 误差为 3.93%，但材料中心 B 为 38.58%、BEM 边界法向 B 为 38.40%。Maxwell 参考自身的界面两侧法向 B 差约 11.9%；将探针距离放大 10/100 倍，BEM 对照差仍约 38.3%–38.4%。补查命令为同一入口的 `--diagnose-fem-bem-aedt`，无需重复边界组装。该记录保留原源码哈希；钢材中心参考采样尚未覆盖 300 A/m 以上的 B–H 分支。

完整 2,718,864 磁性单元的 P1/P0 原型现已完成一次通过既定采样门槛的求解：CPU FEM/PyAMG、CUDA H 边界，FP32 系数存储及 FP64 积分/迭代/累加，并对 119 个误差块重算完整 K 系数。4 次 Newton 更新后弱残差 `5.84e-10`，原始边界算子抽查残差 `6.94e-8`；缓存流程 17 分 22 秒，其中非线性求解 13 分 31 秒，整卡采样峰值 14.71 GiB。另一次边界首次构建耗时 41.81 分钟、整卡峰值 15.37 GiB；这是不同阶段实跑，不能冒充一次完整冷启动时间。复现命令及缓存行为见[完整网格实测](../../references/architecture.md#cuda-hierarchical-boundary-and-sparse-amg)。

后续在同一完整 AEDT 网格上，共享 ACA 行中重复的 P1 面片积分，并在 GPU 跳过已完成块的积分，将 H 首次装配/压缩从 **41 分 49 秒降至 27 分 20 秒，减少 34.64%**。积分阶数、FP64 计算及误差设置不变，新旧完整数值缓存的二进制 SHA-256 相同；独立算子检查和保留收敛解的 Maxwell 采样复查通过。本次未重新运行非线性迭代。默认自动选择共享行路径；分块对比可用 `run_fmm.py --benchmark-fem-bem-assembly` 复现，完整边界基准及保留解检查见上述设计说明。

本次空气/材料 B/边界法向 B 对保留的 333,613 总单元 Maxwell 解的采样差为 3.24% / 7.37% / 9.33%，均在 5% / 10% / 10% 门槛内。净通量守恒最大相对误差 `7.14e-12`，界面势迹抽查误差 0.114%；但全表面原始 P1 单元法向 B 与 BEM 通量差仍 27.61%，不能认证高阶局部界面精度。没有同一 867.9 万网格的 Maxwell 场解或同精度耗时对照。求解、失败历史、原始采样向量和独立界面诊断均保存在现有 `fmm_verification.json`。

## 1. 安装

建议 Python >=3.12，且 `magtense` 原生 wheel 能在本机正常工作。对于你当前的 CUDA 13.x 环境：

```powershell
uv pip install -r requirements.txt
uv pip install 'cupy-cuda13x[ctk]'
```

如果已有完整的 CUDA Toolkit 13.x，可以用 `uv pip install cupy-cuda13x`。CUDA 12.x 改成 `cupy-cuda12x`，**不要同时安装多个 CuPy CUDA 版本**。此方案只是调用 CuPy；没有必要安装 `nvcc` 编译 CUDA C++ 源码。

检查 GPU：

```powershell
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount()); cp.show_config()"
```

## 2. 跑一个没有 MagTense 和 CUDA 的数学自检

```powershell
python gpu_solver.py --self-test --backend numpy
python -m pytest -q tests
```

这个自检使用人工构造的退磁矩阵，**不代表验证了 MagTense 的实际四面体核函数，也不能代表 GPU 加速效果**。

## 3. 使用你的 Maxwell 导出模型

首先使用已提供的原版 AEDT 转换器（在 Windows + AEDT 上）：

```powershell
python bridge.py export --project 'D:\sim\camera.aedt' --design Maxwell3DDesign1 --setup 'Setup1 : LastAdaptive' --out out --line-mm '0,0,3:10,0,3:101'
python bridge.py inspect --out out
```

然后执行 GPU 非线性求解，无需修改 MagTense 的多 B–H Fortran 补丁。多材料迭代在 CuPy 中实现；MagTense 提供独立检查和最终观测场，默认 native 模式还用其构造 K：

```powershell
python gpu_solver.py --out out --backend cupy --dtype float32 --operator volume --method proximal --cache-k
python bridge.py compare --out out --prediction gpu_B.csv --prefix gpu_
```

结果位于：

- `out/gpu_B.csv`：空气观测点 `x,y,z,Bx,By,Bz`，SI 单位。
- `out/gpu_magnetization.npz`：各磁性四面体的最终 `M` (A/m)，以及几何与元素顺序哈希。
- `out/gpu_solve_info.json`：张量生成或缓存加载、非线性迭代耗时，方法、步数、残差历史与是否收敛，以及空气场极限求值记录。
- `out/gpu_comparison.csv` / `out/gpu_metrics.json`：和 Maxwell 对比（若有参考场文件）。
- `out/K_float32.npy` 或 `out/K_volume_r2_f1.5_float64.npy` 等：由算子、积分设置、精度区分的完整稠密张量缓存。

任一方法未收敛都会抛出异常，不输出本次的成功结果。对高磁导材料优先试用 FP64、体积算子与 proximal；Picard 可用于算法诊断，但可能很慢或发散。

原生空气场解析式在个别坐标可能出现可去奇点。求值器只对非有限结果尝试两个方向的对称 Richardson 极限，验证全部偏置点在空气中并核对步长/方向一致性；名义坐标不变，不删测点。无法达到容差则报错。本例 485 点中 1 点使用该处理，极限一致性差约 0.052 nT。

例如：

```powershell
python gpu_solver.py --out out --backend cupy --dtype float64 --operator volume --method proximal --tol 1e-8 --cache-k
```

如果要在 CPU 验证同一条求解逻辑：

```powershell
python gpu_solver.py --out out --backend numpy --dtype float64 --operator volume --method proximal
```

注意：CPU 依然依赖 MagTense 原生模块生成 K，和不带 MagTense 的数学 `--self-test` 不同。

## 4. 距离扫描与热启动

同一磁性几何下，`--cache-k` 根据几何哈希安全复用整块稠密 K；材料曲线和激励变化无需重算 K。不同距离、相对位置变化时，缓存会因为几何哈希不匹配而自动废弃，然后重新构造整个 K（**本版本尚未实现仅更新跨组件矩阵块**）。

若仅发生移动、网格 tile 顺序与材料索引完全一致，可以显式允许复用上一个收敛磁化初值：

```powershell
python gpu_solver.py --out next_distance --backend cupy --operator volume --method proximal `
  --warm-start previous_distance/gpu_magnetization.npz --allow-warm-geometry-change
```

必须确保元素顺序确实对应，程序只核对材料与索引签名，不能验证每个网格单元仍对应相同物理位置。

## 5. 数值实现（含模型限制）

未知量是每个磁性 Tile 的均匀磁化强度 `M_i`，磁静场关系为：

```
H = H_external + K @ M
R(M) = M - material_law(H)
J v = v - d(material_law)/dH @ (K @ v)
```

GPU 通过 cuBLAS 计算 `K @ M`，由 CuPy 计算多种 `B(H)` 插值。可选 proximal 隐式局部更新与 Anderson 混合、矩阵自由 Newton–GMRES + 3x3 block-Jacobi，或阻尼 Picard。大数组在 GPU 常驻，诊断标量同步到 CPU。

* 确切支持：TET4；Maxwell TET10 网格的 4 个角点由旧转换器提取；每个四面体一个磁化矢量；永磁体剩磁 + 各向同性 recoil `mu_r`；线性软磁体；多条不同采样长度的、单值、无磁滞、各向同性 B–H 曲线；全局匀强外场。
* **不支持**：磁滞、磁导率各向异性、非线性永磁体退磁曲线、随时间变化、线圈电流源的几何及激励自动转换、复杂空气中磁力计算、自动适应局部细化。
* 超出 `B–H` 最后一个节点时，以常数饱和 `M` 外推，即 `B=mu0(H+M_sat)`，不是线性外推，也不是 Maxwell 的材料模型默认保证。**用户应检查曲线是否足够长并覆盖实际工作区。**
* 对每个四面体的完整 3×3 影响矩阵全部存 9 分量，即没有冒险把张量压成 6 分量。FP32 显存：`36*N^2` 字节；N=5000 为约 **0.90 GB**，N=10000 为 **3.60 GB**，N=20000 为 **14.40 GB**。加上 CuPy、GMRES、CUDA 库等工作区。CPU 侧按目标点批量 `--batch 32` 生成，不会一次性分配完整 FP64 K。
* 默认由 MagTense Fortran 在 CPU 构造 K；显式 `--k-builder cuda` 在 GPU 计算所有源积分及接收平均。这两条路径都是 O(N²) 稠密算子。无矩阵 `--k-builder fmm` 的近似与限制见上方说明。不要把 K 阶段的加速倍数当作整个求解流程或同精度 Maxwell 对比的加速倍数。
* 由于 MagTense 模型与 Maxwell FEM 的离散不同，残差收敛并不保证外部磁场精度。对复杂薄隔磁片必须做 Maxwell 网格/Tile 收敛测试。

## 6. 排错

- `MagTense ... not installed`：需先安装 `magtense` 对应平台的原生二进制。
- `CUDA backend unavailable`：CUDA 驱动、CUDA wheel 或运行环境不可用；先 `--backend numpy` 检查数学路径。
- `K tensor layout/native H check failed`：退磁算子与 MagTense 原生直接求场结果不符，**停止使用当前输出**。首先检查是否使用了不兼容的 MagTense 版本。
- `Dense K ... GPU memory`：缩减磁性 Tile 数量；无空气体网格不会占据 K 的行列。
- `Newton GMRES failed`：先检查实际残差与原生算子一致性；高磁导率材料可使用 `--dtype float64 --operator volume --method proximal`。更换离散算子后需重新验证场精度。
- `Warm-start geometry changed`：仅当确认同一元素 ID 顺序下发生刚体移动时，使用 `--allow-warm-geometry-change`。

基础参考：MagTense 官方 `python/src/magtense/magstatics.py`；CuPy 官方 `cupyx.scipy.sparse.linalg.gmres` / `LinearOperator` 文档。
