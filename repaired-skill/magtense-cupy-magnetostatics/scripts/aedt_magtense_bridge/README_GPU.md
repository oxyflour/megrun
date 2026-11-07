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
* 默认由 MagTense Fortran 在 CPU 构造 K；显式 `--k-builder cuda` 在 GPU 计算所有源积分及接收平均。两者都是 O(N²) 稠密算子，未实现 FMM。最终空气测点场和原生随机场校验仍调用 CPU MagTense。不要把 K 阶段的加速倍数当作整个求解流程或同精度 Maxwell 对比的加速倍数。
* 由于 MagTense 模型与 Maxwell FEM 的离散不同，残差收敛并不保证外部磁场精度。对复杂薄隔磁片必须做 Maxwell 网格/Tile 收敛测试。

## 6. 排错

- `MagTense ... not installed`：需先安装 `magtense` 对应平台的原生二进制。
- `CUDA backend unavailable`：CUDA 驱动、CUDA wheel 或运行环境不可用；先 `--backend numpy` 检查数学路径。
- `K tensor layout/native H check failed`：退磁算子与 MagTense 原生直接求场结果不符，**停止使用当前输出**。首先检查是否使用了不兼容的 MagTense 版本。
- `Dense K ... GPU memory`：缩减磁性 Tile 数量；无空气体网格不会占据 K 的行列。
- `Newton GMRES failed`：先检查实际残差与原生算子一致性；高磁导率材料可使用 `--dtype float64 --operator volume --method proximal`。更换离散算子后需重新验证场精度。
- `Warm-start geometry changed`：仅当确认同一元素 ID 顺序下发生刚体移动时，使用 `--allow-warm-geometry-change`。

基础参考：MagTense 官方 `python/src/magtense/magstatics.py`；CuPy 官方 `cupyx.scipy.sparse.linalg.gmres` / `LinearOperator` 文档。
