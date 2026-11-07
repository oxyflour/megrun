# MagTense + CuPy 非线性 GPU 静磁求解器（原型）

本目录包含一个真实可执行的 **MagTense CPU 解析四面体退磁张量 + CuPy GPU 非线性迭代** 原型。它直接使用之前 `bridge.py export` 生成的 `model.json`、`mesh/*.npz` 和 `points.csv`；无须重新在 Maxwell 划网格或修改数据格式。

**已经离线验证**：CPU NumPy 求解数学测试、双 B–H 曲线、向量化 Newton 切线、完整 9 分量张量排列、缓存、收敛报错、比较 CSV 接口。**尚未验证**：在当前环境内实际加载 MagTense Fortran 二进制、在真实 NVIDIA GPU 上运行 CuPy、与真实 AEDT 结果的端到端数值误差。代码内提供 `K` 与原生 MagTense `H` 的运行时自检，执行真实模型时会自动检查。

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

然后执行 GPU 非线性求解（无需修改 MagTense 的多 B–H Fortran 补丁，**因为 MagTense 在这里仅负责解析 K 和最终观测场，多材料迭代完全在 CuPy 中实现**）：

```powershell
python gpu_solver.py --out out --backend cupy --dtype float32 --method newton --cache-k
python bridge.py compare --out out --prediction gpu_B.csv --prefix gpu_
```

结果位于：

- `out/gpu_B.csv`：空气观测点 `x,y,z,Bx,By,Bz`，SI 单位。
- `out/gpu_magnetization.npz`：各磁性四面体的最终 `M` (A/m)，以及几何与元素顺序哈希。
- `out/gpu_solve_info.json`：张量生成、非线性迭代耗时，Newton 步数、残差历史与是否收敛。
- `out/gpu_comparison.csv` / `out/gpu_metrics.json`：和 Maxwell 对比（若有参考场文件）。
- `out/K_float32.npy`：指定 `--cache-k` 时写入的完整稠密张量，磁性几何不变时可再次使用。

Newton-GMRES 未收敛会抛出异常并**且不写出“成功”结果**。你可以用 FP64 检查精度问题，或用 `--method picard --damping 0.02` 做算法诊断（高磁导率材料 Picard 很可能很慢或发散）。

例如：

```powershell
python gpu_solver.py --out out --backend cupy --dtype float64 --method newton --tol 1e-7 --gmres-rtol 1e-5 --cache-k
```

如果要在 CPU 验证同一条求解逻辑：

```powershell
python gpu_solver.py --out out --backend numpy --dtype float64 --method newton
```

注意：CPU 依然依赖 MagTense 原生模块生成 K，和不带 MagTense 的数学 `--self-test` 不同。

## 4. 距离扫描与热启动

同一磁性几何下，`--cache-k` 根据几何哈希安全复用整块稠密 K；材料曲线和激励变化无需重算 K。不同距离、相对位置变化时，缓存会因为几何哈希不匹配而自动废弃，然后重新构造整个 K（**本版本尚未实现仅更新跨组件矩阵块**）。

若仅发生移动、网格 tile 顺序与材料索引完全一致，可以显式允许复用上一个收敛磁化初值：

```powershell
python gpu_solver.py --out next_distance --backend cupy --method newton `
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

GPU 通过 cuBLAS 计算 `K @ M`，由 CuPy 计算多种 `B(H)` 插值及切线，使用矩阵自由 GMRES + 3x3 block-Jacobi + Newton line search。也提供阻尼 Picard。**大数组在 GPU 常驻，只有标量诊断同步到 CPU。**

* 确切支持：TET4；Maxwell TET10 网格的 4 个角点由旧转换器提取；每个四面体一个磁化矢量；永磁体剩磁 + 各向同性 recoil `mu_r`；线性软磁体；多条不同采样长度的、单值、无磁滞、各向同性 B–H 曲线；全局匀强外场。
* **不支持**：磁滞、磁导率各向异性、非线性永磁体退磁曲线、随时间变化、线圈电流源的几何及激励自动转换、复杂空气中磁力计算、自动适应局部细化。
* 超出 `B–H` 最后一个节点时，以常数饱和 `M` 外推，即 `B=mu0(H+M_sat)`，不是线性外推，也不是 Maxwell 的材料模型默认保证。**用户应检查曲线是否足够长并覆盖实际工作区。**
* 对每个四面体的完整 3×3 影响矩阵全部存 9 分量，即没有冒险把张量压成 6 分量。FP32 显存：`36*N^2` 字节；N=5000 为约 **0.90 GB**，N=10000 为 **3.60 GB**，N=20000 为 **14.40 GB**。加上 CuPy、GMRES、CUDA 库等工作区。CPU 侧按目标点批量 `--batch 32` 生成，不会一次性分配完整 FP64 K。
* 在原始 MagTense Fortran 代码上构造 K，CPU 构建复杂度仍为 O(N²)；**GPU 只加速非线性迭代，不能声称整个仿真有 GPU 倍数提速**。下一步若要扫大量隔磁片形状，应 GPU 化解析四面体-目标点张量或研究近远场混合算子。
* 由于 MagTense 模型与 Maxwell FEM 的离散不同，残差收敛并不保证外部磁场精度。对复杂薄隔磁片必须做 Maxwell 网格/Tile 收敛测试。

## 6. 排错

- `MagTense ... not installed`：需先安装 `magtense` 对应平台的原生二进制。
- `CUDA backend unavailable`：CUDA 驱动、CUDA wheel 或运行环境不可用；先 `--backend numpy` 检查数学路径。
- `K tensor layout/native H check failed`：退磁算子与 MagTense 原生直接求场结果不符，**停止使用当前输出**。首先检查是否使用了不兼容的 MagTense 版本。
- `Dense K ... GPU memory`：缩减磁性 Tile 数量；无空气体网格不会占据 K 的行列。
- `Newton GMRES failed`：可以尝试 `--dtype float64`、修改 `--gmres-rtol` 或使用阻尼 Picard 诊断。高磁导率材料可能需要更强的非线性/近场预条件器。
- `Warm-start geometry changed`：仅当确认同一元素 ID 顺序下发生刚体移动时，使用 `--allow-warm-geometry-change`。

基础参考：MagTense 官方 `python/src/magtense/magstatics.py`；CuPy 官方 `cupyx.scipy.sparse.linalg.gmres` / `LinearOperator` 文档。
