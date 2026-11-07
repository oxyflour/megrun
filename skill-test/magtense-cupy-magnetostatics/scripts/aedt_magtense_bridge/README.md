# Maxwell AEDT → MagTense：任意四面体静磁场精度对比

**目标**：不重新画长方体，直接读取**已完成求解的 Maxwell 3D Magnetostatic `.aedt` 项目**，将各磁性零件的体网格转换为 MagTense type-5 tetrahedral Tile，自动剔除空气/非磁性体，在同一批**空气中**的采样点对比 Maxwell 与 MagTense 的 `B = (Bx,By,Bz)`，输出误差指标。非线性软磁 B–H 曲线可以导出并传入 MagTense 底层 Fortran 求解函数。

**状态**：已增加多种不同非线性 B–H 曲线的原生接口补丁、材料索引和测试；详见 [README_MULTI_BH.md](README_MULTI_BH.md)。当前容器**没有 Ansys AEDT 许可证及 MagTense 原生计算库**，所以未对你的真实 `.aedt` 文件进行端到端数值运行，也未声称测试了两种求解器的实际精度。需要在你安装了 AEDT 和 MagTense 的环境执行真实对比。

## 1. 安装

**多种 B–H 材料先安装补丁版 MagTense**：Windows 可在配置好 ifx、MKL 和 make 后直接运行 `powershell -ExecutionPolicy Bypass -File .\scripts\install_multibh.ps1`；Linux/WSL2 可用 `bash scripts/install_multibh.sh`。完整说明请看 [README_MULTI_BH.md](README_MULTI_BH.md)。

推荐 Windows 11 + Python **3.12 或更新**。确保已安装并完成 Maxwell 求解（不仅是 `.aedt` 文件，工程对应的 `.aedtresults` 也要可用）。

```powershell
cd aedt_magtense_bridge
python -m pip install -r requirements.txt
python -m pytest -q
```

`export` 必须在带有 AEDT 的 Windows 环境执行；`solve` 和 `compare` 可以复制 `out/` 到另一台安装 MagTense 的 Windows/Linux 机器执行。

## 2. 一键从真实 Maxwell 工程导出

先在 Maxwell 中确保是 **3D Magnetostatic** 且已有 `Setup1 : LastAdaptive` 的结果。请把下面的设计名、工程路径、采样点改成真实值。示例沿一条位于空气中的线采样 101 个点：

```powershell
python bridge.py export `
  --project "D:\sim\camera.aedt" `
  --design "Maxwell3DDesign1" `
  --setup "Setup1 : LastAdaptive" `
  --out out `
  --line-mm "-4,0,3:14,0,3:101" `
  --max-tiles 10000
```

**默认遍历模型中的全部 3D solid：**

- 自动跳过 `Air`、`Vacuum`；对于无电流、相对磁导率约等于 1、无矫顽力的非磁性体也跳过。这些物体的物理空间**不会被挤掉**；所有磁体顶点位置仍保持 AEDT 全局坐标。
- 自动按**每一个零件**调用 PyAEDT 的 `create_fieldplot_volume(..., "Mesh")` 和 `export_field_plot` 导出 `.aedtplt`，这样能够准确保留“单元属于哪个实体/材料”的关联。
- `TET4` 使用 4 个顶点；`TET10` 取 AEDT **正确的角点序号 0,2,5,9**，转换为 4 角点的 MagTense Tile。避免把中间节点误当成角点；自动统一四面体朝向，检查零体积单元。
- 对曲面二次四面体会**线性化**：舍弃中间边节点，所以边界会有离散几何误差。`model.json` 的 `mesh_stats` 会报告该状态。
- 用物体的 CAD 包围盒检查 `.aedtplt` 节点究竟采用工程单位还是米；如果可验证，会自动纠正 mm/m 混淆。若不能验证则发出警告。
- 识别永磁体 Hc / recoil permeability、线性软磁、**多种独立非线性软磁 B–H 曲线**（需要先编译仓库内的 MagTense Fortran 补丁），并**显式阻止**不支持的激励/材料。不会把非线性 B–H 偷换成常数磁导率。
- 额外用 PyAEDT 从已有结果在**同样的采样点**导出 `B_Vector` 作为 Maxwell 参考数据，存为 `out/maxwell_B.fld`。

如果只想验证部分磁性零件，可以写：

```powershell
python bridge.py export --project "D:\sim\camera.aedt" --design "Maxwell3DDesign1" --out out --line-mm "-4,0,3:14,0,3:101" --objects Magnet_A Magnet_B Shield
```

如果是**任意分布的空间采样点**，准备 `points.csv`（单位 **米**，有无表头均可）：

```csv
x_m,y_m,z_m
0.001,0.000,0.004
0.002,0.000,0.004
0.004,0.000,0.004
```

然后使用 `--points-csv points.csv` 替换 `--line-mm`。**采样点必须在空气/真空内**，脚本会检测落在磁性四面体内的点并拒绝比较；不能在磁体内部简单使用 `B = μ0H`。

## 3. 估算规模、求解、比较

```powershell
python bridge.py inspect --out out
python bridge.py solve --out out --max-tiles 10000 --max-error 1e-5 --max-it 500
python bridge.py compare --out out
```

**`inspect` 只做统计，不会运行求解器。** 推荐先用磁性网格 **1,000～5,000** 个四面体调试，再增至 10,000。输出报告以 6 个 float32 张量分量、`24N²` 字节计算粗略显存/内存量级，不是 MagTense 真实峰值或性能承诺。稠密相互作用的时间复杂度为 `O(N²)`，含非线性迭代时可能更慢。

`solve` 输出：

- `out/magtense_B.csv`：相同空气采样点坐标和 MagTense `B` 向量，单位 **T**。
- `out/magnetization.npz`：每个磁性四面体迭代结束后的 `M`，单位 **A/m**。
- `out/solve_info.json`：磁性 Tile 总数、误差终止参数和磁化迭代指标。

`compare` 输出：

- `out/comparison.csv`：逐点 `(Bx,By,Bz)_Maxwell`、`(Bx,By,Bz)_MagTense` 和差值。
- `out/metrics.json`：RMS 向量误差（mT）、最大误差（mT）、整体相对误差百分比等。

其中整体相对误差定义为：

\[
E_{rel}=100\%\cdot\frac{\sqrt{\sum_i \|\mathbf{B}^{MT}_i-\mathbf{B}^{MX}_i\|^2}}{\sqrt{\sum_i\|\mathbf{B}^{MX}_i\|^2}}
\]

如果你的 AEDT 版本输出 `.fld` 格式与当前解析器不同，**不会悄悄错位比较**；可以将 Maxwell 参考转成六列的 `x_m,y_m,z_m,Bx_T,By_T,Bz_T` CSV，再运行：

```powershell
python bridge.py compare --out out --maxwell-csv maxwell_reference.csv
```

## 4. 多材料和非线性限制

脚本默认从 AEDT 读取对象材料参数。

| Maxwell 材料 | 处理 | 说明 |
|---|---|---|
| 永磁体 Hc + 线性 µr | 自动转换为 `Br=-µ0 µr Hc` 矢量，MagTense type 1 | 支持均匀磁化方向与线性 recoil；局部坐标系磁化需要覆盖配置 |
| 线性软磁 µr | MagTense type 3 | 支持各向同性常数 µr |
| **一条或多条 B–H 曲线** | MagTense type 2，为每条独立曲线分配 `stateFunctionIndex`，提供 `M(H)=B(H)/µ0-H` | 支持**无磁滞、各向同性、单值**的软磁曲线，**两条以上需编译 `patch_magtense.py`** |
| 空气、真空、非磁性物体 | 自动省略网格 | **不省略空间距离** |
| 两种以上**不同** B–H 曲线 | **支持（补丁版）** | 独立曲线不做重采样；使用相应每个四面体的 1-based `stateFunctionIndex`；未经补丁的官方 wheel 会明确报错 |
| 电流线圈、时变激励 | 默认报错 | 第一版没有从 AEDT 导出电流分布，**不能**忽略激励后声称模型一致 |
| 非线性带磁滞永磁材料、各向异性 B–H | 报错 | 必须扩展物理模型 |
| CAD 曲面、高阶四面体 | 角点线性化 | TET10 与 MagTense 的 TET4 **不完全等价** |

如果 PyAEDT 无法解析材料表达式（例如一个材料参数引用工程变量），可用 `overrides.example.json` 的格式覆盖指定材料或对象：

```powershell
python bridge.py export --project "D:\sim\camera.aedt" --design "Maxwell3DDesign1" --out out --line-mm "-4,0,3:14,0,3:101" --overrides overrides.json
```

`overrides.json` 支持 `"objects"`（按 CAD 对象名，优先级最高）和 `"materials"`（按材料名）。其中 `direction` 必须是**全局坐标系**下的永磁体剩磁方向；`br_T` 是 T，`bh_H_Apm_B_T` 每行是 `[H(A/m), B(T)]`。例如：

```json
{
  "objects": {
    "Magnet_A": {"kind":"permanent", "mu_r":1.05, "br_T":1.20, "direction":[0,0,1]}
  },
  "materials": {
    "Fe_Example": {"kind":"nonlinear", "bh_H_Apm_B_T":[[0,0],[100,0.25],[1000,1.2],[10000,1.7]]}
  }
}
```

注意 `Fe_Example` 中的 B–H 仅是示意数据，不是准确材料曲线。必须换成你的 Maxwell 材料数据。建议保证 B–H 数据覆盖非线性求解中的实际磁场范围。

### 重要：必须先构建补丁版 MagTense

原生 Fortran `loadStateFunction` 在官方版本使用共享二维表，因此**必须修改 Fortran 并重新编译**。此仓库已包含 `patch_magtense.py`，可自动修改 MagTense 原始仓库中 `FortranToPythonIO.f90`、`IterateMagnetSolution.f90` 和 `magstatics.py`。详见 **[README_MULTI_BH.md](README_MULTI_BH.md)**，其中包含 Windows/Linux 构建命令和 `native_smoke.py` 验证脚本。

修复后的桥接器不将多条曲线合成一条：每种不同曲线对应一个 Fortran `stateFcn(i)`；同样的曲线复用索引；不同长度的曲线保留原始数据点且通过表头记录有效点数。

## 5. 没有 AEDT 的离线演示

```powershell
python bridge.py demo --out demo_out
python bridge.py inspect --out demo_out
python bridge.py solve --out demo_out --dry-run
# 如果已安装 MagTense，可去掉 --dry-run，解 6 个四面体拼成的永磁立方体
python bridge.py solve --out demo_out
```

合成立方体为 `1×1×1 mm`，磁化沿 X 轴、`Br=1.2 T`；有 6 个 TET4。**它没有 Maxwell 的参考结果，不能据此宣称精度。**

## 6. 建议的真实精度实验

先拿 Maxwell 的**两个永磁体**（软磁材料暂时关闭），将磁性四面体数量控制在 100～1000，观察空气中 50～200 个测点的场误差。如果成功，再加入一个**非线性隔磁片**，测试 1k、2k、5k 个磁性四面体的精度收敛和耗时。

两边边界条件应尽可能一致（MagTense 为开放无限空间场；Maxwell 空气盒和外边界可能带来误差），尤其不要在 Maxwell 用一个非常靠近磁体的截断磁场边界。磁力需另外对比，本工具当前**仅对比磁场 B**。

## 参考源码和 API

- [MagTense 官方仓库](https://github.com/cmt-dtu-energy/MagTense)
- [MagTense Python Tile 和 run_simulation 源码](https://github.com/cmt-dtu-energy/MagTense/blob/master/python/src/magtense/magstatics.py)
- [MagTense Fortran 非线性状态函数](https://github.com/cmt-dtu-energy/MagTense/blob/master/source/DemagField/IterateMagnetSolution.f90)
- [PyAEDT create_fieldplot_volume / export_field_plot / export_field_file](https://github.com/ansys/pyaedt/blob/main/src/ansys/aedt/core/visualization/post/post_common_3d.py)
- [PyAEDT .aedtplt 网格连接关系与 TET10 角点](https://github.com/ansys/pyaedt/blob/main/src/ansys/aedt/core/visualization/plot/pyvista.py)

## GPU nonlinear solver (new)

See **[README_GPU.md](README_GPU.md)**. The new `gpu_solver.py` uses MagTense to
construct the TET demagnetization operator, CuPy for multiple nonlinear B-H
curves and Newton-GMRES, and can compare against the Maxwell reference using:

```bash
python gpu_solver.py --out out --backend cupy --method newton --cache-k
python bridge.py compare --out out --prediction gpu_B.csv --prefix gpu_
```

This CuPy solver **does not require the native multi-BH patch**; nonlinear
material laws are calculated on the GPU, not in the old MagTense Fortran loop.
The original CPU multi-BH patch remains available for CPU baseline comparisons.
