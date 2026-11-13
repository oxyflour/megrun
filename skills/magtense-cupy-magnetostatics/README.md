# MagTense + CuPy nonlinear magnetostatics — Agent Skill

可直接安装到 **Codex / 支持 Agent Skills 的客户端**。包含实际 Python 项目（不是仅有提示词）：Maxwell AEDT 已求解网格导出、任意 TET4/TET10 角点转换、磁性材料过滤、多 B–H 静磁模型、CuPy/NumPy 求解以及与 Maxwell 的 B 场对比。

## 安装

将整个 `magtense-cupy-magnetostatics` 文件夹放入以下任意位置：

- **Codex 用户级**：`~/.agents/skills/magtense-cupy-magnetostatics/`（Windows：`$HOME\.agents\skills\magtense-cupy-magnetostatics`）
- **Codex 项目级**：`<repo>/.agents/skills/magtense-cupy-magnetostatics/`

ZIP 已带最外层同名文件夹。Windows 可直接运行：

```powershell
New-Item -ItemType Directory -Force "$HOME\.agents\skills" | Out-Null
Expand-Archive -Path .\magtense-cupy-magnetostatics-skill.zip -DestinationPath "$HOME\.agents\skills" -Force
```

在 Codex 中使用 `$magtense-cupy-magnetostatics` 调用，也可让其自动根据 `SKILL.md` 描述触发。Skill 本身**不负责自动安装 AEDT、CUDA 或 MagTense**。

## 快速上手

进入该 Skill 的代码目录（`scripts/aedt_magtense_bridge`）：

```powershell
python -m pip install -r requirements.txt
python -m pip install "cupy-cuda13x[ctk]" # CUDA 13.x only; match your CUDA environment
python gpu_solver.py --self-test --backend numpy
python -m pytest -q tests
```

如果已经有 `bridge.py export` 生成的 `out`，使用：

```powershell
python gpu_solver.py --out out --backend cupy --dtype float64 --operator volume --method proximal --tol 1e-8 --cache-k
python bridge.py compare --out out --prediction gpu_B.csv --prefix gpu_
```

真实 Maxwell 模型从头导出的命令和使用限制参见 `SKILL.md` 或 `scripts/aedt_magtense_bridge/README_GPU.md`。

## 状态与限制

- **当前检查**：129 项通过、1 项因缺少 gfortran 跳过，包含 CUDA、FMM、自适应近场分块压缩、批量空气极限、FEM–BEM 积分/算子/AMG，以及原生磁盘包导入、完整性校验和离线求解检查；已有四组真实模型的 FP64 稠密 K 与 Maxwell 场对比。真实材料 FP32 精度尚未做这组完整验证，历史稠密路径的结果不能当作 FMM 精度结论。
- **实测基准**：Maxwell 2022 R2、RTX 3090 Ti，3 mm 双模组、VACOFLUX 50 厂家数据表、零线圈电流。3629 个磁性四面体，485 个空气测点；相对细 Maxwell 的 B 向量 L2 差约 3.66%，RMS 1.48 mT，最大 7.49 mT。材料表插值和网格均有近似，不能当作实物精度或普遍误差保证。
- **GPU K 构建**：新增 `--operator volume --k-builder cuda`，在 GPU 上计算解析四面体源积分及相同的四点/256 点接收平均；兼容默认仍为 CPU 原生核。精度与阶段耗时见 `references/validation-gpu-k-20261107.json`。历史 CPU 路径构 K 约 13 分钟、GPU 迭代约 4 秒，不代表新路径的端到端时间。
- **无矩阵 FMM**：新增 `--operator volume --k-builder fmm`，可选 CPU FMM3D 或 `--fmm-farfield cuda` 的原生 Windows GPU 插值 FMM，无完整 K。GPU 路径复用树，不依赖 WSL/JAX/FMM3D；CPU 模式需额外安装 `fmm3dpy>=2.1,<3`。源积分误差、内存预算、抽样原生残差与使用命令见 [README_GPU.md](scripts/aedt_magtense_bridge/README_GPU.md#无矩阵-fmm-模式)。尚未完成百万单元容量或性能验收。
- **精度边界**：非线性材料为无磁滞、各向同性、单值 B–H；TET10 简化为 TET4；没有线圈源几何转换或受力积分。稠密 CUDA 构 K 后仍保留原生检查和 CPU 空气测点计算；FMM 的收敛残差针对近似算子，另输出原生抽样指标。
- **内存边界**：native/cuda 路径为完整 `3N×3N` K 张量；FP32 约 `36N²` 字节，FP64 约 `72N²` 字节。FMM 依赖源积分点、树、近场索引及迭代工作区，近场块可按预算分批计算。
- **FEM–BEM 替代路线**：P1 势 / P0 边界通量，默认稠密 CPU 参考，新增 CUDA H 边界与 CPU FEM/AMG。实际案例的完整 271.9 万磁性单元已在 4 次 Newton 更新后达到 `5.84e-10`，空气/材料 B/边界法向 B 采样误差 3.24% / 7.37% / 9.33%，通过初始门槛；全边界原始 P1 单元法向通量差仍 27.6%，局部高阶精度和 Maxwell 加速比未认证。[实现、命令、资源与验收范围](references/architecture.md#cuda-hierarchical-boundary-and-sparse-amg)。小模型及此前失败记录保留，不能与本次完整网格结果混用。
- **原生磁盘转移**：[Ansys 18.2 目标流程](scripts/aedt_magtense_bridge/README.md#ansys-182-原生导出与无-maxwell-服务器) 使用内置 IronPython，无需 PyAEDT；服务器使用 `import-native` / `solve-fem-bem`，无需 Maxwell 或 MagTense 原生库。实际 2022 R2 原生导出、只生成网格的导出和无 Ansys 依赖进程求解已验证，18.2 源机实测待完成。

MagTense 本体来自 https://github.com/cmt-dtu-energy/MagTense ，依赖其平台原生库且适用其许可证；该 ZIP 没有捆绑 MagTense 原始仓库或编译后的二进制。此包包含的可选原生多 B–H 补丁会修改外部源码，应仅在授权下执行。
