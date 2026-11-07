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

- **本次修复版检查**：44 项通过、1 项因缺少 gfortran 跳过；另通过 NumPy/CuPy、FP64/FP32 的 CLI 解析线性解检查。
- **实测基准**：Maxwell 2022 R2、RTX 3090 Ti，3 mm 双模组、VACOFLUX 50 厂家数据表、零线圈电流。3629 个磁性四面体，485 个空气测点；相对细 Maxwell 的 B 向量 L2 差约 3.66%，RMS 1.48 mT，最大 7.49 mT。材料表插值和网格均有近似，不能当作实物精度或普遍误差保证。
- **耗时范围**：本例 CPU K 构建约 13 分钟，GPU 隐式迭代约 4 秒；首次端到端速度优势尚未证明。
- **精度边界**：非线性材料为无磁滞、各向同性、单值 B–H；TET10 简化为 TET4；没有 GPU 张量生成、FMM、线圈源几何转换或受力积分。
- **内存边界**：完整 `3N×3N` K 张量；FP32 约 `36N²` 字节，FP64 约 `72N²` 字节。

MagTense 本体来自 https://github.com/cmt-dtu-energy/MagTense ，依赖其平台原生库且适用其许可证；该 ZIP 没有捆绑 MagTense 原始仓库或编译后的二进制。此包包含的可选原生多 B–H 补丁会修改外部源码，应仅在授权下执行。
