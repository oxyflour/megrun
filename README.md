# MagTense / CuPy 磁静态求解与 Maxwell 验证

本仓库保存 Agent Skill、数值求解代码，以及 2026-11-07 的 Maxwell 对比记录。

## 代码入口

- Skill 源码：[skills/magtense-cupy-magnetostatics](skills/magtense-cupy-magnetostatics/README.md)。安装和调用方式见该目录的 [SKILL.md](skills/magtense-cupy-magnetostatics/SKILL.md)。
- Python 工具及测试：`skills/magtense-cupy-magnetostatics/scripts/aedt_magtense_bridge/`。
- 早期 v2 合成模型的输入、场数据、磁化结果、测试记录和环境信息统一归档于 [validation/20261107_repair_staged/v2_synthetic](validation/20261107_repair_staged/v2_synthetic/TEST_REPORT.txt)。旧源码副本及两个旧检查入口已移除，原始版本可从 Git 提交 `bdc71d9` 追溯；当前开发与测试统一使用 `skills/magtense-cupy-magnetostatics/`。
- `validation/` 保存各阶段的模型准备脚本、材料曲线及来源、精简网格、场数据、收敛记录和报告。
- 与 Maxwell 数值精度对齐的计划提纲及分项方案见 [accuracy-study.md](docs/plan/accuracy-study.md)。

求解器提供四面体接收体积平均算子、非线性隐式本构迭代，以及空气测点的稳定场计算。支持 NumPy/CuPy；依赖平台原生 MagTense 库。native/cuda 的完整 K 矩阵仍需二次方存储，FP64 约为 `72 N²` 字节。新增可选 `--operator volume --k-builder fmm`，不生成完整 K；远场可选 CPU FMM3D 或 `--fmm-farfield cuda` 的原生 Windows GPU 插值 FMM，GPU 路径复用树并直接计算分批近场 H，无须 WSL。安装、内存预算和误差边界见 [FMM 使用说明](skills/magtense-cupy-magnetostatics/scripts/aedt_magtense_bridge/README_GPU.md#无矩阵-fmm-模式)。尚未完成百万单元容量或性能验收。

新增 `--operator volume --k-builder cuda --backend cupy`，在 GPU 上计算完整 K 的解析源积分与近邻归约，保持原四点/256 点积分和材料方程。CPU 原生构建仍为兼容默认。见 [GPU K 耗时与精度报告](validation/20261107_gpu_k/FINAL_REPORT.txt) 及 [阶段数据](validation/20261107_gpu_k/summary.json)。

## 已有验证结果

验证条件为 3 mm 间隔的双 camera 电磁模型、零线圈电流、公开材料 B–H 曲线。结果是与 Maxwell 的数值对比，不是实物精度认证。

| 验证范围 | 空气测点数 | B 向量相对 L2 差 |
| --- | ---: | ---: |
| 无隔磁片，VACOFLUX 50 轭铁 | 485 | 3.6551% |
| MUMETALL 隔磁片 0.10 × 8.8 × 3.4 mm | 566 | 3.7492% |
| MUMETALL 隔磁片 0.20 × 8.8 × 3.4 mm | 566 | 4.1559% |
| MUMETALL 隔磁片 0.20 × 10 × 5 mm | 566 | 3.8782% |

两组测点集不同，不能直接把上述百分比相减来评估隔磁效果。0.10 mm 隔磁片另有固定 Maxwell 网格的增量场诊断，ΔB 相对 L2 差为 9.0781%；该结果不适用于另两种尺寸。

三组主尺寸的 Maxwell 自适应求解及 CuPy `1e-8` 本构残差检查均通过。中间尺寸的加密模型在 24000 次迭代后残差为 `2.1528e-8`，未通过验收，没有认证的加密场结果。MUMETALL 曲线来自公开图表读取，工作点处于已核查的低场段；高场段仍存在读取不确定性。

详见：

- [无隔磁片完整报告](validation/20261107_repair_staged/FINAL_REPORT.txt)
- [隔磁片完整报告](validation/20261107_shield_sizes/FINAL_REPORT.txt)
- [隔磁片结果摘要](validation/20261107_shield_sizes/final_summary.json)
- [材料曲线质量审计](validation/20261107_shield_sizes/curve_quality_audit.json)

## 运行检查

进入 Skill 的 Python 工具目录，在具有相应依赖的环境中执行：

```powershell
python -m pip install -r requirements.txt
python gpu_solver.py --self-test --backend numpy
python -m pytest -q tests
```

当前源码检查为 129 项通过、1 项因缺少 gfortran 跳过，包含 CUDA、FMM、自适应近场分块压缩、批量空气极限、FEM–BEM 积分/算子/AMG，以及原生磁盘包导入、完整性校验和离线求解检查。已有四组真实模型 FP64 稠密 K 与 Maxwell B 场对比的历史结论保持原范围；FMM 的检查与限制见上方使用说明。Maxwell 导出需已安装并授权的 AEDT；CUDA/CuPy 依赖需匹配本机环境。

当前双模组模型已生成本机 [加密 AEDT 工程](validation/20261109_fmm_refined/gap3mm_fmm_3m.aedt)，实际为 **2,718,864 个磁性单元、8,679,245 个总单元**，并通过重新打开后的网格数量核对。复现入口为 [prepare_mesh.py](validation/20261109_fmm_refined/prepare_mesh.py)，计数和原工程不变性检查见 [mesh_generation.json](validation/20261109_fmm_refined/mesh_generation.json)。工程与同名 `.aedtresults` 网格目录保留本机；没有提交该大模型的 Maxwell 场求解。FMM 的 3629 单元流程验证见 [fmm_verification.json](validation/20261109_fmm_refined/fmm_verification.json)。大模型实际运行入口为 [run_fmm.py](validation/20261109_fmm_refined/run_fmm.py)，导出/求解状态、采样内存峰值及残差写入 [large_fmm_run.json](validation/20261109_fmm_refined/large_fmm_run.json)，原始资源采样为 `large_fmm_resources.csv`。失败阶段使用非零退出码，不能把完成网格导出当作求解通过。

2026-11-10 的原生 Windows GPU 大模型实跑已因耗时和收敛速度不实用而停止：约 7 小时 53 分钟内完成 8 次全量场计算、4 次材料更新，FMM 本构残差由 `2.952195` 降至 `2.459533`，未达到 `1e-8`。最近一次场计算约 58.8 分钟，解析近场修正占约 91%。**当前实现未通过大规模非线性求解的实用性验收，不能作为已验证的 Maxwell 替代方案。** 同网格大模型的 Maxwell 端到端计时尚未进行，因此不报告严格加速倍率。停止原因、原始退出码和性能评估保存在上述运行记录中。

替代路线的 [CUDA 分层 BEM＋稀疏 FEM/AMG 原型](skills/magtense-cupy-magnetostatics/references/architecture.md#cuda-hierarchical-boundary-and-sparse-amg) 已在该案例全部 2,718,864 个磁性单元上完成耦合求解：4 次 Newton 更新，弱残差 `5.84e-10`；缓存流程 17 分 22 秒。另一次完整 H 首次构建由 41 分 49 秒降至 27 分 20 秒，新旧数值缓存二进制相同；这些独立阶段实测不等于一次完整冷启动求解。空气、材料 B、边界法向 B 对保留的 333,613 总单元 Maxwell 解的采样误差为 3.24% / 7.37% / 9.33%，通过声明的 5% / 10% / 10% 门槛。但全边界原始 P1 单元法向通量差仍为 27.6%，**尚未认证 Maxwell 高阶局部精度或同精度加速比**。命令、精度配置、资源和限制见链接；记录在同一 `fmm_verification.json` 的 `fem_bem_large_runs`，失败历史保留。

新增 [Ansys 18.2 原生导出与无 Maxwell 服务器流程](skills/magtense-cupy-magnetostatics/scripts/aedt_magtense_bridge/README.md#ansys-182-原生导出与无-maxwell-服务器)：源机用内置 IronPython 导出网格、材料、变量、激励/边界和原始设置，服务器校验磁盘包后运行 FEM–BEM。无需 PyAEDT；已在 2022 R2 上实测原生导出及无 Ansys 依赖进程求解，18.2 源机实测待完成。

## 仓库与本地产物

Git 保留代码、小型 NPZ 网格及磁化结果、材料表、场数据和报告。虚拟环境、约 10 GB 的 K 矩阵缓存、原生 AEDT 工程及结果数据库、原始网格转储、下载的厂家 PDF 和 ZIP 分发包保留在本地，由 `.gitignore` 排除。

历史验证脚本仍有 AEDT 工程等本机绝对路径，迁移运行时需要调整；原来引用旧源码副本的脚本已改为按仓库位置定位当前工具。复现 v2 Newton 失败的诊断和回归测试使用该案例已有的 `gpu_solver_v2_original.py`，其 SHA256 与被清理的原求解器一致。历史记录中的命令和环境路径保留运行当时的值，不代表当前执行入口；当前检查使用上面的命令，并将新求解结果写入独立输出目录，避免覆盖历史证据。`skill-test/` 下已忽略的本机虚拟环境仍供当前测试使用。部分报告和清单引用上述本地产物；仅克隆仓库不能重新进行原生 Maxwell 求解。外部材料来源和产物哈希保存在相应 provenance / manifest JSON 中。
