# MagTense / CuPy 磁静态求解与 Maxwell 验证

本仓库保存修复后的 Agent Skill、数值求解代码，以及 2026-11-07 的 Maxwell 对比记录。

## 代码入口

- 当前修复版：[repaired-skill/magtense-cupy-magnetostatics](repaired-skill/magtense-cupy-magnetostatics/README.md)。安装和调用方式见该目录的 [SKILL.md](repaired-skill/magtense-cupy-magnetostatics/SKILL.md)。
- Python 工具及测试：`repaired-skill/magtense-cupy-magnetostatics/scripts/aedt_magtense_bridge/`。
- `skill-test/` 保存早期提取版本、检查脚本和测试记录；继续开发请使用 `repaired-skill/`。
- `validation/` 保存各阶段的模型准备脚本、材料曲线及来源、精简网格、场数据、收敛记录和报告。

修复版提供四面体接收体积平均算子、非线性隐式本构迭代，以及空气测点的稳定场计算。支持 NumPy/CuPy；依赖平台原生 MagTense 库。完整 K 矩阵仍需二次方存储，FP64 约为 `72 N²` 字节。

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

进入修复版 Python 工具目录，在具有相应依赖的环境中执行：

```powershell
python -m pip install -r requirements.txt
python gpu_solver.py --self-test --backend numpy
python -m pytest -q tests
```

已有修复版检查记录为 44 项通过、1 项因缺少 gfortran 跳过；另有 NumPy/CuPy FP64/FP32 的 CLI 检查。Maxwell 导出需已安装并授权的 AEDT；CUDA/CuPy 依赖需匹配本机环境。

## 仓库与本地产物

Git 保留代码、小型 NPZ 网格及磁化结果、材料表、场数据和报告。虚拟环境、约 10 GB 的 K 矩阵缓存、原生 AEDT 工程及结果数据库、原始网格转储、下载的厂家 PDF 和 ZIP 分发包保留在本地，由 `.gitignore` 排除。

历史验证脚本含本机绝对路径，迁移运行时需要调整。部分报告和清单引用上述本地产物；仅克隆仓库不能重新进行原生 Maxwell 求解。外部材料来源和产物哈希保存在相应 provenance / manifest JSON 中。
