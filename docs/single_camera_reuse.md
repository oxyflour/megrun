# 单摄 Maxwell 场复用与增量响应

本实验只为当前 B0500 模组、0.1 A / 100 匝、两个相同模组同向沿 +X 放置建立原型。
间隙为电磁实体包围盒间隙，中心距 = 8.8 mm + 间隙。

## 验收量

分别记录总 Fz，以及 `Fz(gap) - Fz(2 mm)`。
后一指标消除固定单摄力基线的有限元偏差，不等于相对无限远单摄的绝对干扰力验收。
工作目标是后者相对现有细 Maxwell 参考的绝对差 ≤10 μN；总力目标 ≤0.2%。
参考自身的逐轮变化不是严格误差上界，因此接近 5–10 μN 的差仍需后续更精细或同网格差分确认。

## 直接场复用

保存单摄 Maxwell 的原生 Lorentz 力、线圈内实际 stranded J、空气域 B，以及磁性体内 H。
另一颗模组通过平移单摄空气域场表示，直接漏磁力为 `∫J_single × B_other_single dV`。
绕组积分使用嵌套矩形轮廓的体积 Gauss 积分，正体积权重与 STEP 铜体体积一致。
实际 J 来自 Maxwell，避免以理想矩形电流方向代替拐角内的真实电流分布。

Maxwell 磁静态模型中的 DC 电流分布与磁场求解解耦，见
[官方求解说明](https://ansyshelp.ansys.com/public/Views/Secured/Electronics/v251/en/Subsystems/Maxwell/Content/Maxwell/MagnetostaticFieldCalculation.htm)。
当前模组位置、电流固定时复用 J；改变几何、绕组或电流配置需要重新检查。

直接复用假设两个模组磁化保持孤立时状态，遗漏互相改变轭铁磁化的效应。
旧单摄场的第一轮实算显示总力误差不到 0.1%，但距离变化最大差 33.31 μN，未通过目标。
两个积分分辨率的直接漏磁力最大差 0.20 μN，说明仅增加积分采样不能解决主要偏差。

## 增量响应

单摄原生 H 工作点记为 H0。各磁性体磁化导数 D = ∂M/∂H。
对各向同性 B(|H|) 曲线，D 的纵向系数为 `dB/dH / μ0 - 1`，
横向系数为 `B/(μ0|H|) - 1`。磁化方向变化必须用完整 3×3 张量表示。
实现使用原生密集分段线性 B-H 数据，而不是只有纵向斜率的标量磁导率。

增量方程为 `δM_A = D_A (H_inc,A + K_AA δM_A + K_AB δM_B)`，B 模组同理。
K 使用高斯体积格林算子，属于粗空间近似；自退磁项为 -I/3。
以 `L = (I + D/3)^(-1) D` 进行局部预处理，并只分解一次单摄自响应矩阵。
不同间隙更新跨模组 K，并迭代磁化增量。
只把增量产生的力加到原生单摄力和原生直接漏磁力上，不用粗算法重算整个单摄基线。

另用锚定的非线性增量响应检查线性化敏感性。旧场 0.7 mm 粗离散下，
线性与非线性结果差不到 0.001 μN，但最大距离变化差仍约 13.46 μN。
这表明线性化对该近似模型影响小；不能据此断定高斯空间算子准确。

双摄力仅用于比较，未参与系数拟合、材料调整或比例缩放。
目前还没有由 Maxwell 外场扰动求解得到的响应库，也没有导出原生有限元刚度矩阵。

## 单摄场库质量

旧单摄有 165,544 个四面体，能量误差 0.36018%，最后一步 Fz 变化 18.89 μN。
最终补算从这份已完成的单摄网格继续，加入两侧空气查询区的 0.6 mm 长度约束，
线圈目标 0.22 mm，初始网格 388,485 个单元。约束盒标记为非模型体，物理实体体积保持一致。
采用能量目标 0.25%、每步 Fz 变化目标 5 μN，非线性残差 1e-4。

实际在第四轮之后用 Ansys 自带 `desktopproxy -abort` 正常中止，保留 853,531 个四面体的完整第四轮解。
能量误差 0.14103%，最后一步 Fz 变化 6.898 μN，**未达到原设定的完整自适应停止条件**。
固定单摄基线在相对 2 mm 的差分中相消，本轮接受这份解作探索性场库。
这不证明局部场无网格偏差。原生 asol 轮号、网格、LastAdaptive 力与第四轮历史、本构和安匝均经过审计，
场库导出必须显式指定 `--frozen-pass 4` 并校验原生与提取文件哈希。

从头生成的另一份 440,935 单元试算运行 1235.72 秒后停止，完整保留，没有纳入精度结论。
原生求解日志、网格、力历史、J/安匝与 B-H 场检查均保存。

## 新距离查询

在单摄外部的两侧空气域保存 0.2 mm 规则网格 B。
它与其他模组磁性体和线圈查询范围相匹配，可在 1–3 mm 间隙内插值查询，无需启动 Maxwell。
新距离脚本不读取双摄力数据。
插值误差另与原生直接采样比对；自响应矩阵和每个已查询间隙的交叉积分系数均可缓存。
新距离首次查询仍需生成跨模组响应和积分系数；缓存重复查询时间应单独记录。

最终 0.5 mm 离散包含 936 个磁性单元。最大距离差分参考差 15.845 μN，未通过 10 μN 目标；
相对 0.7 mm 的距离差分变化最多 0.698 μN，增量力粗/细积分差最多 0.071 μN。
完整离线查询与原生逐点场采样的最大 Fz 差为 0.048 μN。
因此目前证据支持快速探索距离趋势，尚不足以替代 Maxwell 的精确干扰差分计算。

```powershell
.\.venv\Scripts\python.exe scripts\build_reused_single_camera_field_reference.py
.\.venv\Scripts\python.exe scripts\validate_single_camera_field_library.py `
  --source-directory reused_single --record-name reused_validation_record.json --frozen-pass 4
.\.venv\Scripts\python.exe scripts\predict_camera_pair_from_library.py `
  --gaps 1.25 1.75 2.25 2.75 --output outputs\single_camera_reuse\extra_gaps.json
.\.venv\Scripts\python.exe scripts\check_single_library_interpolation.py
.\.venv\Scripts\python.exe scripts\summarize_single_camera_reuse.py
```

上述 `--frozen-pass` 命令仅对已保留的中止解做提取与审计，不再发起 Maxwell 求解。
新建完整求解时省略此参数；已有目录通过原生求解记录恢复，不能把中止记录改成正常完成。

最终实际结果以 [实验报告](../outputs/single_camera_reuse/report_zh.md) 和汇总为准。
当前三组双摄参考之前已经见过，因此不是完全未知数据盲测。
不同 camera 型号、转角、强饱和变化、小横向力和任意电流均尚未验证。
