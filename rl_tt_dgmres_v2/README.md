# RL TT-DGMRES 第二版

第二版在重启周期开始时联合选择更新维数、矩阵向量乘法截断精度、正交化截断精度和解更新截断精度。第一版代码保留；第二版的舍入器、求解器回调、控制器和实验输出位于本目录。

## 运行

在项目根目录运行，环境需要 `torch`、`tntorch`、`numpy` 和 `pytest`。本机验证使用 `D:\py\Anaconda3\python.exe`。

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& 'D:\py\Anaconda3\python.exe' -m pytest rl_tt_dgmres_v2/tests -q
```

先用小样本核验完整流程；这个配置不覆盖全部规模，也不足以判断学习效果：

```powershell
& 'D:\py\Anaconda3\python.exe' -m rl_tt_dgmres_v2 `
  --output results_rl_v2_smoke `
  --train-count 9 --validation-count 3 --test-count 3 --ood-count 2 `
  --episodes 18 --eval-every 9 --training-seeds 20261001 20261002 `
  --repeats 1 --seconds-budget 10 --max-iterations 80
```

完整默认配置为训练/验证/测试/OOD实例数 `96/12/12/6`，每个训练种子288个episode，两个种子，每个测试配置重复3次：

```powershell
& 'D:\py\Anaconda3\python.exe' -m rl_tt_dgmres_v2 --output results_rl_v2_full
```

输出目录必须尚不存在。`--skip-legacy` 可跳过第一版实现对照。时间预算在周期和迭代回调处检查，并包含最后独立核验；单个数值运算期间不强行中断，实际失败耗时可能超过预算。

2026-10-01 本轮正式运行使用以下配置，与上面的默认配置不同。复现时须换用尚不存在的输出目录：

```powershell
& 'D:\py\Anaconda3\python.exe' -m rl_tt_dgmres_v2 `
  --output results/rl_tt_dgmres_v2_run_20261001 `
  --train-count 96 --validation-count 9 --test-count 12 --ood-count 6 `
  --episodes 192 --eval-every 96 --training-seeds 20261001 20261002 `
  --repeats 3 --seconds-budget 20
```

完成后的独立计数、源码和配对统计检查可运行：

```powershell
& 'D:\py\Anaconda3\python.exe' audit_rl_tt_dgmres_v2.py `
  --results results/rl_tt_dgmres_v2_run_20261001
```

## 接口

- `problems.ProblemSpec`：兼容第一版字段，增加每轴局部边权 `edge_weights` 和仅依赖规模的 `curriculum_tier`；JSON数据可用 `spec_from_dict` 恢复。
- `problems.build_problem(spec)`：构造TT算子、右端项、常数零空间向量和评估用参考解。`online_view()` 隐去参考解、随机种子和问题标签。
- `problems.make_splits(seed, train_count, validation_count, test_count, ood_count)`：生成算子组互斥的数据划分。独立调用的默认数量为 `48/12/12/6`，实验CLI默认训练数量为96。
- `runner.ACTIONS`：更新维数 `8/16/24` 与三组精度的笛卡尔积，共9个动作。
- `runner.run_episode(problem, config, policy, evaluate_reference=False)`：`policy` 接收20维在线数值状态，返回动作下标；结果包括summary、逐周期记录、训练转移、恢复和舍入日志。训练与验证关闭参考解误差评估。
- `rounding.FastRounder(audit_mode="tail")`：纯TT的float64 CPU舍入器。使用 `EpisodeConfig(rounding_audit="full")` 可在Python接口中启用逐次差分核验。
- `EpisodeConfig(orth_error_mode="arnoldi_budget")`：第二版默认以Arnoldi列尺度分配正交化绝对误差预算；`"relative"` 保留逐次中间向量相对误差模式，供受控比较。
- `EpisodeConfig(direct_breakdown_correction=True)`：第二版默认在指数一、单步近不变分支尝试基于真实变换残差的一维修正；所有第二版策略共享此选项。

上面显式写出 `evaluate_reference=False`，用于不计算参考解误差的训练和验证；函数实际默认值是 `True`。`predict` 命令目前是对合成问题运行冻结模型的实验示例，仍会在验收后计算已知参考解误差，并只输出指标和日志，不返回最终TT解。因此这一版是研究实验接口，不是可直接接入任意无真解问题的通用求解器接口。参考解不进入策略状态、动作选择或数值验收。

三组 `(matvec_eps, orth_eps, solution_eps)` 分别为 `(1e-8,1e-10,1e-10)`、`(1e-5,1e-8,1e-8)`、`(1e-3,1e-6,1e-7)`。默认模式下，`orth_eps` 表示下面定义的列预算比例。残差和变换残差截断精度平时采用配置值，失败恢复时一起收紧；独立诊断精度保持配置值。控制状态包含已观察的残差、秩、周期时间、迭代数、过去动作和恢复次数；不包含参考解误差或未来最优动作。

## 问题与数据隔离

问题家庭包含Neumann路径、带正边权的路径和带正边权的环图。每个局部图连通，其对称Laplacian只有常数零空间；正系数Kronecker和因此是指数一奇异系统。按局部最大特征值之和归一化后 `||A||₂=1`。构造只用小型局部矩阵及TT运算，不展开 `n**d` 张量。

ID数据覆盖 `n=4…12`、`d=2…6`、请求的低秩参考张量参数 `rhs_rank=1…5`；该参数不是承诺右端项经过算子作用后的精确TT秩。OOD采用 `d=7,8`、`n=4,5,6` 和秩参数不超过3，控制构造与核验开销。兼容与带零空间污染的右端项同时出现。

算子组键依据实际归一化局部矩阵构造，忽略问题名、参考解种子和右端项参数，也消除共同缩放；矩阵字节的SHA256另行保存。训练、验证、测试和OOD的算子组互斥。训练episode仅使用训练集；验证集选择固定动作、模型checkpoint和两个训练种子中的最终模型。测试与OOD在这些选择完成后运行，网络冻结，结果不反用于选择参数。

## 数值保护与验收

第二版的固定、启发式和神经策略共享选择性重正交、秩上限、残差检查和失败恢复。安全过滤器会在接近目标或停滞时收紧精度。舍入失败、残差增长或范围约束失效时，回到最后通过检查的周期解；已花费时间和已完成Arnoldi步仍计入预算。

接近不变子空间时，正交化中间向量可能因抵消而很小，对它持续要求严格相对误差容易造成不必要的秩增长。默认 `arnoldi_budget` 以截断后的矩阵向量乘积 `u_j = round(A v_j)` 为尺度，设本列预算 `tau_j = orth_eps * ||u_j||`，将其均分给最多 `2(j+1)` 次MGS减法后的舍入，第二遍重正交也使用这一份列预算。若仅在两遍正交化结束后统一舍入，则该次使用整列预算。舍入器以 `absolute_tolerance` 接收每次绝对误差额度；因此个别很小中间向量的相对误差可以大于 `orth_eps`，日志会明确记录预算模式和额度。

当新方向范数落到矩阵向量乘法与正交化的截断噪声尺度内，求解器触发独立候选残差核验，不再将这个方向归一化加入基底。较宽松的投影残差触发阈值也仅用于提前安排候选核验；它不放宽最终验收目标。该列预算和噪声尺度描述浮点局部截断，不是整个迭代的严格误差界，也不能直接证明发生精确不变子空间或宣布收敛。

对于指数 `a=1`、本周期仅完成一步 `k=1` 的近不变分支，默认尝试直接最小化 `||A(b-Ax0) - y A²v||`，据此构造 `x0 + y v` 候选。该计算使用未截断的算子作用，形成有效候选时额外需要4次矩阵向量乘法；实际次数记录在 `direct_correction_matvecs` 等字段中，耗时计入在线求解和奖励。它不假定已出现精确不变子空间，修正后的解仍须通过独立残差与范围核验。

候选未达目标但通过周期检查、且本次求解的变换残差下降超过1%时，从该部分修正继续重启，保留原秩上限和策略控制，不消耗失败恢复次数。缺乏进展则进入有限失败恢复。两条路径都继续累计已完成迭代和耗时，受总预算限制。

当前设计在首次失败恢复后，将本episode余下所有周期永久切换到专用保守配置 `RECOVERY_ACTION`：更新维数24，三种精度均为 `1e-12`；残差与变换残差的截断精度也收紧到原配置与 `1e-12` 中的较小值。它不是策略可选的9个动作之一，日志用执行动作下标 `-1` 及独立控制参数标明。失败恢复时秩上限逐次扩展但受恢复上限约束。日志同时记录策略提出的动作和实际执行的动作，便于识别保护机制接管的比例。

默认 `tail` 模式累计QR/SVD舍入中的丢弃奇异值尾能量，避免每次形成输入与输出之差。`full` 模式额外用独立QR范数核验该差。检测到强消去时，舍入器先尝试QR/SVD规范化并强制核验实际差分；只有候选不满足当前误差预算且原表示符合秩上限时，才原样保留输入。接近机器精度的请求或QR范数为零另有直接保留保护。这些都是浮点诊断，尾能量不是含舍入误差的严格上界。

最终重新用未截断TT运算核验

`eta = ||A(b-Ax)|| / (||x|| + ||Ab||)`

以及解的常数零空间分量。默认目标为 `eta <= 1e-6`、范围缺陷不超过 `1e-9`，且未超时。失败重算、状态提取、网络推断和最终核验均计入在线时间；评估用参考解误差在其后计算。训练耗时单列。保护机制不构成一般收敛保证；秩、迭代、恢复次数和时间限制仍可导致失败。

## 如何解读结果

`legacy_fixed` 用于比较两代实现；第二版改变了舍入实现、正交策略、动作空间和恢复机制，因此相对第一版的加速不能全部归因于强化学习。Arnoldi列绝对误差预算、噪声方向保护、提前候选核验和单步近不变分支的直接修正也由所有第二版策略共享，属于数值实现改进。学习策略相对相同第二版底层的 `v2_fixed` 和 `v2_heuristic` 才用于衡量策略选择的增量收益。

固定主策略由验证集在9个动作中选择，测试顺序随机化。配对加速比先对每个实例的重复运行时间取中位数，再计算实例间比值的几何平均；只有双方全部重复成功的实例进入配对统计。失败仍完整计入成功率和状态统计。少量训练种子与这些已知零空间的对称指数一系统，不能代表一般非正规或高指数系统。

主要产物为 `manifest.json`、`splits.json`、`dataset.json`、源码快照、训练/验证记录、`controller.pt`、逐实例评估日志、`aggregate.json`、`paired_speedups.json` 和 `REPORT.md`。正式实验开始后修改数值或训练源码会触发源文件哈希检查失败；应使用新输出目录重跑。
