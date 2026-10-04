# 论文 Neumann–Poisson 算例的专项学习策略

本目录训练针对论文问题族的参数选择器，不修改已有 V1–V4、论文正文或论文图表。

## 问题与数据

以 2026-09-29 上财模板修订稿第五章为准：每维 n=12，等权对称 Neumann 路径 Laplacian 的 Kronecker 和归一化至算子范数 1。制造参考解使用三个 CP 项，余弦模态系数为标准正态随机数乘 (1+q)^(-2)，CP 项权重为 1、1/2、1/4；整体去常量分量后归一化。右端项为 A x_star + 0.1 ||A x_star|| q0，因此线性方程不相容，验收采用 A(b-Ax) 对应的变换后向误差。

问题构造复刻原脚本的 NumPy 随机流和 1e-13 的三次制备截断，不使用旧版随机 TT 核制造解。CP 项数不是最终右端项的 TT 秩。参考解仅用于结果评价，不输入策略网络。

训练、验证、测试使用不同的右端项随机种子；同一维数可以共享同一个算子。这是针对既定算子族的新右端项泛化，不是算子互不重合的测试。论文原种子 20260906 不进入训练或验证，保留为冻结后的单独展示，不冒充从未被研究者看过的盲测。

## 比较方法

- `fixed`：验证集选出的全局固定配置。
- `fixed_by_dimension`：分别在各维数的验证问题上选出的固定配置，是主要对照。
- `rl`：按当前状态在初始周期和各重启周期选择参数；没有把握时回退至该维数的固定配置。
- `rl_static`：只在初始周期选择一次，用于检查后续动态决策的额外收益。

四者共享 V3 单周期调度、V2 数值核、选择性重正交化、截断误差检查、恢复机制和验收条件。仍使用更新维数 8/16/24 与三档精度组成的九个动作。固定版本不计算其不需要的网络特征，学习版本的在线特征和推断时间计入求解时间。

针对该族重新采集真实分支续跑：同一训练状态尝试不同首动作，再沿规定的后续策略求解，以剩余时间和失败记录学习动作价值。旧版不同问题族的结果不会充作新训练标签。正式求解时网络冻结，仅进行参数选择。

## 使用流程

在工作区根目录执行，始终为同一批次指定同一个 `--output`：

```text
python -m rl_tt_dgmres_neumann prepare --output results/neumann_new_run --train-dims 4 --paper-dims 4 --train-count 6 --validation-count 2 --test-count 4 --max-states 2 --repeats 3 --seconds-budget 75 --paper-seconds 75 --max-tt-rank 256 --recovery-max-rank 256 --diagnostic-max-rank 512 --threads 4
python -m rl_tt_dgmres_neumann fixed --output results/neumann_new_run
python -m rl_tt_dgmres_neumann collect --output results/neumann_new_run --iteration 0
python -m rl_tt_dgmres_neumann fit --output results/neumann_new_run --iteration 0
python -m rl_tt_dgmres_neumann validate --output results/neumann_new_run --iteration 0
python -m rl_tt_dgmres_neumann freeze --output results/neumann_new_run
python -m rl_tt_dgmres_neumann evaluate --output results/neumann_new_run
python -m rl_tt_dgmres_neumann paper --output results/neumann_new_run
```

`prepare` 要求新目录；运行参数、分组、源码哈希及原论文构造来源写入批次清单。先用 `--help` 核对可选预算和样本量。固定参数和学习门控仅由验证集选择，进入测试或论文展示后不能继续训练或选择该批次模型。

## 首轮专项批次

`results/rl_tt_dgmres_neumann_d4_20261001` 的范围在正式训练前固定为 n=12、d=4：6 个训练 RHS、2 个验证 RHS、4 个测试 RHS，每个训练问题最多采样两个重启状态，每个状态完整运行九个首动作分支。只做第 0 轮完整 Monte Carlo 策略改进，400 个网络训练 epoch；门控在预设的三个候选中用验证集选择。每种测试方法重复计时三次，统计单位仍为独立 RHS，而不是计时次数。

共同配置为最终容差 1e-7、最多 300 步、75 秒软时间预算、正常与恢复秩上限 256、诊断秩上限 512、四线程。单次 TT 运算和终审可能使实际用时超过软预算，是否接受仍按统一的验收规则判断。仅用原论文四维 RHS 作单独展示（`--paper-dims 4`）；未声称五、六或十维已完成专项训练。

四维、五维和六维的前期开发试跑保存在独立的 `neumann_*_pilot_20261001` 目录。已观察的 RHS 种子和预热种子均被排除出正式划分。较高维度在当前预算或秩上限下的失败也保留，不能以本批次四维结果替代高维证据。

该批次是小样本可行性试验，不足以单独支持广泛泛化或稳定提速结论。实际完成状态以批次清单、模型文件和评估报告为准；本说明不预先宣称训练成功或性能优于固定策略。

完成训练、冻结及测试后，可通过根目录 `run_trained_neumann.py` 在新的 RHS 上部署。它读取已保存的模型与计算配置，不重新训练；默认只允许本批次已训练的维数，输出独立记录，不写入测试集或覆盖原实验。

```text
python run_trained_neumann.py --run results/rl_tt_dgmres_neumann_d4_20261001 --seed 123456 --dims 4 --mode rl --output deployment/neumann_new_rhs_123456.json
```

常规论文问题采用 1e-7 的变换后向误差目标；十维原例采用论文相应的 1e-5 目标，双方一致。新求解器使用的秩预算和恢复机制不等同于原图实现，报告应列出实际配置，不把新计时冒充原图的直接复现。已知零空间约束与变换残差达标也不保证前向解误差同样小。

测试必须同时报告达标情况和共同成功问题的配对时间；不删除失败问题，不以单个加速案例替代整体证据。离线数据采集、训练及验证成本单独记录，不能计为免费的一次性端到端提速。
