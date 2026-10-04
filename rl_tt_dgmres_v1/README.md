# 强化学习自适应 TT-DGMRES：第一版

保留 TT-DGMRES 数值内核，固定重启长度，由离散 Q 网络在每个周期开始选择 TT 截断容差。新增模块与旧论文程序分开，复用 `adaptive_tt_dgmres` 的只读求解器及 TT 后端，不覆盖原实验。

## 首轮已完成的结果

正式结果位于 `results/rl_tt_dgmres_v1_completed_20260930/REPORT.md`。48 个训练实例完成 192 次训练，产生 547 条周期转移和 2928 次参数更新；模型、生成参数和原始轨迹均已保存。新旧模块共 90 项测试通过，冻结模型的独立加载推理也已通过。

学习策略在 12 个常规留出实例上全部达标（每个重复两次），但相对于验证集选出的固定容差，逐实例配对时间比的几何均值为 0.861，尚无整体加速。6 个更高维留出实例上，学习策略均未达标，固定策略有 3 个实例两次均达标。不能把第一版流程跑通等同于性能优势；详细核验见 [VERIFICATION.md](VERIFICATION.md)。原论文图表未修改。

## 本版已经实现

- 归一化 Neumann–Poisson 奇异系统的数据生成：变化尺寸、维数、各向异性、右端频率和秩；包含相容与不相容右端。
- 按实际算子分组隔离训练、验证、测试；保存生成参数、实际矩阵 fingerprint 与代码 SHA256。
- 每周期选择容差，默认只作用于 `matvec` 后的 TT-round。其他截断精度、正交策略和诊断精度固定。
- 两层 64 节点 MLP、经验回放、目标网络及 Double-DQN 更新。有限 episode 采用 γ=1；训练不是旧版的固定后续策略成本回归。
- 奖励为负的完整在线耗时/时间预算，失败额外罚 2。周期转移、失败、实际执行的动作和局部截断审计均落盘。
- 独立重算 `eta = ||A(b−Ax)||/(||x||+||Ab||)` 并检查零空间分量。参考解只用于最终评价，不进入状态、奖励或验收。
- 固定参数、手工规则和学习策略的配对留出比较，以及模型保存、重新加载和推理入口。

这里 `update_dim=m` 指更新空间维数，代码内部正常周期执行至多 `m+1` 次 Arnoldi（指数 a=1）。不把两种计数混用。

## 运行训练

在 `D:\张量tt格式下的dgmres算法` 目录运行。使用现有 Python/Torch/tntorch 环境，无需下载预训练模型或外部数据。

需要重建环境时，可使用项目已有的 `requirements_tt_simpler_dgmres.txt`；本轮是在 Python 3.9.13 的 CPU 环境中验证的。

```powershell
python -u -m rl_tt_dgmres_v1 --output results/rl_tt_dgmres_v1_new_run --seed 20260930
```

默认生成 48 个训练、12 个验证、12 个同分布测试、6 个分布外实例，训练 192 个完整 episode。默认固定更新维数 12，精度目标 1e-6。输出目录必须新建或为空，防止覆盖旧数据。

快速流程检查：

```powershell
python -u -m rl_tt_dgmres_v1 --output results/rl_v1_quick_check --episodes 4 --train-count 4 --validation-count 2 --test-count 2 --ood-count 1 --eval-every 2 --eval-repeats 1 --batch-size 4
```

`--scope arnoldi` 是另外一项明确标记的实验：同时控制 matvec 和 orth 截断。默认 `matvec` 实验不会暗中使用它。重启长度学习不在本版范围内。

## 使用训练好的模型

```powershell
python -m rl_tt_dgmres_v1.predict --checkpoint results/rl_tt_dgmres_v1_completed_20260930/controller.pt --n 6 --dims 3 --contamination 0.1 --output results/rl_v1_prediction.json
```

推理会恢复训练时的特征定义、动作及数值配置。模型加载不继续探索或更新权重。`last_checkpoint.pt` 还保留优化器、回放和 RNG 状态。

中断后从最近一个完整检查点继续，必须指定新的输出目录，旧记录不会覆盖：

```powershell
python -u -m rl_tt_dgmres_v1 --resume-from results/rl_tt_dgmres_v1_run_20260929 --output results/rl_tt_dgmres_v1_resumed_run
```

检查点之后已经写出、但没有对应权重状态的轨迹不纳入续训历史；它们仍保留在原目录。恢复过程继承原实验配置，不能在中途改变容差目标、数据划分或动作集合。

## 文件

| 文件 | 内容 |
|---|---|
| `problems.py` | 算例、算子身份、独立核验 |
| `runner.py` | 周期状态/动作接口、同口径数值验收、完整代价记录 |
| `learning.py` | Double-DQN、回放、训练与 checkpoint |
| `experiment.py` | 数据生成、验证选择、正式留出测试与报告 |
| `predict.py` | 加载模型求解新的同类算例 |
| `tests/` | 数值性质、数据隔离、动作作用范围、训练及恢复测试 |

输出中 `splits.json` 与 `dataset.json` 足以按归档代码重新生成 TT 问题；不保存不可承受的完整高维数组。`train/episode_*.json` 是实际求解产生的训练轨迹，不是编造的标签。`controller.pt` 按验证集成功数优先、达标耗时其次选择；测试集只作最后评价。

## 验证和数值边界

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest adaptive_tt_dgmres/tests rl_tt_dgmres_v1/tests -q
```

为避免在极小的正交化余量上触发旧后端的消减误差置零，当余量范数小于 1e-10 且原秩已满足上限时，本版保留该 TT 向量而不压缩。恒等复制的截断误差为零；这不放宽任何容差，所有策略完全一致，并在日志中显式记录。随后仍由 Arnoldi 中断检查与独立残差核验决定是否停止。

残差核验属于无额外 TT 截断的浮点计算，不是区间误差认证；变换残差达标也不等于同样大小的前向误差，因此单独报告参考解误差。当前硬秩限制不够时直接拒绝，不伪造满足精度的结果；尚未实现失败周期回滚。

完整在线时间以 `wall_seconds` 为准。`round_seconds`、`round_audit_seconds` 和 `near_cancellation_check_seconds` 是部分耗时分项，不能单独代替总时间。早期训练日志的极小向量恒等保留分支曾误记 `cycle=0`；实际周期仍可由 `cycles` 中的 `round_start` 切片确定，训练状态、动作和奖励不受该日志字段影响。修正前的源码随原实验归档。

第一版结果以实际 `REPORT.md` 为准。高维留出可能受固定秩上限限制；一次训练与两次计时不足以证明普遍加速或跨问题族泛化。旧论文图表没有改成 RL 结果。
