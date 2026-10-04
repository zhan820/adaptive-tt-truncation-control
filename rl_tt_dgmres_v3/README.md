# RL TT-DGMRES V3

本版使用真实续跑回报进行 Monte Carlo 近似策略迭代。训练时从同一个真实重启状态出发，分别尝试九个首动作，随后沿基准策略（或上一轮学习策略）运行至终止。标签是实际剩余求解时间和失败结果；没有用参考解生成奖励，也没有将模型预测冒充试跑数据。

固定与学习配置共用 V2 数值核和 V3 单周期外层调度。32 维在线特征包括输入 TT 秩、存储量、范数与已有迭代信息。固定基线不计算其不需要的策略特征；RL 的全部特征、推断、正常初始化、失败恢复和最终数值核验均计入在线时间。网络在测试期间冻结，只有参数选择在线进行。离线保存检查点仅用于训练。

命令按下列次序执行（所有命令指定相同的 `--output`）。`prepare` 要求新目录；后续步骤可以恢复未完成的逐算例日志。

```powershell
python -m rl_tt_dgmres_v3 prepare --output results/v3_example
python -m rl_tt_dgmres_v3 fixed --output results/v3_example
python -m rl_tt_dgmres_v3 collect --output results/v3_example --iteration 0
python -m rl_tt_dgmres_v3 fit --output results/v3_example --iteration 0
python -m rl_tt_dgmres_v3 validate --output results/v3_example --iteration 0
python -m rl_tt_dgmres_v3 collect --output results/v3_example --iteration 1
python -m rl_tt_dgmres_v3 fit --output results/v3_example --iteration 1
python -m rl_tt_dgmres_v3 validate --output results/v3_example --iteration 1
python -m rl_tt_dgmres_v3 freeze --output results/v3_example
python -m rl_tt_dgmres_v3 evaluate --output results/v3_example
```

固定主动作在验证集九个联合参数中选优，学习门控也只在验证集选择。进入独立测试后，训练与选择命令拒绝继续运行。测试按照算例和重复次序串行、随机交错运行，保存所有失败。`rl_static` 仅在首周期选一次，用于区分按问题选择与后续在线调整的贡献。

结果必须同时看成功率、配对时间和计时区间；不能只报告双方成功的小子集或把经验门控称作收敛保证。模型首次载入和问题构造不计入每次求解时间；训练、验证选择与载入成本应另行报告。此处的奇异系统仍是已知常量零空间的指数一对称系统，不能直接外推到一般 Drazin 逆求解问题。
