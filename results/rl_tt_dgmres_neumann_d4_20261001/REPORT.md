# Neumann–Poisson 专项 RL TT-DGMRES

采用论文的各向同性归一化 Neumann 算子和 CP3 右端项族；训练、验证、测试按完整右端项实例和随机种子隔离。测试与训练共享有限个算子，只评估同算子新右端项的泛化，不声称算子不相交泛化。原论文 seed 20260906 的展示例独立保存，不混入测试统计。
固定策略和 RL 只读复用 V3 数值后端、单周期调度、保护、精度和预算。固定基线不计算不需要的策略特征；RL 的在线特征、推断和终审费用计入墙钟时间。模型加载及公共问题构造在计时外。
主基线为验证集逐维数选优的固定动作；全局固定动作是辅助对照。学习门控与基线续跑同样使用逐维数固定动作，未见维数预先退回全局固定动作。
训练标签来自同一真实重启状态的九动作分叉及完整续跑，拟合剩余时间/原始预算和失败风险。失败罚是成本目标的一部分，不构成收敛或普遍加速保证。
模型、轮数及门槛在测试前冻结。测试所有失败保留；配对时间仅取双方所有重复均成功的实例，不能替代成功率。Bootstrap 的单位是新右端项实例，不支持对任意 Neumann 算子的推论。

|方法|成功次数|全重复成功实例|
|---|---:|---:|
|fixed|12/12|4/4|
|fixed_by_dimension|12/12|4/4|
|rl|12/12|4/4|
|rl_static|6/12|2/4|

## 配对墙钟时间

- fixed_by_dimension/rl: 1.0859448244181578; 95% RHS-instance bootstrap interval [1.0349451150256408, 1.1378188062813486]; common successful instances 4.
- fixed/rl: 1.1080610618302695; 95% RHS-instance bootstrap interval [1.0565569759405267, 1.1468066368029821]; common successful instances 4.
- fixed_by_dimension/rl_static: 1.0823914661141074; 95% RHS-instance bootstrap interval [0.9492614726299318, 1.2341923902913758]; common successful instances 2.
- rl_static/rl: 1.0222243465436098; 95% RHS-instance bootstrap interval [0.9279897298303383, 1.1260282103095625]; common successful instances 2.

验收记录：{"primary_baseline": "fixed_by_dimension", "success_count_no_lower": true, "paired_speedup_above_one": true, "paired_ci_lower_above_one": true, "success_rate_statistical_noninferiority_proven": false, "operator_disjoint": false, "scope": "new independent RHS on the same finite isotropic Neumann operators"}。
观测成功数不降低不等于已证明成功率非劣；小测试集区间只作有限样本描述。rl_static 与 rl 的比较用于检查首周期选参以外的在线改参增益。集成不确定性门控为经验规则，不是严格置信界。论文已知案例使用本新求解器且无预条件，不冒充原图实现或原图计时。
