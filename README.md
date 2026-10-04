# 自适应TT截断控制

**Adaptive TT Truncation Control for TT-DGMRES**

面向奇异 Neumann–Poisson 线性系统的 TT-DGMRES 参数控制项目。学习策略根据 Krylov 迭代状态，在初始周期和重启周期选择 TT 截断精度与重启长度，并通过数值保护和固定配置回退完成求解。

仓库包含可运行代码、已训练模型、训练与测试原始记录、开发试跑、源码快照及一致性校验。当前完成的专项训练范围为 `n=12, d=4`；训练、验证、测试分别使用 6、2、4 个完整右端项实例，共采集 108 条真实动作分支续跑记录。网络在部署过程中保持冻结。

## 安装与运行

原实验使用 Windows、Python 3.9、CPU float64。保存的冻结批次使用 Windows 路径协议，建议先在该环境复现。

```powershell
git clone https://github.com/zhan820/adaptive-tt-truncation-control.git
cd adaptive-tt-truncation-control
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

已有模型可直接求解新的右端项，无需重新训练：

```powershell
python run_trained_neumann.py --run results/rl_tt_dgmres_neumann_d4_20261001 --seed 123456 --dims 4 --mode rl --output deployment/new_rhs_123456_rl.json
```

固定策略对照使用 `--mode fixed`，输出文件需使用另一个新名称。程序会拒绝训练、验证、测试或开发中已使用的种子，拒绝覆盖已有输出。退出码 0 表示达标，2 表示已记录未达标结果。

## 从头训练

`train_neumann.py` 使用仓库内已有的问题构造来源文件，可以在新的实验目录完成全部阶段；已有批次保留为冻结记录。

```powershell
python train_neumann.py prepare --output results/neumann_new_run --train-dims 4 --paper-dims 4 --train-count 6 --validation-count 2 --test-count 4 --max-states 2 --repeats 3 --seconds-budget 75 --paper-seconds 75 --max-tt-rank 256 --recovery-max-rank 256 --diagnostic-max-rank 512 --threads 4
python train_neumann.py fixed --output results/neumann_new_run
python train_neumann.py collect --output results/neumann_new_run --iteration 0
python train_neumann.py fit --output results/neumann_new_run --iteration 0
python train_neumann.py validate --output results/neumann_new_run --iteration 0
python train_neumann.py freeze --output results/neumann_new_run
python train_neumann.py evaluate --output results/neumann_new_run
python train_neumann.py paper --output results/neumann_new_run
```

数据采集运行真实求解器，耗时显著长于网络拟合。首轮正式采样约 85.6 分钟，网络拟合约 12.1 秒；开发试跑、基线搜索和验证另有成本。参数在独立测试之前冻结。

## 已保存结果

| 评估范围 | 固定/学习达标次数 | 配对时间加速比 |
|---|---:|---:|
| 4 个独立新右端项，每个重复 3 次 | 12/12、12/12 | 1.086 |
| 论文原始四维右端项，重复 3 次 | 3/3、3/3 | 1.465 |

独立测试对应几何平均时间比降低约 7.9%。原始展示例的时间中位数为 41.82/28.54 秒，迭代步数为 175/154。两者均采用 `eta <= 1e-7` 的验收阈值。原始展示例与独立测试分别统计。

这些是同算子新右端项的小样本结果；独立测试总迭代步数并未减少，五、六、十维尚无完成的专项训练结论。固定与学习策略共用后端和数值保护，在线特征与推断费用计入学习版求解时间。只在首周期选一次参数的对照及全部开发失败也保留在原始记录中。

完整说明见 [专项结果](results/rl_tt_dgmres_neumann_d4_20261001/RESULTS_SUMMARY.md) 与 [机器生成报告](results/rl_tt_dgmres_neumann_d4_20261001/REPORT.md)。

## 代码和数据

- `rl_tt_dgmres_neumann/`：论文问题族构造、训练、验证和测试。
- `rl_tt_dgmres_v3/`：真实续跑的 Monte Carlo 价值学习与周期调度。
- `rl_tt_dgmres_v2/`、`rl_tt_dgmres_v1/`、`adaptive_tt_dgmres/`：数值核、特征、基础接口和 TT 运算。
- `run_trained_neumann.py`：冻结模型部署入口。
- `results/rl_tt_dgmres_neumann_d4_20261001/`：模型、108 条训练分支、完整测试和源码快照。
- `results/neumann_*_pilot_20261001/`：所有已保存开发试跑，包括失败。
- `DELIVERY_MANIFEST.json`：原交付包中每个文件的字节数与 SHA-256。

原交付清单用于核验归档实验文件；发布时新增的首页、训练入口和 CI 配置由 Git 版本记录。算法与实验来源说明见 [PROVENANCE.md](PROVENANCE.md)。

## 验证

```powershell
New-Item -ItemType Directory -Force deployment | Out-Null
python audit_rl_tt_dgmres_neumann.py --output results/rl_tt_dgmres_neumann_d4_20261001 --report deployment/audit_new.json
python -m pytest rl_tt_dgmres_neumann/tests rl_tt_dgmres_v3/tests test_run_trained_neumann.py --import-mode=importlib -q
```

证据审计核对哈希、实例隔离、冻结状态、原始记录和统计汇总；数值测试检查问题构造与求解器行为。实际计时会随运行环境变化。
