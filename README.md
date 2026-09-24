# AIC 多模态视觉理解与推理

本仓库按比赛阶段整理代码：

| 目录 | 内容 |
| --- | --- |
| [`preliminary/`](preliminary/README.md) | 从初赛服务器保存的原始算法脚本、试验版本、辅助工具和小型配置；请从这里了解初赛方案。 |
| `code/` | 复赛阶段的算法、运行脚本、配置和测试。 |
| `results/` | 已纳入版本控制的结果材料。 |

数据集、模型权重、运行日志、检查点及大体积预测文件不在 Git 仓库中。初赛运行前提和命令见 [`preliminary/README.md`](preliminary/README.md)。`preliminary/` 与 `code/` 是不同阶段的实现，不能直接视为同一版本。

复赛后续实验顺序见[算法改进建议](code/docs/semifinal_algorithm_improvement_proposal.md)。

复赛的 `center` CPU 基线对每条查询输出固定中心框 `[0.25, 0.25, 0.75, 0.75]`，用于验证数据读取、推理流程和提交格式，不代表真实视觉模型效果。在准备好复赛数据后，可从仓库根目录运行：

```bash
cd code
python -m pip install -r requirements.txt
python scripts/run_submit.py --config configs/semifinal/center.yaml --no-resume
```
