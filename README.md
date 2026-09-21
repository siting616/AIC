# 复赛基线

本目录包含复赛完整运行材料：`code` 为算法代码、脚本和依赖，`复赛数据集-基于大模型的多模态视觉理解与推理` 为数据集，`outputs` 为运行结果。

当前运行的是 `center` CPU 基线：对每条查询输出固定中心框 `[0.25, 0.25, 0.75, 0.75]`，用于验证数据读取、推理流程和提交文件格式，不代表真实视觉模型效果。

提交文件：`outputs/center/submissions/submission.zip`

运行命令（在本目录的 `code` 目录执行）：

```powershell
cd D:\AIC_Challenge\复赛\code
pip install -r requirements.txt
python scripts\run_submit.py --config configs\semifinal\center.yaml --no-resume
```
