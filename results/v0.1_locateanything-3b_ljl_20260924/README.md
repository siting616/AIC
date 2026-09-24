# v0.1 LocateAnything-3B：RGB 零样本直接定位

- **日期/分支**：2026-09-24，`ljl`
- **状态**：全量预测已生成并归档，未提交平台，官方分数未知
- **模型/硬件**：`nvidia/LocateAnything-3B`，AutoDL RTX 4090 24 GB

## 文件

| 文件 | 内容 |
|---|---|
| `submission.json` | 复赛全部 5690 条查询，原始字段及顺序保持不变，仅增加归一化 bbox |
| `submission_locany_1792.zip` | **用户要求原样保留的归档包**；SHA-256 为 `52dc59eb45ed1efc4f5db3d3d64329f76dbcd64230b4f73fe0c0c476c6c3b061` |
| `run_report.json` | 环境、推理配置、耗时、异常与验证统计 |

此 ZIP 内文件名为 `submission.json`。仓库既有平台记录要求 `prediction.json`；当前归档包未改名，**不应视为平台可直接上传的包**。

## 算法迭代

v0.0 采用 Qwen3-VL 解析/裁决与 Grounding DINO 候选框；v0.1 改为 **LocateAnything-3B 直接接收 visible RGB 图与完整英文 Query，生成目标框**，以测试端到端定位能否减轻候选召回限制。模型仅使用 RGB，未引入红外或深度。完整设计与后续迭代见[算法迭代记录](../../code/docs/locateanything_iteration.md)。

推理采用 BF16、Hybrid 解码、`do_sample=False`、`max_new_tokens=128`；最长边默认 1792 像素。模型输出 0–1000 量化坐标，除以 1000 得到赛题所需 `[0,1]` 框。无有效框时依次尝试 1536、1024 像素；多框时暂取首个合法框，并在逐条断点记录中保留原始回答。

## 实测与限制

| 指标 | 结果 |
|---|---:|
| 完成量 | 5690 / 5690 |
| 原始全量推理墙钟时间 | 219.1 分钟 |
| 单条耗时中位数 | 2.249 秒 |
| 峰值预留显存 | 20.062 GB |
| 最终单框 / 多框 | 5395 / 295 |
| 自动重试 | 21 条 |
| 原始运行最终无效框 | 1 条 |
| 定向补推后无效框 | 0 条 |

无标注测试集无法计算本版 ACC@0.5。v0.1 与 v0.0、v3.0 的首框分别有 39.03%、35.34% 的查询 IoU < 0.5，表示**模型分歧**，不能当作错误率。295 条多框暂取首框是主要未验证风险。

`004054_002` 的原始模型框横坐标顺序错误；本归档中的该条结果由用户指定目标裁图后的定向改写提示词补推获得，框为 `[0.364, 0.386, 0.559, 0.469]`。这不是统一自动回退策略。赛题原文禁止人工干预测试结果生成，因此正式参赛前应使用不依赖人工目标提示的统一算法重新生成该条预测。

## 复现

运行代码：[LocateAnything 全量脚本](../../code/scripts/locateanything_full_run_v01.py)。服务器使用 Python 3.10.20、PyTorch 2.12.0+cu126、Transformers 4.57.1；模型权重来自 [NVIDIA 官方模型卡](https://huggingface.co/nvidia/LocateAnything-3B)，需另行核对其非商业许可与赛事要求。

```bash
python code/scripts/locateanything_full_run_v01.py \
  --data-root /root/aic_multimodal_dataset \
  --model /root/autodl-tmp/aic_v1_project/models/locateanything_3b \
  --output /root/autodl-tmp/aic_v1_project/outputs/locany_full_1792
```

脚本按 Query 追加 JSONL，运行中断后可续跑。版本归档未包含模型权重、原始测试图像或用户裁图。
