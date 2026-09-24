# 初赛算法归档

这里保存了从初赛服务器 `/root/autodl-tmp/aic_v1_project` 取回的算法源码、辅助工具和场景列表。目录结构及算法脚本保持服务器原样；仅清除了 Python 缓存和 Jupyter 自动检查点。服务器目录本身没有 Git 历史，因此这里是一次源码快照，不代表后续复赛代码的当前版本。

## 目录与算法

| 路径 | 用途 |
| --- | --- |
| `src/run_all_v1.py` | v1 全量主流程：按场景解析查询、生成候选框、排序并输出提交 JSON。 |
| `src/parse_scene_000002_v2.py`、`src/run_scene_000002_stage*.py` | 场景 `000002` 的解析与阶段性试验脚本。 |
| `src/v2/run_v2_pilot_p0*.py` | v2 试运行版本，重点处理序数、深度、兜底查询和红外路由。 |
| `src/v2/run_v2_depth_r4*.py` | 深度关系判断的 r4 系列试验。 |
| `src/v2/run_v2_r5a_topk_boxwise.py` | 高召回候选框与逐框验证的 r5a 试验。 |
| `tools/` | 选择试验场景、合并 v1/v2 预测、构建消融提交和排查结果。 |
| `configs/` | 固定的 v2 Pilot-50 场景列表与单场景 smoke 列表。 |

v1 使用 **Qwen3-VL-8B-Instruct** 解析查询并做最终语义判断，使用 **Grounding DINO base** 检测目标与参照物候选框。脚本依据查询内容选择 RGB、红外或深度信息，对关系、数量和序数约束进行排序。v2 复用 v1 检查点，在指定场景上尝试更细的序数、深度及候选框复核；它不是独立的全量运行入口。

服务器的 `outputs/v1_full/run_summary.json` 记录 v1 完成 2000 个场景、9555 条查询，失败数为 0、兜底次数为 266。v2 的 Pilot-50 包含 50 个场景、497 条查询。这些是运行记录，并非准确率或成绩；日志、预测文件及数据集没有随源码提交。

## 运行前提

原服务器使用 Python 3.10；读取到的环境版本包括 PyTorch 2.12.0+cu126、Transformers 5.15.0、Accelerate 1.14.0、`qwen-vl-utils` 0.0.14、OpenCV 5.0.0、NumPy 2.2.6 和 Pillow 12.3.0。这是原环境记录，其他机器应按自己的 CUDA 环境安装兼容的 PyTorch。模型脚本会从 `<project-root>/models/hf_cache/hub/` 查找以下本地 Hugging Face 快照，且以 `local_files_only=True` 加载：

- `models--Qwen--Qwen3-VL-8B-Instruct`
- `models--IDEA-Research--grounding-dino-base`

需要自行准备兼容 CUDA 的 PyTorch 环境、上述模型快照和初赛数据集。默认数据路径是 `<project-root>/data/official/初赛数据集-基于大模型的多模态视觉理解与推理/`，其中需包含 `queries/queries.json` 及其记录引用的可见光、红外、深度文件。也可通过 `--data-root` 指向其他位置。模型、数据和输出不纳入 Git。

## 运行示例

以下命令在 Linux shell 中从本目录执行；`$PWD` 是本目录的绝对路径，`$DATA_ROOT` 是初赛数据集根目录。先用单场景验证环境，再运行全量数据：

```bash
cd preliminary
export DATA_ROOT=/path/to/初赛数据集-基于大模型的多模态视觉理解与推理

python src/run_all_v1.py \
  --project-root "$PWD" --data-root "$DATA_ROOT" \
  --scene 000002 --output-dir "$PWD/outputs/v1_smoke"

python src/run_all_v1.py \
  --project-root "$PWD" --data-root "$DATA_ROOT" \
  --output-dir "$PWD/outputs/v1_full" --resume
```

`--resume` 从同一输出目录的 `checkpoint_predictions.json` 继续；首次运行也可以不加。全量输出包括 `submission.json`、`checkpoint_predictions.json`、`run_summary.json`、`failures.json` 以及逐场景记录。输出框为归一化坐标。

在全量 v1 检查点和提交文件已经生成后，才运行 v2 Pilot。例如：

```bash
python src/v2/run_v2_pilot_p0_r3.py \
  --project-root "$PWD" --data-root "$DATA_ROOT" \
  --scene-list configs/V2_PILOT_50.txt \
  --v1-checkpoint outputs/v1_full/checkpoint_predictions.json \
  --v1-submission outputs/v1_full/submission.json \
  --output-dir outputs/v2_pilot50_p0_r3

python tools/merge_v2_pilot_into_v1.py \
  --baseline outputs/v1_full/submission.json \
  --pilot outputs/v2_pilot50_p0_r3/submission_v2_pilot.json \
  --scene-list configs/V2_PILOT_50.txt \
  --output outputs/v2_pilot50_p0_r3/submission_hybrid.json
```

v2 脚本保留了服务器上的历史默认路径，其中 `--v1-submission` 默认指向一份未归档的备份文件，因此示例显式指定当前 v1 提交文件。其他 v2 文件是不同试验版本，应分别指定输出目录并结合各自的 `--help` 与源码使用。消融工具依赖对应的 `change_report.json` 等运行产物。
