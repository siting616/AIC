# Cloud Patch V35 GPU Rescue

该补丁同步3×3小目标切片和复杂Query多Prompt任务。压缩包必须解压到云端AIC项目根目录。

```bash
cd ~/autodl-tmp/AIC
tar -xzf cloud_patch_v35_gpu_rescue.tar.gz
python scripts/run_tiled_hard100.py --help | grep grid-size
```

先执行小目标任务：

```bash
python scripts/run_tiled_hard100.py \
  --hard100 outputs/experiments/small_target_rescue_v1/dev_small_misses.json \
  --expected-count 0 \
  --grid-size 3 \
  --overlap 0.20 \
  --output-dir outputs/experiments/small_target_rescue_v1/run
```

再执行多Prompt任务：

```bash
python scripts/run_multiprompt_candidates.py \
  --config configs/refcoco_multiprompt.yaml \
  --ids-json outputs/experiments/query_decomposition_rescue_v1/dev_complex_misses.json \
  --images-dir external_data/coco_val_images \
  --output-dir outputs/experiments/query_decomposition_rescue_v1/run \
  --max-prompts 4 \
  --per-prompt 5 \
  --top-k 10
```

两个脚本默认支持断点续跑。中断后重复执行相同命令即可，不要添加 `--no-resume`。
