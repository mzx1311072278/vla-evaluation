# VLM Episode Attempt Evaluation

在 `Genie02_report/attempt_eval/` 目录运行本工程。它只做本地 VLM 推理，不训练模型，不调用云端 API，不需要 API key，不修改原始数据集。

## 环境

Ubuntu 24.04 示例：

```bash
conda create -n attempt-eval python=3.10 -y
conda activate attempt-eval
cd Genie02_report/attempt_eval
```

PyTorch CUDA 版本要按本机 CUDA 环境安装。示例：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

如果 CUDA/PyTorch 版本不匹配，请到 PyTorch 官网选择适合本机驱动的安装命令。

## 下载本地模型

Hugging Face：

```bash
pip install -U huggingface_hub

huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir /home/xin/models/Qwen2.5-VL-3B-Instruct
```

ModelScope：

```bash
pip install modelscope

modelscope download --model Qwen/Qwen2.5-VL-3B-Instruct \
  --local_dir /home/xin/models/Qwen2.5-VL-3B-Instruct
```

运行时用 `--model_path` 指向本地模型目录：

```bash
--model_path /home/xin/models/Qwen2.5-VL-3B-Instruct
```

## 运行

```bash
cd Genie02_report/attempt_eval

python run_episode_attempt_eval.py \
  --dataset_root /path/to/lerobot_dataset \
  --model_path /home/xin/models/Qwen2.5-VL-3B-Instruct \
  --image_key observation.images.right_wrist \
  --output_dir outputs/attempt_eval \
  --review_mode manual_review \
  --max_global_frames 8 \
  --global_sample_interval 2.0 \
  --max_dense_frames 8 \
  --dense_sample_interval 0.5 \
  --dense_region last_third \
  --max_image_size 336
```

测试前几个 episode：

```bash
python run_episode_attempt_eval.py \
  --dataset_root /path/to/lerobot_dataset \
  --model_path /home/xin/models/Qwen2.5-VL-3B-Instruct \
  --limit 2 \
  --dry_run
```

`--dry_run` 只读取 episode 映射并抽帧，不加载 VLM。

## 输出

默认输出到：

```text
outputs/attempt_eval/
├── episode_results/
├── sampled_frames/
│   └── episode_000/
│       ├── global/
│       └── dense/
├── attempt_summary.csv
└── attempt_summary.json
```

字段含义：

- `episode_success`：来自元数据/推理的 episode 级成功结果。
- `pre_success_failed_attempt_count`：成功 episode 中，最终成功抓取之前的失败抓取次数；失败 episode 不计算。
- `attempt_count`：兼容字段，成功 episode 中等于 `pre_success_failed_attempt_count + 1`。
- `success_count`：兼容字段，成功 episode 为 `1`，失败 episode 为 `0`。
- `failed_count`：兼容字段，成功 episode 中等于 `pre_success_failed_attempt_count`。
- `needs_manual_review`：是否需要人工复核。`manual_review` 下固定为 `null`，留给用户填写。
- `review_note`：人工复核备注，默认空字符串。
- `vlm_valid`：VLM 输出是否通过 JSON 解析和字段校验。
- `parse_error`：VLM 输出为空、JSON 解析失败或字段校验失败时的错误原因。
- `raw_response`：VLM 原始输出，仅保存在单个 episode JSON 和 summary JSON 中。
- `global_frame_count` / `dense_frame_count`：全局抽帧和局部密集抽帧数量。
- `frame_timestamps`：每张抽帧图片对应的类型、episode 相对时间和视频时间。
- `auto_warning`：程序发现的风险提示，例如 `low_confidence`、`too_few_frames`、`invalid_vlm_json`。
- `review_mode`：`manual_review` 或 `auto_review`。

`manual_review` 模式只写入风险提示，不自动决定是否复核。`auto_review` 模式会按阈值把有风险的 episode 标为 `needs_manual_review=true`：

```bash
--review_mode auto_review \
--confidence_threshold 0.7 \
--min_episode_duration 3.0 \
--min_sampled_frames 3
```

## 显存不足

不要把完整视频送入模型。本脚本默认每个 episode 最多 16 张 global 图、16 张 dense 图，最大边长 384。显存不足时优先降低：

```bash
--max_global_frames 8
--max_dense_frames 8
--max_image_size 336
```

仍然不足时，换更小的 VLM 或关闭其他占用显存的程序。

## 数据集映射

脚本读取 `meta/episodes/chunk-*/*.parquet`，自动匹配包含 `file_index`、`from_timestamp`、`to_timestamp` 的列，并根据 `--image_key` 寻找：

```text
videos/observation.images.right_wrist/chunk-000/file-XXX.mp4
```

如果 parquet 字段不匹配，终端会打印所有 columns，按输出检查字段名或换正确的 `--image_key`。

## transformers 排查

如果加载 Qwen2.5-VL 时报 `Qwen2_5_VLForConditionalGeneration` 不存在，升级：

```bash
pip install -U transformers accelerate qwen-vl-utils
```

如果提示模型文件缺失，确认 `--model_path` 是完整下载后的本地目录，并包含 `config.json`、权重文件和 processor/tokenizer 文件。
