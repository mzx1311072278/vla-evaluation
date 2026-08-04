# Genie02 真机评测报告工具

本目录提供 Genie02 真机评测的 B 侧工具，用于读取 rollout 产生的 Session 数据与轨迹文件，计算 Episode 派生指标和 Session 核心指标，并生成 Markdown 评测报告。

## 功能

- 计算每条 Episode 的综合与左右臂平滑度
- 汇总 GSR、成功 TTS 和 Episode 平滑度
- 生成 `episode_metrics.csv`
- 生成 `metrics_core.json`
- 生成便于阅读的 `report.md`
- 支持 Native NPZ、Session 内 LeRobot Parquet 和原始 LeRobot 数据集目录

## 文件说明

| 文件 | 作用 |
|---|---|
| `genie02_eval_report.py` | B 侧总入口，依次执行全部三个任务 |
| `genie02_episode_metrics.py` | 读取 Session 与轨迹，生成 Episode 派生指标 |
| `genie02_metrics_core.py` | 汇总 Session 级核心指标 |
| `genie02_markdown_report.py` | 根据四个输入文件生成 Markdown 报告 |
| `genie02_eval_common.py` | 公共数据格式校验和文件读写函数 |

## 环境要求

- Python 3.10 或更高版本
- NumPy
- Pandas 和 PyArrow（使用 LeRobot Parquet 时需要）

推荐使用项目已有的 `genie2` Conda 环境：

```bash
conda activate genie2
pip install -r requirements.txt
```

也可以不激活环境，直接通过 `conda run -n genie2` 执行命令。

依赖版本已在 `requirements.txt` 中固定，便于复现实验报告生成环境。

## 输入目录要求

运行前需要准备一个 Session 目录；也可以直接传入包含 `meta/info.json` 与 `data/` 的原始 LeRobot 数据集目录，程序会自动从 LeRobot 元数据合成 Session 信息。标准 Session 目录基本结构如下：

```text
<session_dir>/
├── session.json
├── episodes.csv
├── rollout_config.yaml
├── raw_refs.json                 # 可选
└── ...                           # NPZ、Parquet 或外部数据引用

<运行命令的目录>/
└── report_YYYYMMDD/
    ├── episode_metrics.csv       # 运行后生成
    ├── metrics_core.json         # 运行后生成
    ├── report_YYYYMMDD.md        # 运行后生成
    └── smoothness_curve.svg      # 运行后生成，Episode 平滑度概览图
```

### `session.json`

必须包含以下字段：

```json
{
  "schema_version": "1.0",
  "session_id": "20260624_example_ee",
  "created_at": "2026-06-24T14:30:00+08:00",
  "status": "completed",
  "rollout_config_path": "rollout_config.yaml",
  "rollout_mode": "ee",
  "policy_path": "/path/to/policy",
  "task": "Pick up the object and place it in the basket.",
  "num_episodes_target": 10,
  "fps": 30,
  "dataset_backend": "native",
  "dataset_root": "/path/to/dataset"
}
```

其中：

- `status`：`recording`、`completed` 或 `aborted`
- `rollout_mode`：`ee`、`pi05` 或 `default`
- `dataset_backend`：`native` 或 `lerobot`

### `episodes.csv`

CSV 表头必须为：

```csv
session_id,episode_index,episode_path,trajectory_path,t_start,t_end,duration_s,outcome,operator_intervened,notes
```

约束：

- `episode_index` 在同一 Session 内不能重复
- `outcome` 必须是 `success` 或 `failure`
- `duration_s` 必须等于 `t_end - t_start`
- `operator_intervened` 必须是 `true` 或 `false`
- `trajectory_path` 可以留空；留空时程序会尝试通过 `episode_path` 定位轨迹
- 路径支持绝对路径和相对于 Session 目录的相对路径

## 使用方法

### 一键完成 B 侧全部任务

在本代码目录下执行：

```bash
conda run -n genie2 python genie02_eval_report.py <session_dir>
```

例如，Session 数据就在当前目录时：

```bash
conda run -n genie2 python genie02_eval_report.py .
```

`.` 表示当前目录。如果 Session 或原始 LeRobot 数据集位于其他目录，应传入实际路径：

```bash
conda run -n genie2 python genie02_eval_report.py /path/to/session_dir
conda run -n genie2 python genie02_eval_report.py /path/to/lerobot_dataset
```

总入口会按照以下顺序执行：

1. 生成 `episode_metrics.csv`
2. 生成 `metrics_core.json`
3. 读取 `session.json`、`episodes.csv`、`episode_metrics.csv` 和 `metrics_core.json`，生成 `report_YYYYMMDD.md` 和 `smoothness_curve.svg`

所有输出文件均写入运行命令所在目录下的 `report_YYYYMMDD/` 文件夹。
如需指定其他输出目录，可使用 `--output-dir <dir>`。

### 分阶段执行

如需单独调试某个阶段，必须按照以下顺序运行：

```bash
python genie02_episode_metrics.py <session_dir>
python genie02_metrics_core.py <session_dir>
python genie02_markdown_report.py <session_dir>
```

分阶段执行时，中间文件同样默认读写 `./report/`。
如果使用自定义输出目录，三个命令都要传入相同的 `--output-dir <dir>`。

## 轨迹读取规则

程序按照以下优先级查找轨迹：

1. `episodes.csv.trajectory_path`
2. `session.json.trajectory_log_dir`
3. `raw_refs.json.trajectory_log_dir`
4. `episodes.csv.episode_path`

NPZ 轨迹字段优先级：

1. `smooth_send_y`，对应时间字段优先使用 `smooth_send_t`
2. `sent_y`，对应时间字段优先使用 `sent_t`
3. `action`，主要用于 LeRobot Parquet

如果存在 `is_intervention` 或 `complementary_info.is_intervention`，只使用值为 `0` 的策略帧计算平滑度。EE 模式优先按 `meta.json` 或 LeRobot `meta/info.json` 的向量列名选取所有 EE 的 xyz 列；单右臂数据会汇总到右臂平滑度。

## 指标定义

| 指标 | 计算方式 |
|---|---|
| GSR | 成功 Episode 数 / Episode 总数 |
| TTS | 所有成功 Episode 的 `duration_s` 均值 |
| 平滑度 | 报告值 `S = log10(E + 1)`，原始量 `E = Σ ||j_k||² * Δt`，`j_k ≈ (x_k - 3x_{k-1} + 3x_{k-2} - x_{k-3}) / (Δt)^3`；综合与左右臂分别计算，越小越平滑 |

Session 平滑度汇总统计所有轨迹有效的 Episode，不区分成功或失败。

## 输出文件

### `episode_metrics.csv`

记录每条 Episode 的结果、时长、综合与左右臂平滑度、坐标空间、有效帧数和跳过原因。

### `metrics_core.json`

记录 Session 级汇总结果，包括：

- Episode 总数、成功数和失败数
- GSR
- 成功 Episode 平均 TTS
- Episode 左右臂平滑度的均值、标准差、最小值和最大值

### `report_YYYYMMDD.md`

报告包含以下章节：

1. 评测配置
2. 核心指标
3. Episode 明细
4. 失败案例
5. 结论

### `smoothness_curve.svg`

核心指标中的 Episode 平滑度概览图。每个 Episode 一根柱，柱高为综合平滑度；颜色区分成功/失败，黑色描边表示存在遥操介入。

## 常见问题

### 提示缺少 `session_dir`

错误信息：

```text
genie02_eval_report.py: error: the following arguments are required: session_dir
```

执行命令时需要在末尾传入 Session 目录。当前目录是 Session 目录时使用 `.`：

```bash
python genie02_eval_report.py .
```

### 无法读取 Parquet

确认当前 Python 环境已安装 Pandas 和 PyArrow：

```bash
pip install -r requirements.txt
```

### 无法计算平滑度

程序不会因此中断整个 Session。对应行的 `smoothness` / `left_smoothness` / `right_smoothness` 会留空，具体原因记录在 `smoothness_skipped_reason` 中。请检查轨迹路径、轨迹字段、时间戳、有效帧数和介入标记。

## 查看命令帮助

```bash
python genie02_eval_report.py -h
```
