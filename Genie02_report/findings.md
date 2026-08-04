# Genie02_report 分析发现

## 初步结构
- 根目录：episode 指标计算、聚合、Markdown 发版报告。
- `attempt_eval/`：数据读取、抽帧、VLM 调用、复核策略、结果落盘。

## 入口与依赖
- 根目录入口：`genie02_eval_report.py::generate_report()`，按顺序调用 episode 指标、核心汇总、Markdown 报告。
- 根目录核心依赖：NumPy；LeRobot Parquet 输入另需 pandas、pyarrow。
- VLM 入口：`attempt_eval/run_episode_attempt_eval.py::main()`。
- VLM 核心依赖：transformers、accelerate、qwen-vl-utils、OpenCV、Pillow、pandas/pyarrow。
- 代码以函数式管线为主，不是面向对象框架；数据类/核心类只有 `Trajectory`、`EpisodeMeta`、`ReviewConfig`、`LocalVLMClient`。

## 规模
- 根目录 Python 约 1300 行；`attempt_eval` Python 约 950 行。
- 公共格式校验集中在 `genie02_eval_common.py`，指标算法集中在 `genie02_episode_metrics.py`。

## 根目录调用链
1. `genie02_eval_report.generate_report()`：校验目录并准备输出目录。
2. `generate_episode_metrics()`：`load_session/load_episodes` -> 每条 `_metric_row` -> `_load_trajectory` -> `_smoothness` -> `episode_metrics.csv`。
3. `generate_metrics_core()`：读取 episode 与派生指标，按 `episode_index` 一对一校验/连接，计算 GSR、成功 TTS、左右臂统计，写 `metrics_core.json`。
4. `generate_markdown_report()`：重读 session、episode、派生指标、核心指标，生成 SVG 柱图和日期命名 Markdown。

## 输入兼容与算法
- `genie02_eval_common.load_session/load_episodes` 同时支持标准 Session 文件和原始 LeRobot 数据集；后者从 `meta/info.json` 与 episode parquet 合成 session/episodes。
- `Trajectory` 统一 NPZ 与 Parquet 为 values/times/intervention/space/arm。
- 轨迹字段优先级：`smooth_send_y` -> `sent_y` -> `action`；EE 模式只选 xyz。
- 平滑度先过滤非有限值和介入帧、按时间排序去重，再以时间中位间隔或 1/fps 计算三阶差分 jerk；报告值为 `log10(sum(jerk^2)*dt+1)`。
- 单 episode 的轨迹异常不会终止任务，而是写入 `smoothness_skipped_reason`；输入契约/跨文件一致性异常会终止。

## attempt_eval 调用链
1. `run_episode_attempt_eval.main()` 解析参数，创建 `ReviewConfig`，通过 `read_episode_metadata()` 构造 `EpisodeMeta` 列表。
2. 每个 episode 用 `sample_episode_frames()` 做 global 稀疏抽帧和 dense 密集抽帧；优先 PyAV，失败后回退 OpenCV。
3. 元数据明确 failure 时跳过 VLM；dry-run 只抽帧；其余通过 `LocalVLMClient.analyze()` 调用本地 Qwen2.5-VL。
4. `extract_json()` 从纯文本/夹杂文本中抽 JSON，`validate_vlm_result()` 校验成功标志、失败尝试数量、置信度、时间等，并过滤“接近但未闭合”假尝试。
5. `apply_review_policy()` 根据置信度、时长、抽帧数量、遮挡、JSON 有效性追加 warning；manual 模式不自动设复核，auto 模式有 warning 即需复核。
6. `save_episode_result/write_summary` 输出逐 episode JSON、汇总 JSON、扁平 CSV。

## attempt_eval 核心对象
- `EpisodeMeta`：episode 与视频片段的映射数据。
- `ReviewConfig`：复核阈值参数。
- `LocalVLMClient`：本地模型/processor 生命周期与单次多图推理。
- `Trajectory`：属于另一条指标管线，与 attempt_eval 没有直接调用关系。

## 样例与边界
- 当前样例是 60 episode、30 FPS、右腕 480x640、10 维单右臂 EE+rot6d+夹爪的 LeRobot v3 数据。
- `report_20260708` 显示 54/60 成功，GSR 0.9；60 条右臂平滑度有效，左臂为空，符合 `single_arm=right` 推断。
- `attempt_eval/outputs` 已有多版 60 条逐 episode 结果，可用于对照 VLM 输出过滤与 warning。
- 两条管线没有代码级汇合：主报告不读取 `attempt_summary.csv/json`。
- 没有发现自动化测试目录或测试文件。

## 阅读重点与风险
- 项目是批处理 CLI/函数式管线，不需要寻找“应用主类”。
- 先从公开编排函数倒推，再读私有 helper；不要从 common 文件第一行顺读全部实现。
- 平滑度使用单一中位 `dt`，介入帧过滤后也不重采样；理解/修改算法时要注意不规则时间间隔和过滤后间隙的影响。
- VLM prompt 针对成功 episode 和药盒抓取，且模型类硬编码 Qwen2.5-VL，不是通用 VLM 适配层。
- `attempt_eval` 的失败 episode 被直接跳过，指标语义是“成功前失败抓取次数”，不是所有 episode 的尝试次数。
- 默认输出目录相对运行命令所在目录，而非数据集目录。
