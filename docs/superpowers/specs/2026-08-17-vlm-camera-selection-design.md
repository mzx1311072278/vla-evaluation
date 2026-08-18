# VLM 多摄像头选择设计

## 1. 背景与目标

当前 VLM 评测只能使用 Profile 中固定的单个 `image_key`。目标是在新建评测时允许用户
从当前数据集的视频流中选择任意多个摄像头，并在没有勾选时自动分析数据集中的全部
摄像头。

本功能在 `codex/qwen3-vl-support` 分支继续实施。摄像头选择是每次评测任务的参数，不是
模型 Profile 的固定配置；同一个 Profile 可以针对不同摄像头组合创建可复现的任务。

## 2. 已确认的产品行为

- 新建评测页直接列出数据集摄像头复选框，不使用折叠式多选下拉框。
- 用户可以勾选一个或多个摄像头。
- VLM 启用且没有勾选任何摄像头时，后端把选择解析为数据集全部摄像头。
- 单个评测任务最多联合分析 3 个摄像头。
- 多摄像头采用联合分析：一个 Episode 只调用一次 VLM，生成一份综合判断。
- 每个摄像头分别使用 Profile 中现有的全局帧和密集帧上限，不在摄像头之间平分。
- VLM 未启用时摄像头选项不参与评测，任务保存空的摄像头列表。

因此，假设 Profile 每个摄像头最多抽取 8 张全局帧和 8 张密集帧，选择 3 个摄像头时
单次 Episode 最多向模型提供 48 张图片。界面需要提示图片量、推理时间和显存开销会随
摄像头数量增加。

当前目标数据集最多包含 3 个摄像头。服务端仍执行数量硬限制：若未来数据集包含超过 3
个摄像头，空选择不能隐式突破限制，提交接口返回 422 并要求用户明确选择不超过 3 个。

## 3. 摄像头发现与持久化

数据集检查结果增加排序后的 `camera_keys`。LeRobot 数据集从以下两类现有元数据合并
发现视频流：

1. `meta/info.json` 中 `dtype: video` 的 feature；
2. Episode parquet 中 `videos/<key>/file_index` 一类已支持的视频列。

只暴露通过当前数据集检查、并且可以解析 Episode 视频引用的摄像头。摄像头键按字符串
排序，以保证页面、任务参数和运行键稳定。导入任务和 CLI 数据集扫描把该列表保存到
`Dataset.inspection_json["camera_keys"]`。

历史数据集可能只有 `inspection_json.errors`。打开新建评测页时，若缺少合法的
`camera_keys` 列表，服务端对数据集执行一次现有的安全检查来补充页面数据；不要求用户
重新导入。此回退只用于发现摄像头，不静默改变数据集身份或绕过 READY 状态校验。

若启用 VLM 但数据集没有可用摄像头，新建页显示无可用视频流，提交接口返回 422，不创建
任务。

## 4. 页面与提交语义

新建评测页在“VLM 评测”后展示“摄像头（可多选）”：

- 每个可用摄像头是一项原生 checkbox，值为完整 camera key；
- 初始状态全部不勾选；
- 固定帮助文本说明“未选择时分析数据集全部摄像头”；
- 显示当前数据集可用摄像头数量；
- 前端在选择达到 3 个时禁用其余未选项，并保留服务端最多 3 个的权威校验；
- 提示多选会按摄像头数量增加抽帧、耗时和显存占用。

表单使用重复的 `camera_keys` 字段。提交接口只接受零个或多个字符串值，并执行以下规范
化：

1. VLM 未启用：保存 `[]`；
2. VLM 已启用且表单为空：使用数据集全部摄像头；
3. VLM 已启用且有选择：去重并按数据集摄像头顺序保存；
4. 任意提交值不属于该数据集：返回 422，不创建任务。
5. 规范化结果超过 3 个摄像头：返回 422，不创建任务。

规范化后的明确列表保存到：

- `EvaluationJob.params_json.camera_keys`；
- `EvaluationJob.provenance_json.camera_keys`；
- provenance 内嵌的 `params`；
- 任务 `run_key` 的规范化参数 JSON。

因此，空选择和手动勾选全部会归一化成同一个任务身份；不同摄像头组合不会误复用已有
成功任务。评测任务详情页展示实际摄像头列表。

## 5. Worker 与兼容性

Evaluation Worker 从数据库中读取任务创建时保存的 `params_json.camera_keys`，不在执行时
重新选择“当前全部摄像头”。这保证重试仍使用原任务的明确列表。Worker 仍会执行现有
数据集指纹校验；数据集变化时任务按原有规则失败。

网页创建的新任务始终保存 `camera_keys`。为兼容本功能上线前的历史任务：

- 参数中没有 `camera_keys` 时，Worker 使用该任务 Profile 的 `image_key`；
- Profile 的 `image_key` 字段继续保留；
- 现有命令行 `--image_key` 入口继续执行单摄像头分析，不改变用法。

重试不会重新读取网页选择，也不会把历史单摄像头任务自动扩大为全部摄像头。

## 6. Episode 数据与联合抽帧

数据读取层按 Episode 聚合所选摄像头的视频引用。每个摄像头保留自己的视频路径、相对
路径、起止时间；Episode 的成功标记和长度仍只有一份。

评测每个 Episode 时按规范化摄像头顺序依次抽帧：

- 每个摄像头独立应用 `max_global_frames`、`global_sample_interval`、
  `max_dense_frames`、`dense_sample_interval` 和 `dense_region`；
- 证据图片保存到 `frames/episode_<index>/<camera-safe-name>/`，避免不同摄像头覆盖；
- 每条 frame timestamp 增加 `camera_key`，图片文件名保持与 timestamp 条目一一对应；
- 传给 VLM 的图片按“摄像头顺序，再按该摄像头抽帧顺序”排列。

Prompt 的帧清单显式写出每张图的 `camera_key`、帧类型、Episode 时间和视频时间，使模型
能够区分不同视角。VLM 仍只返回现有的一份 Episode JSON，结果 schema、计数规则、报告
核心指标和人工复核策略不变。

Episode 结果额外记录：

- `camera_keys`：本次联合分析的摄像头；
- `sampled_frame_count_by_camera`：各摄像头实际抽帧数；
- 每个 `frame_timestamps` 条目的 `camera_key`。
- `input_token_count` 和 `context_token_limit`：本地模型处理后的真实输入长度和上下文上限；
- `cuda_peak_memory_allocated_bytes` 和 `cuda_peak_memory_reserved_bytes`：本地 CUDA 推理时
  当前 Episode 的 PyTorch 峰值显存指标。

现有总 `sampled_frame_count`、`global_frame_count` 和 `dense_frame_count` 继续保留，值为所有
摄像头之和。

## 7. 错误处理

- 伪造或过期的摄像头键：网页提交返回 422。
- 启用 VLM 但没有可用摄像头：网页提交返回 422。
- 某个摄像头无法解析 Episode 元数据：任务在 VLM 阶段失败，不用其他摄像头静默替代。
- 某个摄像头在单个 Episode 抽帧失败：该 Episode 按现有 sampling fallback 记录失败，
  继续处理后续 Episode；结果保留摄像头上下文供排查。
- 多摄像头导致 GPU OOM：沿用现有 `GPU_OOM` 错误映射；用户可以减少勾选数量后创建新
  任务。

本地 VLM 在 Processor 生成输入后、调用 `model.generate()` 前执行上下文硬校验：

```text
input_token_count + max_new_tokens <= context_token_limit
```

`input_token_count` 直接读取处理后的 `inputs.input_ids.shape[-1]`，不使用图片张数估算。
`context_token_limit` 从已加载 checkpoint 配置读取：优先读取 Qwen3 的
`text_config.max_position_embeddings`，兼容读取 Qwen2.5 的顶层
`max_position_embeddings`。若模型没有提供合法正整数上限，则在推理前报模型配置错误，
不猜测默认值。超过上限时该 Episode 生成明确的上下文超限 fallback，不调用
`generate()`，并保留实际输入、输出预留和上限供排查。

本地 CUDA 后端在每个 Episode 推理前重置 PyTorch 峰值统计，在成功或失败后读取
`max_memory_allocated()` 与 `max_memory_reserved()` 并写入 Episode 结果。Worker 当前保持
单并发，因此这些数值可作为该 Episode 的进程内峰值观测；它们不是 `nvidia-smi` 的整卡
占用，也不包含其他进程。CPU 本地后端和 API 后端将这两个字段记为 `null`。API 后端无法
观察远端模型的真实 tokenization 和 GPU 峰值，因此本次不伪造精确数据，其对应保护依赖
远端服务自身的上下文限制和监控。

错误消息和公开结果不暴露数据集绝对路径。取消、进度回调、原子结果写入和 VLM 客户端
释放行为保持不变。

## 8. 测试与验收

自动化测试覆盖：

- 数据集检查从 info feature 和 parquet 视频列发现、合并并排序摄像头；
- 新导入和 CLI 扫描的数据集保存 `camera_keys`，历史数据集可以安全回退发现；
- 新建页用复选框展示摄像头及“未选择＝全部”提示；
- 空选择归一化为全部、部分选择保持数据集顺序、重复值去重；
- 超过 3 个摄像头的显式或默认选择被拒绝，前端达到 3 个后阻止继续勾选；
- 非法摄像头键和无摄像头 VLM 提交被拒绝；
- VLM 禁用时摄像头规范化为空列表；
- 参数、provenance 和 run key 使用实际列表，不同组合不互相去重；
- Worker 传递任务级摄像头列表，历史任务回退到 Profile `image_key`；
- 多摄像头 Episode 元数据正确聚合，各摄像头独立抽取完整上限；
- VLM 单个 Episode 只调用一次，并收到带摄像头标签的全部图片；
- 本地客户端使用 Processor 后的真实 token 数校验输入与输出预留，超限时不调用模型；
- Qwen3 与 Qwen2.5 分别从正确配置位置读取上下文上限；
- 本地 CUDA 结果记录逐 Episode allocated/reserved 峰值，CPU 与 API 结果记录空值；
- 结果保存逐摄像头计数以及带 `camera_key` 的 timestamp；
- 任务详情页展示实际摄像头列表；
- 既有单摄像头 CLI、本地 VLM、API VLM、重试、报告与取消测试保持通过。

手工验收使用至少包含两个视频流的小数据集：分别提交空选择、单摄像头和双摄像头任务，
确认任务详情、provenance、证据帧目录、VLM 请求图片数和 Episode 结果均与选择一致；在
RTX 4090 上记录双摄像头峰值显存和耗时。
