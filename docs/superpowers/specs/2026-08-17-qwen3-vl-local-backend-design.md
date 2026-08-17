# Qwen3-VL 本地后端兼容设计

## 1. 背景与目标

当前本地 VLM 后端在 `LocalVLMClient` 中直接加载
`Qwen2_5_VLForConditionalGeneration`，因此只能使用 Qwen2.5-VL 架构。目标是在不影响
现有 Qwen2.5-VL 评测的前提下，增加 `Qwen/Qwen3-VL-8B-Instruct` 本地推理支持。

本功能从 `codex/evaluation-job-timeout` 分支继续实施，以保留较长评测任务超时配置。
Qwen3-VL 和 Qwen2.5-VL 使用独立评测 Profile，避免任务去重错误复用另一模型的历史结果。

## 2. 范围

包含：

- 本地 VLM 后端同时支持 Qwen2.5-VL 和 Qwen3-VL Instruct 模型。
- Profile 显式记录本地模型族。
- 为 Qwen3-VL-8B-Instruct 增加独立 Profile。
- 统一模型加载、视觉预处理和确定性生成行为。
- 更新 GPU 依赖、部署说明和任务 provenance。
- 使用模拟依赖完成自动化测试，并提供服务器真实 GPU 冒烟步骤。

不包含：

- Qwen3-VL Thinking 模型。
- 量化、Flash Attention 2 或多 GPU 分片的自动配置。
- 在网页上让普通用户自由选择模型路径或模型族。
- 修改 VLM Prompt、结果 JSON Schema 或人工复核策略。
- 将 Qwen3-VL 的准确率视为与 Qwen2.5-VL 等价；上线前仍需样本验证。

## 3. Profile 设计

本地 VLM Profile 增加可选字段：

```yaml
vlm:
  backend: local
  model_family: qwen3_vl
  model_path: /srv/vla-eval/data/models/Qwen3-VL-8B-Instruct
```

允许值为：

- `qwen2_5_vl`
- `qwen3_vl`

为了兼容已有 Profile，`backend: local` 且缺少 `model_family` 时默认解释为
`qwen2_5_vl`。API 后端不接受 `model_family`，API 模型身份继续由 `vlm.api.model`
描述。

保留现有 `genie02-full` Profile 和版本，继续指向 Qwen2.5-VL。新增
`genie02-qwen3-vl.yaml`，名称为 `genie02-qwen3-vl`，版本从 `1.0.0` 开始，默认指向
Qwen3-VL-8B-Instruct。独立名称使任务运行键与 Qwen2.5-VL 分离。

提交评测时，任务 provenance 增加 `vlm_model_family`。报告继续展示模型路径，并可以
从 provenance 追溯实际模型族。

## 4. 本地客户端设计

`LocalVLMClient` 接收 `model_family`。模型使用
`AutoModelForImageTextToText.from_pretrained()` 加载，Processor 继续使用
`AutoProcessor.from_pretrained()`。Transformers 的 AutoModel 映射负责选择
Qwen2.5-VL 或 Qwen3-VL 的具体生成类，避免复制两套客户端。

加载要求：

- 仅从配置的本地目录加载，不自动访问网络。
- CUDA 可用时使用 `device_map="auto"` 和 `dtype="auto"`。
- CPU 回退保留 FP32，但明确属于低性能降级路径。
- 输入 Tensor 移动到模型报告的设备，不写死 `cuda` 字符串。
- 加载异常继续转换为不泄漏私有路径的 `ModelLoadError`。

Profile 声明的模型族必须与模型目录 `config.json` 中的 `model_type` 一致。支持的映射为：

- `qwen2_5_vl` -> `qwen2_5_vl`
- `qwen3_vl` -> `qwen3_vl`

不一致时在加载权重前失败，提示模型配置不匹配，避免误把模型目录与 Profile 混用。

## 5. 视觉预处理与生成

现有多图消息结构保持不变：每个抽帧作为一个 image 内容项，最后追加包含时间信息的
Prompt。

调用 `process_vision_info` 时传入
`processor.image_processor.patch_size`。Processor 调用增加 `do_resize=False`，因为
`qwen-vl-utils` 已经完成与模型 patch size 对齐的缩放，避免二次调整图片。

生成参数显式设置：

```python
model.generate(
    **inputs,
    max_new_tokens=configured_max_new_tokens,
    do_sample=False,
)
```

这样不会继承 Qwen3-VL checkpoint 默认的采样设置，保证相同输入的评测 JSON 尽可能
可复现。生成 token 裁剪逻辑保持不变，解码时设置
`clean_up_tokenization_spaces=False`。

Prompt 仍使用 `genie02-attempt-v1`，JSON 解析、字段校验、失败降级和人工复核逻辑均不
改变。

## 6. 依赖与部署

GPU extra 更新为至少包含：

- `transformers>=4.57.0`
- `qwen-vl-utils==0.0.14`
- `torchvision`
- 现有 `torch`、`accelerate`、Pillow、OpenCV、PyAV 依赖

不在本次改动中强制安装 Flash Attention 2。Qwen3-VL-8B-Instruct 的 BF16 权重约
17.5 GB，单张 24 GB RTX 4090 理论上可行，但还需要视觉 token、KV cache、CUDA 和
框架开销，因此不能保证所有样本均完全驻留显存。继续保持 Evaluation Worker 单并发，
沿用每个 Episode 最多 16 张、最大边 336 的抽帧限制。

`/czj` 部署时模型示例路径为：

```text
/czj/model/Qwen3-VL-8B-Instruct
```

运行时 Profile 位于：

```text
/czj/code/vla-evaluation/data/profiles/genie02-qwen3-vl.yaml
```

部署人员同步 Profile 后必须把 `model_path` 改为服务器真实路径，并让 Web 与 Evaluation
Worker 使用同一个 `VLA_EVAL_PROFILES_ROOT`。

## 7. 错误处理与回退

- 模型目录不存在或不完整：任务在 VLM 阶段以 `MODEL_LOAD_FAILED` 失败。
- Profile 模型族与 `config.json` 不一致：在权重加载前失败，并保留已有 METRICS 产物。
- Transformers 版本过低或模型架构不可用：转换为模型加载错误，日志保留底层异常供管理员
  排查。
- 单 Episode 推理或 JSON 解析失败：沿用现有逐 Episode fallback，继续处理其他 Episode。
- 显存不足：沿用现有 `GPU_OOM` 映射；管理员可切回 `genie02-full`、减少抽帧或后续评估
  量化方案。

Qwen2.5-VL Profile 始终保留，作为部署和精度问题的即时回退路径。

## 8. 测试与验收

自动化测试覆盖：

- 旧 Profile 缺少 `model_family` 时仍加载为 `qwen2_5_vl`。
- Qwen3 Profile 正确加载 `qwen3_vl`，API Profile 拒绝 `model_family`。
- 非法模型族和模型目录声明不一致时失败。
- 本地客户端通过 AutoModel 加载，并传递正确的 dtype、device map 和离线参数。
- Qwen3 patch size 传入 `process_vision_info`，Processor 禁止二次 resize。
- 生成显式关闭采样，输入移到模型设备，解码保持确定性设置。
- Evaluation 编排把模型族传给客户端，提交任务把模型族写入 provenance。
- 现有 Qwen2.5、API 后端、报告和任务重试测试保持通过。

开发机测试不下载模型，也不要求 CUDA。服务器验收顺序：

1. 验证依赖可以导入，CUDA 可见。
2. 验证模型目录包含 `config.json` 和完整权重。
3. 用 `genie02-qwen3-vl` 对一个小数据集运行 VLM。
4. 检查 `attempt_eval/attempt_summary.json`、逐 Episode JSON 和证据帧。
5. 检查报告 provenance 显示 `qwen3_vl` 和正确模型路径。
6. 使用代表性长 Episode 观察峰值显存与耗时，再决定是否扩大使用范围。

## 9. 官方接口依据

实现所需的官方资料与版本依据记录在
`docs/research/qwen3-vl-integration.md`。该记录包括 Transformers AutoModel 映射、
Qwen3-VL 预处理差异、确定性生成要求及 RTX 4090 显存边界。
