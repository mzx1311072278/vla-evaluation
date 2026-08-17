# 分析发现

## 文件

- 目标文件存在，大小约 7.9 MB。
- 文件路径暗示评测配置包含：`zqyh_2cm_mixed`、末端执行器 6D 旋转、仅右臂、pi05 stage2、ACP；具体含义须由代码和数据验证。
- 同一数据集目录包含 `meta/info.json`、`meta/stats.json`、`meta/tasks.parquet`、`meta/episodes/chunk-000/file-000.parquet`，以及右腕相机的两个 MP4 文件。
- 视觉帧没有直接存成该目录下的图片文件；主 Parquet 很可能保存视频引用/时间戳，而图像载荷位于 MP4，待 schema 验证。

## 待确认

- 已确认 schema、行数、行组、压缩编码、字段形状、范围、缺失和索引关系。
- 已确认 xyz 单位为米；具体坐标参考系未记录。
- 已确认 rot6d 是两个近似/严格正交单位三向量；按行或按列解码的约定未记录。
- 夹爪数值范围近似 `[-pi/4, 0]`，但单位及开闭方向未由元数据定义。

## 环境

- Codex 捆绑 Python 不包含 `pyarrow`；将检查项目依赖或系统现有环境，避免无必要安装。
- 系统 Python (`/Library/Developer/CommandLineTools/usr/bin/python3`) 同样缺少 `pyarrow`。
- 项目 `requirements.txt` 明确要求 `pandas==2.3.3`、`pyarrow==21.0.0`、`numpy==2.3.5`，可能存在项目虚拟环境。

## 数据集级元数据（`meta/info.json`）

- LeRobot `codebase_version=v3.0`，`robot_type=genie02`。
- 共 60 个 episode、44,397 帧、1 个任务，30 FPS；训练切分为 episode `0:60`。
- 主数据文件按 `data/chunk-{chunk_index}/file-{file_index}.parquet` 组织；视频按特征名单独存 MP4。
- `action` 与 `observation.state` 均为 10 维 float32，名称依次为右末端 xyz、6 个 rot6d 分量、右夹爪位置。
- `complementary_info.policy_action` 也是同名 10 维 float32；需进一步比较它与实际 `action`。
- `complementary_info.is_intervention`、`complementary_info.state` 是标量 float32；`collector_policy_id` 是字符串。
- 右腕视频帧逻辑形状为 `[480,640,3]`，AV1/YUV420p、30 FPS、无音频、非深度图。

## 主 Parquet 初步物理结构

- 文件由 Arrow 24.0.0 写出，Parquet format 2.6；44,397 行、60 个 row group、11 个物理叶子列。
- 每个 row group 对应一个 episode（row group 0 的 `episode_index` 恒为 0，row group 1 恒为 1；全部对应关系待聚合验证）。
- 所有列使用 Snappy 压缩，常见编码是 PLAIN、RLE、RLE_DICTIONARY。
- 主 Parquet 中没有 `observation.images.right_wrist` 列；视频帧不以路径/二进制逐行保存，而是由 episode 元数据映射到外部 MP4。
- 三个 10 维向量在 Arrow 中是 `fixed_size_list<float32>[10]`，Parquet 物理层为 LIST group 下的 FLOAT element。
- schema metadata 含 Hugging Face Features 定义和 fingerprint `4bdcbc1784d2a112`。
- 60 个 row group 与 episode 0..59 严格一一对应，每组 469..2,130 行，平均 739.95 行。
- 所有 11 列实际 null 数都是 0；三个向量每行长度都严格为 10。
- `index` 是 0..44,396 的全局连续行号；`frame_index` 在每个 episode 内从 0 重启且连续；`timestamp` 同样从 0 重启并以约 1/30 秒递增。
- `task_index`、`complementary_info.is_intervention`、`complementary_info.state` 全部恒为 0。
- `collector_policy_id` 全部恒为 `zqyh_2cm_mixed_ee_pi05_stage2_acp`。
- `action` 与 `complementary_info.policy_action` 在全部 44,397 行、10 个维度上完全相同；后一列在本文件中是冗余副本。
- `action` 与 `observation.state` xyz 平均 L2 差 0.01647 m，P95 0.05361 m，最大 0.10451 m。
- `observation.state` 的两个 rot6d 三向量几乎严格单位正交；`action` 的对应向量接近但不完全单位正交。
- episode 成功标签不在主 Parquet 中，而在伴随 episode 元数据里；该元数据为 54 success / 6 failure。

## 代码证据

- `genie02_eval_common.py` 通过 episode 元数据中的 `data/chunk_index`、`data/file_index` 定位主 Parquet，并用 `episode_index` 筛选帧。
- `genie02_episode_metrics.py` 将 `action` 作为轨迹优先来源；EE 平滑度只取名称末尾为 x/y/z 的维度。
- 项目代码可确认 `timestamp` 用秒级差值计算 episode 时长，介入标记以非零表示介入。

## 既有报告证据与限制

- 仓库旧报告将 `||action_xyz-state_xyz||` 的单位明确写为米，因此 EE xyz 可按米解释。
- 旧报告把 `complementary_info.policy_action` 称为“原始 policy action”，把 `action` 视为采集/执行动作；但当前文件直接验证两列逐元素完全相同，因此当前数据不能用于分析二者之间的后处理差异。
- 旧报告声明在线动作反归一化、滤波、限幅与控制接口转换参数未保存在数据集中，因此不能从当前文件恢复这些算法细节。
- `VLA抓取模型评测发版报告.md` 部分段落针对 13 episode/18,015 帧的旧数据，而当前目标是 60 episode/44,397 帧；旧报告数值不能直接作为当前文件统计。
- 仓库现有 `Genie02_report/findings.md` 记录当前 60-episode 样例曾被主报告处理，并确认是单右臂 EE rot6d + 夹爪。

## 存储与伴随映射

- 文件总大小 8,260,319 B；压缩列块合计 8,172,840 B，footer/魔数等开销 87,479 B。
- 列块未压缩合计 8,742,372 B，Snappy 后为 93.49%，仅节省约 6.5%；浮点轨迹本身压缩率有限。
- 三个 10 维向量列占 7,418,174 B 压缩空间；完全重复的 `policy_action` 单列占 2,523,737 B。
- episode 长度 469..2,130 帧，中位 649.5 帧；时长 15.6..70.967 秒，中位 21.617 秒。
- 视频 file-000 对应 episode 0..36，file-001 从 episode 37 开始；每个 episode 的视频时间窗长度严格等于 `length/30`。
- 任务索引 0 映射文本：`Place the medicine in front of your arm into the basket.`

## Web 报告差异反馈环（2026-08-07）

- 参考文档：`Genie02_report/VLA抓取模型评测发版报告.md`。
- 当前 Web 页面：`/reports/d9338238-e7b7-4559-870a-7b33153b9823`。
- 自动检查稳定发现参考文档 33 个二至四级章节在 Web 页中均不存在。
- 参考文档包含 3 行公式/数学定界内容；当前页面没有 MathJax、KaTeX 或等价数学渲染入口。
- 当前 Web 路由主要读取 `metrics_core.json`、`episode_metrics.csv` 和可选 VLM 尝试汇总，因此内容范围天然小于完整发版文档。
- 参考文档不能直接作为当前结果数据源：其中部分统计属于旧的 13 Episode 数据，而当前演示任务是另一份数据。
- 报告内容对齐最终在 `feature/vla-eval-web-vlm-api-backend` 继续；临时 `codex/report-content-parity` 分支未承载功能代码。

## 原始报告与运行时报告的关系

- `VLA抓取模型评测发版报告.md` 不是运行时报告生成器的模板，也不是某个 API 响应；它是一次人工整理的 13 章发版文档。
- 其中动态结果部分来自当时的 `metrics_core.json`、`episode_metrics.csv` 和 LeRobot 数据；模型架构/默认训练配置部分来自外部 `Evo-RL` 代码背景；风险、发版结论和下一步建议是基于当时阈值与人工判断形成的文字。
- 运行时 `genie02_markdown_report.build_report()` 只生成 5 章：评测配置、核心指标、Episode 明细、失败案例、结论。
- 当前 Web 页面与运行时 Markdown 使用相同核心产物，但 Web 路由没有复用 `build_report()` 的结构/公式文本，因此公式和失败案例没有进入页面。
- 运行时 Markdown 的公式当前以 Markdown 文本中的 `$...$` 表示，Web 页面没有 Markdown/数学渲染器；即使把 Markdown 原文直接塞进模板，也不会可靠渲染公式。

## 已确认的真实来源分类

- **当前任务真实结果**：`metrics_core.json`、`episode_metrics.csv`、`episodes.csv`/LeRobot episode metadata、`attempt_eval/attempt_summary.json`。
- **当前数据集真实元数据**：`session.json` 或 LeRobot `meta/info.json`、`meta/tasks.parquet`、feature schema；已有 `load_session()`、`load_episodes()` 负责兼容解析。
- **当前评测真实配置**：EvaluationJob profile/provenance 与版本化 profile YAML；包括 VLM backend/model/prompt/sampling/review 参数。
- **版本化指标定义**：当前公式在 `genie02_markdown_report.build_report()` 和平滑度实现中；应抽成单一报告定义来源供 Markdown 与 Web 共用，不能在模板另写一份。
- **静态背景而非当前实测**：Pi0.5 架构、Evo-RL 默认训练参数。只有在当前 profile/provenance 明确记录时才能显示为当前配置，否则必须标注为“参考背景”或“未记录”。
- **人工/策略判断**：发版建议、风险结论、准入条件。当前系统没有版本化阈值或审批接口，不能照搬旧文档的“建议暂缓生产发版”和旧数值。

## 当前演示任务的真实数据边界

- 任务 `d9338238-e7b7-4559-870a-7b33153b9823` 使用 `genie02-full@1.0.0`，但 `vlm_enabled=false`。
- 数据集为 `demo-mixed-results`、4 Episode、`genie02_session`，预检错误为空。
- 当前输出只有 `episode_metrics.csv`、`metrics_core.json`、`smoothness_curve.svg`、`report_20260806.md`；没有 `attempt_eval` 目录，因此页面不能声称这次做过 VLM 复核。
- Job provenance 已真实记录 VLM 模型路径、Prompt 版本、抽帧参数和复核阈值，但这些是“提交配置快照”，不是“本次已执行 VLM”的证据；展示时必须同时标出 VLM 已关闭。

## 可复用加载接口

- `load_session(dataset_path)`：统一读取 native `session.json` 或合成 LeRobot session；提供任务、FPS、后端、模式、计划 Episode、状态等。
- `load_episodes(dataset_path, session)`：统一读取 native `episodes.csv` 或 LeRobot episode metadata；提供时长、结果、介入、备注。
- `load_episode_metrics(output_dir, session)`：严格校验并类型化逐 Episode 指标。
- `load_metrics_core(output_dir, session)`：严格校验汇总指标、平滑度左右臂统计。
- Job `provenance_json`：配置版本、应用/Git 版本、VLM backend/model/prompt/sampling/review。
- Profile YAML 的 `outputs.optional` 明确 attempt 汇总位于 `attempt_eval/attempt_summary.json|csv`，不是输出根目录；后续下载/来源表必须使用 profile 定义与安全路径解析，不能伪造根目录文件。

## 当前无法由真实接口支持的内容

- 当前模型实际训练超参数、checkpoint 步数、训练日志和训练时间。
- 部署机 PyTorch/CUDA/cuDNN、GPU/工控机、标定与力传感器信息。
- OOD、鲁棒性、碰撞/力控安全测试结果。
- 自动“准许/暂缓发版”结论与准入阈值。系统没有版本化的 release-gate 配置或审批记录。

这些字段可以在“证据缺口”表中显示未记录及所需来源，但不能填入旧报告示例值或默认训练参数冒充当前值。

## 内容优先级初判

1. 首屏：GSR、成功/失败、成功 TTS、平滑度、待复核，以及基于当前数据的简短事实性结论。
2. 紧随其后：评测配置/任务边界、数据来源与质量、指标定义和公式。
3. 中部：Episode 与 VLM 明细、失败案例表。
4. 后部：模型/运行环境/覆盖范围/未记录项/关键文件。
5. 只有存在真实规则和来源时才显示发版结论；否则显示“未配置自动发版判定”，避免伪结论。

## 原始 13 章来源矩阵

| 原报告章节 | 当前真实来源 / 接口 | 分类 | Web 展示决策 |
|---|---|---|---|
| 1. 报告概述 | `EvaluationJob`、`Dataset`；`load_session(dataset.path)` 的 `task`、`rollout_mode` | 动态 | 用当前任务和控制模式生成事实性摘要；不复用旧任务边界文案 |
| 2. 依据与数据来源 | `job.provenance_json`、`dataset.fingerprint`、profile `outputs`、实际存在的输出文件 | 动态 | 来源表列出“数据是什么、来自哪里、是否实际产生”；VLM 配置与执行证据分开 |
| 2.1 核心结论摘要 | `load_metrics_core()`、VLM 汇总、数据质量状态 | 动态，但无发版门禁 | 首屏仅给结果事实与证据完整性；发版状态显示“未配置自动发版判定” |
| 3.1 运行环境 | `provenance_json.app_version/git_sha`、session/LeRobot `codebase_version/robot_type`（若存在） | 部分动态 | 仅展示已记录字段；PyTorch/CUDA/cuDNN 等进入证据缺口 |
| 3.2 配套模块 | profile `adapter/plugin/image_key`、session `dataset_backend/rollout_mode`、实际输出清单 | 动态 | 配置与产物表，不推断策略类型 |
| 3.3 硬件平台 | LeRobot `meta/info.json.features` 或 native session 可确认的相机/动作 schema、FPS | 部分动态 | 展示数据中实际记录的机器人/图像/动作信息；GPU、传感器、工控机等标未记录 |
| 3.4 后处理与安全策略 | 数据 feature schema、episode 介入字段；无在线控制配置 | 部分动态 | 只展示动作/状态/介入字段；在线限幅、滤波、安全阈值标未记录 |
| 4. 模型版本与清单 | `job.provenance_json`、profile、dataset、产物 | 部分动态 | 显示本次评测配置快照与组件；不从 collector 名称推断训练阶段 |
| 5.1 数据集信息 | `Dataset`、`load_session()`、LeRobot `meta/info.json/meta/tasks.parquet` | 动态 | 数据集概览与输入/动作 schema 表 |
| 5.2 数据质量记录 | `Dataset.inspection_json`、`load_episodes()`、`load_episode_metrics()` | 动态 | 总量、成功失败、介入 Episode、短时长/缺失平滑度等可验证质量项 |
| 5.3 数据预处理流程 | `_smoothness_inputs()`、profile VLM sampling、数据元信息 | 部分动态 + 版本化算法 | 展示实际执行的过滤/采样规则；训练数据增强等标未记录 |
| 6. 模型架构信息 | 当前无任务级模型清单或权重元数据接口 | 不可用 | 不把 Evo-RL 背景冒充当前模型；并入证据缺口 |
| 7. 训练方案与超参数 | 当前无训练记录接口 | 不可用 | 并入证据缺口，不展示默认超参 |
| 8.1 指标定义 | 版本化共享指标定义；实现为 `build_core_metrics()`、`_jerk()`、`_smoothness_value()` | 版本化静态定义 | 公式与口径表，公式文本和计算实现共享单一来源 |
| 8.2 分层覆盖 | 当前无版本化测试矩阵或标签接口 | 不可用 | 显示“未记录分层覆盖”，不自动声称部分覆盖 |
| 9. 结果汇总 | `load_metrics_core()`、`load_episode_metrics()`、`load_episodes()`、VLM attempt summary | 动态 | 首屏核心指标，后续 Episode/VLM/失败原因表 |
| 9.2 额外诊断 | 当前产物仅支持介入、平滑度有效帧/跳过原因；无 action-state 聚合产物 | 部分动态 | 只展示现有产物可验证的诊断；不在请求期重算新指标 |
| 10. 风险分析 | 当前无风险规则引擎；可陈述数据证据缺口 | 部分动态 | 改为“证据与限制”表，不生成主观风险评级 |
| 11. 发版结论 | 当前无 release-gate 配置或审批记录 | 不可用 | 固定显示“未配置自动发版判定”，不得给准许/暂缓结论 |
| 12. 下一步建议 | 当前无任务/审批接口 | 不可用 | 不生成假建议；由证据缺口明确需要补充的数据来源 |
| 13.1 关键文件 | profile `outputs.required/optional` + 实际文件存在性 | 动态 | 保留安全下载；修正 `attempt_eval/` 嵌套路径 |
| 13.2 评测公式 | 共享指标定义 | 版本化静态定义 | 使用原生 HTML 数学结构（上标/下标/分数）可靠渲染，无外部 CDN 依赖 |
| 13.3 未记录项 | 来源矩阵中不可用字段 | 动态状态 + 静态需求定义 | 集中表格显示缺失证据、影响、应由什么接口补齐 |

## 报告内容实现边界

- Web 报告继续只读已持久化评测产物，不在打开页面时重算 GSR、TTS、平滑度或 action-state 诊断。
- `load_session()` / `load_episodes()` 读取数据集，`load_episode_metrics()` / `load_metrics_core()` 读取输出目录；Web 视图模型负责跨来源组合和状态标注。
- 共享指标定义应是结构化、版本化数据，同时供 Markdown 生成器和 Web 使用；公式不再散落在模板字符串中。
- 下载清单由 profile 输出定义驱动，并以白名单与路径包含校验保护；支持 `attempt_eval/attempt_summary.*` 的真实嵌套路径。
- 公式优先使用无 JavaScript 依赖的语义化 HTML（`<var>`、`<sub>`、`<sup>`、分数布局），避免离线环境下 MathJax/KaTeX CDN 失效。
