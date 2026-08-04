# VLA 真机评测系统工程实践调研

## 1. 调研范围与结论

本稿聚焦与具体模型、机器人解耦的评测工程：数据契约、资产谱系、在线采集与安全监督、标注复核、指标/报告流水线、模型比较、回归门禁和数据保留。它不定义跨任务通用的成功率或安全阈值，也不替代机器人本体的功能安全认证。

推荐采用“模块化流水线 + 内容寻址制品库”的方式：LeRobot v3 负责高频状态/动作与多相机视频的通用落盘；评测系统在其外层增加冻结计划、registry 快照、trial/event/annotation、全栈版本清单和门禁决策。MLflow 或 W&B 可作为索引、比较和谱系后端，但不能取代原始文件的 schema、校验和与保留策略。

最重要的工程约束如下：

1. 统计独立单元是经独立复位的一次 `trial`，不是帧；`episode` 是该 trial 中真正执行策略的时序段。trial 初态/setup 检查失败可以有 trial 而没有 episode；Session Preflight 失败则在创建 trial 队列前中止 session。重试必须新建 trial。
2. `validity` 与 `autonomy_outcome` 正交。碰撞、接管、超时或安全停机不能伪装成“无效试验”后删除。
3. 计划、任务/场景、模型/权重、代码、容器、硬件、标定和安全配置必须在执行前冻结到 revision + digest；路径和可变 alias 只可作为辅助信息。
4. 原始策略输出 `policy_action`、安全/后处理后的 `command_action`、机器人反馈 `state` 必须分别记录，并通过 request/command ID 和单调时钟关联。
5. 安全 watchdog 独立于策略进程，失联、过期动作、越界或安全传感器事件必须先安全停止，再异步写日志。
6. 原始证据、标注修订和门禁决策追加写入；纠错使用 `supersedes`，不覆盖历史。
7. 发版先过 hard gate，再做 statistical gate；按任务和扰动单元逐层判定，汇总分不能掩盖关键分层失败。

## 2. 官方资料对架构的直接启示

| 资料 | 可核验事实 | 对本框架的直接采用 |
|---|---|---|
| LeRobotDataset v3 官方文档 | 低维高频信号使用 Parquet，多相机视觉使用 MP4，`meta/info.json` 记录 schema/FPS/path template，`meta/episodes` 记录 episode 长度、任务与共享文件偏移；存储边界与 episode API 解耦 | 保留 LeRobot 作为帧/视频层；trial、事件、安全、标注、版本清单放在兼容 sidecar 中；禁止依赖文件名猜 episode 边界 |
| Google Research RLDS 官方仓库 | 数据集由 episode 和 step 组成；建议 episode ID、agent ID、environment config、experiment ID、invalid 等元数据；step 明确 `is_first`、`is_last`、`is_terminal`，截断与终止有不同语义 | 使用全局唯一 episode ID；显式记录运行主体、环境快照和有效性；区分正常结束、任务终止和截断/安全停止 |
| MLflow Tracking 官方文档 | run 可记录参数、代码版本、指标和输出制品；支持把指标关联到具体 model checkpoint 与 dataset | 一个冻结 campaign/session 映射为可检索 run；每个指标保留 policy bundle、dataset/session digest 和 metric implementation revision |
| MLflow Dataset Tracking 官方文档 | Dataset 对象包含 name、digest、source、schema、profile，支持只记录远端数据的 metadata/lineage | 大视频保留在对象存储，只在追踪系统记录带 digest 的 source；不能只留一个可变目录路径 |
| MLflow Model Registry 官方文档 | registry 提供模型来源 run、版本、alias、tag 和 annotation；alias 是可变引用 | 执行时解析 alias，冻结为不可变 model version + artifact digest；发布后 alias 只用于指向已经通过门禁的版本 |
| W&B Artifacts 官方文档 | artifact 可作为 run 的输入/输出并版本化数据集、模型和评测结果 | 无论选择 MLflow 还是 W&B，都用输入/输出制品图表达 `policy + plan + registry snapshot -> raw session -> annotation -> metrics -> report` |
| ROS 2 Managed Nodes 生命周期设计 | 主状态包括 Unconfigured、Inactive、Active、Finalized；外部 supervisory process 驱动转换，Active 中不可处理错误进入 ErrorProcessing | runner 与 collector 使用显式生命周期；准备/校准期间不允许控制输出；watchdog/上位监督而不是策略自行决定故障恢复 |
| NIST Engineering Statistics Handbook 的比例区间章节 | 推荐用 Wilson 类比例区间；小样本或极少失败时正态近似可能不准确，可用精确二项区间；也给出单侧区间 | 成功率与安全事件率不只报点估计；门禁使用预先冻结的单侧边界，小样本使用精确方法或保守方法，样本不足判 `INCONCLUSIVE` |

这些资料共同支持“原始机器人数据标准化、评测语义另建可审计控制平面”的分层，而不支持把完整发版依据压缩成一个 LeRobot 目录或一个实验追踪 run。

## 3. 三种落地路径

| 路径 | 优点 | 风险 | 结论 |
|---|---|---|---|
| 直接扩展现有 Genie02 CSV/Markdown | 近期成本最低，可立即复用 GSR/TTS/平滑度 | 单任务枚举、无事件流、无版本快照，越扩展越难维护 | 只适合作为兼容导入器 |
| 模块化评测流水线 | registry、采集、标注、指标、门禁边界清晰；机器人/模型通过 adapter 接入 | 需要先稳定 schema 和 SOP | 推荐第一阶段实施 |
| 一次建设完整在线平台 | 调度、看板、权限、审批一体化 | 在口径未稳定前固化错误，建设和验证成本高 | 待模块化流水线经过数轮发版后再做 |

## 4. 数据层实践

### 4.1 控制面与数据面分离

- 控制面：评测计划、registry、分层配额、随机种子、状态机、操作者操作、标注任务、门禁决策。
- 数据面：视频、机器人状态、观测、策略动作、实际下发动作、时延、事件和其他原始日志。
- 索引面：MLflow/W&B 或数据库中的 session/trial/metric/artifact 索引。索引丢失时应能从冻结 manifest 重建，原始数据不能反向依赖某个 SaaS 对象 ID 才可解释。

### 4.2 时间与动作链路

每个高频记录至少包含 `seq`、`t_monotonic_ns`、可选 `t_utc`、`clock_id` 和 source timestamp。端到端链路需区分：观察可用、请求排队、推理开始/结束、策略输出、后处理/安全过滤、命令发送、控制器接受、状态反馈。统一记录一个 `policy_request_id` 和一个或多个 `command_id`，才能计算观察陈旧度、推理耗时、排队耗时、动作下发延迟和闭环响应，而不是用单一 `timestamp` 猜测。

### 4.3 事件与制品

事件采用追加式 JSONL/Parquet，安全事件、watchdog 状态、接管、trial 边界、复位、标定检查和数据质量异常共用 envelope。大文件采用 artifact manifest：逻辑类型、URI、SHA-256、字节数、媒体类型、时间范围、生产者版本、保留等级。报告引用 artifact ID/digest，不引用容易漂移的“最新目录”。

### 4.4 标注与复核

自动判定/VLM 只能生成 `proposed` annotation。成功、失败阶段、安全严重度等发版关键标签按风险采用人工复核或双人盲审；分歧生成第三方 `adjudicated` 修订。所有标注保留 rubric revision、证据时间段、annotator 类型/匿名 ID、时间和 supersedes 链。

## 5. Genie02 现有能力审计

审计对象包括 [Genie02 README](../../Genie02_report/README.md)、[VLA 抓取模型评测发版报告](../../Genie02_report/VLA抓取模型评测发版报告.md)、四个主报告脚本以及 [`attempt_eval/`](../../Genie02_report/attempt_eval/)。结论按“可直接复用、需扩展、当前数据无法补回”分类。

### 5.1 可直接复用

| 现有资产 | 已验证能力 | 建议接入位置 |
|---|---|---|
| LeRobot `meta/info.json`、`meta/tasks`、`meta/episodes`、Parquet/MP4 | schema、FPS、robot type、任务映射、episode 边界、状态/动作/视频 | `adapters/lerobot` 和 raw session 数据面 |
| `action`、`observation.state` | 10 维右臂 EE rot6d + 夹爪时序 | timeseries 标准列的机器人 adapter 映射 |
| `complementary_info.policy_action` | 保留原始策略动作 | 映射为 `policy_action`，但不能替代实际下发动作 |
| `is_intervention`、`collector_policy_id` | 帧级介入标志和策略来源字符串 | 兼容导入；后续补事件原因、操作者和策略 bundle ID |
| `genie02_eval_common.py` | session/episode 字段、枚举、时长、唯一 index 和一对一 join 校验；支持 LeRobot session 合成 | 兼容验证器，逐步迁移到通用 schema validator |
| `genie02_episode_metrics.py` | 读取 NPZ/Parquet；按字段优先级取轨迹；过滤介入帧；计算综合/左右臂 jerk 平滑度并保留跳过原因 | 指标插件 `smoothness_jerk_v1`，保留实现 revision |
| `genie02_metrics_core.py` | GSR、成功 TTS、平滑度统计和跨文件一致性检查 | 基础指标插件；补分层、区间和比较 |
| `genie02_markdown_report.py` | 配置摘要、核心指标、episode 明细、失败表和 SVG | 报告模板的兼容版本 |
| `attempt_eval/` | episode 视频映射、全局/密集抽帧、VLM JSON 校验、低置信/遮挡/短片等复核提示、逐 episode JSON/CSV | 自动预标注器；结果必须进入通用 annotation/review 流 |

### 5.2 需扩展

| 缺口 | 当前表现 | 必需扩展 |
|---|---|---|
| 计划冻结 | `session.json` 只引用可变 config path，原始 LeRobot 还会用目录 mtime 合成创建时间 | campaign/plan revision、canonical digest、审批、分层配额、随机种子和停止规则 |
| Registry | 任务是自由文本，robot type/collector policy 是字符串 | task/scene/object/perturbation/robot/sensor/policy/calibration/safety 实体及冻结快照 |
| Trial/episode 语义 | 仅 `episode_index`，outcome 只有 success/failure | 全局 ID、trial 与 episode 关系、validity、partial/timeout/safety_stop/takeover、失败阶段 |
| 全栈谱系 | 有 `codebase_version` 和背景 Evo-RL commit，但无执行模型清单 | 权重 digest、训练 checkpoint、执行代码 commit/dirty hash、容器 digest、依赖 lock、硬件序列号/固件、标定和安全配置 digest |
| 在线事件/安全 | 无统一 event，介入只有帧标志 | watchdog heartbeat、限位/碰撞/力/急停/接管/恢复事件、严重度、触发值和 safe-stop 结果 |
| 时延与动作链 | 单一 frame timestamp；有 policy action 和 action，但语义不足以恢复所有在线阶段 | 多时钟同步、request/command ID、各阶段时间戳、postprocess/filter/clamp、actual command/controller ack |
| 数据质量 | 可检查文件字段和轨迹有效帧 | artifact checksum、视频掉帧/解码、序号缺口、时钟漂移、topic freshness、跨流边界和 session seal |
| 标注审计 | `notes` 是自由文本；VLM 输出有 review flag 但无审阅者/修订链 | rubric、evidence span、annotator、复核/裁决、supersedes、冻结 annotation set digest |
| 指标统计 | 只有点估计、均值/std/min/max；未按任务/扰动分层 | k/n + 区间、截尾口径、分层宏/微汇总、候选-基线效应量、配对比较、功效/样本不足判定 |
| 发版 | 报告人工写建议，无机器可判门禁记录 | hard/statistical gate 配置、逐层证据、PASS/BLOCK/INCONCLUSIVE、批准/例外审计 |
| 保留与访问 | 本地目录，无 manifest/生命周期 | 内容寻址对象存储、冷热分层、访问控制、legal hold、删除墓碑和可重建索引 |

### 5.3 当前数据无法可靠补回

以下字段若采集时未记录，不能从现有 Parquet/MP4 或文件名可靠推断。后续只能标记 `unknown` 并补测，不能用当前默认值冒充历史事实：

- 当时实际加载的模型权重/checkpoint digest、完整模型配置、训练数据/训练 run 与 checkpoint 选择依据。
- 执行代码的准确 commit/dirty state、容器镜像 digest、PyTorch/CUDA/cuDNN/驱动和部署参数快照。
- 机器人、夹爪、相机、力传感器和工控机的序列号、固件、健康状态；相机内外参与手眼标定 revision/误差。
- 在线限位、动作缩放、滤波、碰撞/力阈值、watchdog 超时、急停状态和保护逻辑版本。
- 实际下发控制器的 post-safety command、controller ack、策略各阶段时延、统一时钟偏差和掉帧/丢包原因。
- 每次 trial 的物体实例、初始位姿、场景布置、扰动等级、随机种子、复位验收和操作者身份。
- 碰撞、近碰、力超限、急停、接管原因与恢复过程的结构化事件。
- 原始成功标签的 annotator、rubric revision、证据片段、复核与裁决历史。

现有历史报告之间还出现“13 episodes”和当前目录中“60 episodes”的不同快照。二者可能来自不同数据阶段，但由于报告未绑定数据集 manifest digest，不能仅凭同名路径证明是哪一版；这正是不可变数据集指纹和报告输入 manifest 必须补齐的原因。

## 6. 采用边界

- LeRobot 的目标是机器人学习数据存储，不原生承担发版审批、在线安全事件和全栈配置冻结。
- MLflow/W&B 的版本和 lineage 能力适合索引，但 tracker 中的 tag/alias 可能可变；门禁证据仍需冻结 manifest 和 digest。
- ROS 生命周期是节点监督设计依据，不等于安全认证；真正的安全停止链路需要独立风险分析和机器人厂商约束。
- VLM 抽帧可降低人工浏览成本，但抽帧会遗漏短时事件，不能单独裁决碰撞、接触力或所有失败尝试。
- 小样本比例需要区间；区间方法本身不能弥补场景覆盖不足、非独立重复或事后修改分层。

## 7. 来源

以下链接均访问于 2026-08-04：

1. Hugging Face, *LeRobotDataset v3.0*: https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3
2. Google Research, *RLDS Dataset Format*: https://github.com/google-research/rlds
3. MLflow, *MLflow Tracking*: https://mlflow.org/docs/latest/ml/tracking/
4. MLflow, *Dataset Tracking*: https://mlflow.org/docs/latest/ml/dataset/
5. MLflow, *Model Registry*: https://mlflow.org/docs/latest/ml/model-registry/
6. Weights & Biases, *Artifacts overview*: https://docs.wandb.ai/models/artifacts
7. ROS 2 Design, *Managed nodes*: https://design.ros2.org/articles/node_lifecycle.html
8. NIST/SEMATECH, *Confidence intervals for proportions*: https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm
