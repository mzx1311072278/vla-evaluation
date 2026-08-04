# VLA 模型真机评测：既有惯例调研与系统框架

调研日期：2026-08-04

## 一、结论先行

VLA 真机评测目前没有一个跨机器人、跨任务统一且被普遍采用的 benchmark。代表性工作大多使用自有机器人、场景和成功判据，以任务成功率为主，并按未见物体、位置、背景、环境、指令或组合变化做泛化测试。单任务常见 5--20 次 rollout；RT-1、RT-2 通过大量任务累计到 3,000+、约 6,000 次，但公开材料通常仍缺逐 trial manifest、统一安全指标、异常剔除和人工判定细则。

因此，企业内部评测系统不能简单照抄某篇论文的成功率表。应该同时解决四件事：

1. **测试覆盖**：ID、单轴 OOD、组合 OOD、扰动恢复、长程任务和可选跨机器人。
2. **测量口径**：完整成功、部分进度、效率、轨迹质量、安全、系统实时性和指令遵循分别统计。
3. **试验可信度**：冻结条件、配对初态、随机/交错执行、明确分母、置信区间和失败分类。
4. **工程可追溯**：模型、代码、权重、容器、硬件、标定、安全配置、场景、视频、事件和标注全部可回溯到不可变 digest。

推荐建设“模块化评测流水线”，复用现有 Genie02 的 LeRobot/NPZ 读取、GSR/TTS/平滑度和 Markdown 报告能力，但把它们降为数据适配器和指标插件；在外层新增 campaign、registry、trial/event/artifact/annotation schema、在线 watchdog、统计比较和发版门禁。

## 二、既往惯例在测什么

| 类别 | 代表做法 | 对内部评测的启示 |
|---|---|---|
| 分布内能力 | RT-1 的 seen instructions、OpenVLA 的各平台任务 | ID 初态仍应随机变化，不能把完全固定摆放称作真实性能 |
| 新组合/语言 | RT-1 unseen instruction；RT-2 emergent semantics；OpenVLA language grounding | 语言选对对象与机械动作成功应分报 |
| 物体/空间/背景 | RT-2 easy/hard OOD；OpenVLA visual/motion/physical/semantic 分栏 | 先做单轴变化，再做组合变化，才能解释失败原因 |
| 跨环境 | RT-1 新厨房；pi0.5 全新家庭 | 新场地更接近业务价值，但须保留对应单轴对照 |
| 扰动与恢复 | RT-1 distractors/occlusion 等有限 robustness 测试 | 企业评测应分别记录扰动层 Full SR、恢复状态率、恢复时间、重试和安全事件 |
| 长程任务 | RT-1 分规划/执行；pi0/pi0.5 用阶段 rubric 和完整成功 | 不能只报 progress；完整自主成功必须独立保留 |
| 跨 embodiment | RT-X、Octo、pi0 系列 | zero-shot 必须声明目标机器人/任务数据是否进入训练或微调 |
| 实时部署 | RT-2 报告不同模型频率；OpenVLA 涉及 5/15 Hz 控制；pi0.5 端到端高频控制 | 模型比较必须冻结频率、动作 chunk、滤波、安全限幅和低层控制器 |

详细证据、每项论文的机器人/任务/rollout/判据/基线/局限见[既有惯例调研](research/01_real_world_precedents.md)。

## 三、论文惯例中最值得复用与最需要补齐的部分

### 值得复用

- RT-1/RT-2/OpenVLA 将泛化拆成物体、空间、背景、环境和语义维度。
- RT-2、OpenVLA、DROID 使用相同或匹配初态做 A/B；pi0.5 交错执行不同策略，降低时间漂移。
- OpenVLA、pi0/pi0.5 对困难/长程任务预定义部分进度 rubric，同时保留完整成功。
- Octo/OpenVLA 把 out-of-box、target-data 微调和 scratch 基线分开比较。
- DROID 在多个地点使用同一硬件栈、ID/OOD 成对条件和逐任务成功定义，适合内部数据/模型贡献消融。

### 必须补齐

- 论文常见每任务 5--20 次，只能筛选大差异，不能证明高可靠。
- 普遍缺少碰撞、近碰、力/速度/工作空间超限、E-stop、接管原因和 safe-stop 延迟。
- 普遍缺少策略输出、后处理命令、控制器接受命令与机器人状态之间的完整动作链。
- 很多论文没有公开异常取消、超时、救援和人工判定协议，成功率分母不可完全复核。
- 汇总成功率容易被简单高频任务主导，也可能掩盖某个关键任务或扰动层完全失败。

## 四、推荐评测框架

### 4.1 数据层级

```text
Campaign（冻结的一次比较/发版评测）
└── Scenario（任务、场景、扰动与风险层）
    └── Session（同一硬件/标定/安全配置下的连续批次）
        └── Trial（独立复位的一次统计试验）
            └── Episode（策略实际控制时序）
                ├── Time series
                ├── Events
                ├── Artifacts
                └── Annotations
```

统计独立单元是 trial，不是帧。trial 初态/setup 检查失败可以有 trial 而没有 episode；任何重试都创建新 trial，并关联原 trial，不能覆盖或删除失败证据。Session Preflight 失败则在创建 trial 队列前中止 session。

### 4.2 测试层级

每个 session 先执行 **Preflight 数据与安全预检**：标定、时钟、方向、限位、watchdog、E-stop、写盘和版本清单。Preflight 属于 SOP，不占测试层级编号。

- **L0 冒烟**：少量简单已见任务，只用于阻断部署错误。
- **L1 ID 基线**：业务核心任务，初态按冻结清单随机，候选与基线配对。
- **L2a--L2d 单轴泛化、L2e 组合 OOD**：分别改变空间、物体、背景/场景和指令，再组合两个及以上因素。
- **L3 扰动恢复**：预注册扰动幅度/时刻，以分配到扰动层的全部 valid trials 计算 canonical Full SR，并在扰动实际施加且此前未失败的 eligible 子集中测恢复状态率和恢复时间。
- **L4 长程任务**：严格完整成功、分阶段进度、失败阶段和自主高层规划分开。
- **L5 跨机器人/场地**：只在模型作出此类能力声明时启用。

完整测试单元、建议重复数和工程通过条件见[分层测试矩阵](framework/01_test_matrix.md)。建议次数是工程起点，不是论文共识；正式样本量应由目标成功率、允许误差、比较设计与风险等级决定。

### 4.3 指标层级

P0 必选指标至少包括：

- `Full SR = 无人工帮助且时限内完整成功 / 有效启动 trial`，报告 `k/n` 与区间。
- trial 有效性、任务结果和安全结果三列正交；接管、安全停机、超时不得伪装成无效试验。
- 逐任务/逐扰动的成功率、宏平均，以及候选相对冻结基线的配对差。
- 成功 TTS，同时保留所有 trial 的行政截尾、吸收性失败和竞争终止口径，避免只看快且成功的幸存样本。
- 碰撞、近碰、限位/力/速度/工作空间超限、E-stop、接管和 safe-stop 结果。
- 端到端时延 p50/p95/p99、deadline miss、观察/动作陈旧度和控制频率。
- 数据完整率、掉帧/序列缺口、时钟偏差、必需 artifact 与版本 digest 完整率。

P1 诊断指标包括子目标/任务进度、失败阶段、路径/动作长度、末端误差、速度/加速度/jerk、抖振、抓取滑移、恢复状态率/恢复时间、语言选对率、资源和能耗。详细定义、公式、传感器来源、分母和陷阱见[指标与统计框架](framework/02_metrics_and_statistics.md)。

### 4.4 统计与标注

- 执行前冻结主要指标、分母、无效规则、超时、停止规则、分层和统计方法。
- 候选/基线共享初态 ID，按随机区组或 ABBA 交错，避免光照、设备热态和操作员学习效应。
- 二元指标报告 Wilson、Jeffreys 或精确二项区间；低安全事件率使用单侧上界，零次观测不等于真实风险为零。
- 优先做配对效果差及其区间；不只比较两个独立百分比或只看 p 值。
- 连续指标明确是全 trial、成功 trial 还是截尾估计；仅成功样本统计时必须并列成功率。
- 逐任务/扰动分层判定；宏平均用于总体观察，micro 结果只做诊断。
- 发版关键视频尽量盲化模型名；模糊样本双人标注，分歧由第三人裁决，所有修订追加记录。

### 4.5 系统与门禁

系统采用 registry、adapter、orchestrator、collector、独立 watchdog、annotation、metric、comparison、gate 和 reporting 模块。原始 `policy_action`、安全/后处理后的 `command_action`、controller ack 与 `state` 必须分别记录，用 request/command ID 和单调时钟关联。

发版顺序采用非补偿门禁：

1. 数据与身份完整性门禁。
2. 安全硬门禁。
3. 关键任务绝对底线门禁。
4. 候选相对基线的统计非劣/改善门禁。
5. 效率、轨迹质量和资源等观察项。

任何 mandatory task x perturbation 层信息不足时判 `INCONCLUSIVE`，不能因为总体均值好看而判通过。系统 schema、状态机、SOP 和完整门禁见[系统与发版框架](framework/03_system_and_release.md)。

## 五、Genie02 现有能力与差距

### 可直接复用

- LeRobot Parquet/MP4、`meta/info/tasks/episodes` 和 Native NPZ 读取。
- `observation.state`、`action`、`policy_action`、介入标记和 collector policy 字段。
- session/episode 一致性校验、轨迹选择、介入帧过滤。
- GSR、成功 TTS、jerk 平滑度、逐 episode CSV、JSON 汇总和 Markdown/SVG 报告。

### 第一优先扩展

- `campaign/scenario/trial/episode/event/artifact/annotation` sidecar schema。
- 模型权重、代码、容器、硬件、固件、标定和安全配置 digest。
- `validity`、`autonomy_outcome`、`safety_outcome`、子目标和失败阶段。
- 独立 watchdog、安全事件、策略动作到实际命令的时延链。
- artifact checksum/session seal、人工复核 revision、分层区间和 gate JSON。

### 历史数据无法补回

旧数据未记录的实际 checkpoint digest、执行环境、硬件序列号、标定、安全参数、实际下发命令、时延、场景随机化、安全事件和标注者历史不能可靠反推，只能标记 `unknown` 并补测。已有材料中还出现 13 episode 与 60 episode 两个快照，但没有数据 manifest digest，无法仅凭同名目录证明对应版本。

## 六、建议的首轮 Genie02 真机评测

首轮目标不是一次建成最终排行榜，而是验证 schema、SOP、指标和门禁能稳定运行。

建议选择 8 个任务：3 个基础抓放、1 个关节物体、1 个精细操作、1 个语言 grounding、1 个扰动恢复、1 个 5 步以上长程任务。

执行顺序：

1. 每个 session 先完成 Preflight；随后 L0 每任务 3 次，验证坐标、动作、相机和日志，不对外报成功率。
2. 候选与当前基线共享每个条件的 20 个初态 ID，采用随机区组/ABBA 交错执行。
3. 每个核心任务做 ID、一个空间 OOD、一个物体 OOD；选 2 个任务补背景/语言与组合 OOD。
4. 扰动任务预注册两档幅度和早/中/晚触发；长程任务记录严格终态与每个子目标。
5. 第一轮只筛选大回归。30--50 次/关键 cell 仅是正式评测预算起点；发布前按 SLO、非劣 margin、基线率和配对 discordance 反算样本量，预算不足则判 `INCONCLUSIVE`。
6. 严重安全事件、保护失效或关键日志缺失直接阻断；其余层按 `PASS/BLOCK/INCONCLUSIVE` 给证据。

## 七、落地路线

| 阶段 | 交付 | 建议顺序 |
|---|---|---|
| P0 数据契约 | frozen plan/registry、trial/event/artifact/annotation schema、manifest/seal | 最先完成 |
| P1 兼容导入 | 把现有 Genie02 读取、GSR/TTS/平滑度接成 adapter/plugin，保证旧结果可复算 | 与 P0 并行验证 |
| P2 在线采集 | request/action/command/state 时延链、watchdog、安全事件、数据质量 validator | 正式扩大测试前 |
| P3 统计与门禁 | 分层区间、配对比较、宏平均、hard/statistical gate JSON | 首轮 schema 验证后 |
| P4 运营平台 | UI、调度、权限、审批、趋势与对象存储治理 | 经过数轮发版后 |

不建议第一步就做完整 Web 平台。先用 CLI/配置文件跑通冻结计划、采集、seal、标注、指标和门禁，能更快暴露口径问题，也能避免把错误 schema 固化到复杂系统中。

## 八、交付文件

- [总览](framework/00_overview.md)
- [代表性真机评测惯例](research/01_real_world_precedents.md)
- [分层测试矩阵](framework/01_test_matrix.md)
- [指标、统计与安全调研](research/02_metrics_statistics_safety.md)
- [指标与统计框架](framework/02_metrics_and_statistics.md)
- [系统工程实践调研](research/03_system_practice.md)
- [系统、SOP 与发版门禁](framework/03_system_and_release.md)

所有下载论文、PDF 转文本、网页快照和解析中间物均位于 `tmp/`，最终报告不依赖临时文件路径。
