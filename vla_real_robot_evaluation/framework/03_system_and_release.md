# VLA 真机评测工程系统与发版门禁

## 1. 目标与设计约束

本模块把模型、机器人、任务和存储后端隔离在 adapter 后面，规定一条可审计的真机评测链路。上层只依赖统一的 registry ID、trial/event/artifact/annotation schema 和指标接口。

必须满足：

- 执行前冻结评测计划、registry 快照、模型/代码/权重/容器、硬件/固件、标定和安全配置；任何变更产生新 session。
- trial 有效性、任务结果和安全结果分列。接管、安全停机、超时均是模型表现，除非事前规则明确为非模型外因，否则不得排除分母。
- 原始数据追加写；seal 后以 digest 校验。标注和门禁纠错采用新 revision，不覆盖历史。
- watchdog 独立于策略进程且 fail closed。日志失败不得阻碍急停，急停也必须生成尽力而为的本地事件记录。
- 发版门禁分 hard gate 与 statistical gate，并在每个 mandatory task/perturbation stratum 上判定；全局均值不能补偿关键层失败。
- 所有数值阈值、置信水平、非劣界、严重度和最小样本规则由风险等级、生产基线和样本预算决定，写入冻结计划；本框架不提供统一阈值。

## 2. 推荐目录结构

```text
vla_eval/
├── configs/
│   ├── plans/                    # 评测计划模板，执行时生成冻结快照
│   ├── metrics/                  # 指标定义、分母与实现 revision
│   └── gates/                    # 风险分级、hard/statistical gate 模板
├── registries/
│   ├── assets/                   # robot/end-effector/sensor/compute
│   ├── tasks/                    # 目标、成功 rubric、阶段、超时
│   ├── scenes/                   # 场地、物体、初始条件、复位方法
│   ├── perturbations/            # 扰动类型、等级、施加/验收方法
│   ├── policies/                 # model/config/weight/runtime bundle
│   ├── calibrations/             # 标定产物、方法、误差与有效期
│   └── safety_profiles/          # 限位、watchdog、停止与恢复策略
├── adapters/
│   ├── robots/                   # 统一 state/command/health/safe_stop
│   ├── policies/                 # 统一 infer(request)->policy_action
│   ├── sensors/                  # 图像/力/外部追踪等
│   └── datasets/                 # LeRobot/native/RLDS 导入导出
├── runtime/
│   ├── orchestrator/             # session/trial 状态机、配额与随机化
│   ├── collector/                # 时序、视频、事件和 manifest
│   ├── watchdog/                 # 独立心跳、边界检查、安全停止
│   └── clock_sync/               # 时钟域、偏差与漂移测量
├── annotation/
│   ├── rubric/                   # 版本化判定说明
│   ├── prelabelers/              # 规则/VLM，仅生成 proposed
│   ├── review/                   # 双人复核与裁决
│   └── exports/                  # 冻结 annotation set
├── analytics/
│   ├── validators/               # schema/完整性/时间/校验和
│   ├── metrics/                  # 插件，每项有定义和 revision
│   ├── comparison/               # 分层、配对、区间与回归
│   └── gates/                    # 门禁求值与证据明细
├── reporting/                    # 报告、图表、release dossier
├── schemas/                      # JSON Schema/Arrow schema/枚举
└── tests/                        # 契约、回放、故障注入和黄金数据

store/
├── sessions/<session_id>/
│   ├── frozen/                   # plan + registry + all manifests
│   ├── trials/                   # trial/episode/event sidecars
│   ├── timeseries/               # Parquet，按 stream/时间分片
│   ├── videos/                   # MP4，按相机分片
│   ├── artifacts/manifest.jsonl  # 所有制品及 digest
│   ├── annotations/              # proposed/reviewed/adjudicated
│   ├── metrics/                  # 带实现 revision 的派生结果
│   ├── reports/                  # 人读与机读报告
│   └── seal.json                 # 根 manifest digest、签署与时间
└── releases/<release_id>/        # 候选、基线、门禁输入/输出与审批
```

目录是逻辑布局，可映射到对象存储。数据库/MLflow/W&B 保存查询索引和 lineage；`store/` 中的 manifest 是可迁移的事实源。

## 3. 模块职责与接口

| 模块 | 输入 | 输出 | 关键责任 |
|---|---|---|---|
| Plan compiler | plan 模板、risk tier、registry revisions | canonical `plan.snapshot.json` + digest | 展开默认值、校验配额/分层/随机化/停止规则，冻结门禁策略 |
| Registry | 版本化实体 | immutable snapshot | 资产、任务、场景、扰动、策略、标定、安全配置的 ID/revision/digest |
| Policy adapter | 标准观察、policy bundle | `policy_action`、推理时间、request ID、模型诊断 | 隔离 Pi/RT/自研策略差异，不接触硬件急停 |
| Robot adapter | `command_action` | state、ack、health、safe_stop | 坐标/单位显式转换；提供幂等 safe-stop |
| Orchestrator | 冻结计划、随机化队列 | session/trial 状态与事件 | 配额、顺序、重试新建 trial、暂停/终止；不直接算指标 |
| Collector | adapters 的流和事件 | Parquet/MP4/JSONL、artifact manifest | 序号、时钟域、校验和、落盘状态和掉帧统计 |
| Watchdog | 心跳、state、command、safety profile | 安全事件、safe-stop | 独立进程/节点；策略卡死或 collector 故障时仍可停止 |
| Validator | sealed session | validation result | schema、ID 引用、校验和、时序、视频、trial 边界和必需证据 |
| Annotation service | rubric、视频/时序/事件 | revisioned annotations | 预标注、盲审、复核、裁决和 annotation set freeze |
| Metric engine | valid trials、frozen annotations、metric specs | 分层 trial/aggregate metrics | 分母、截尾、缺失值和实现 revision 可审计 |
| Comparator | candidate/baseline metrics + matched keys | effect/interval/regression | 配对优先、分层、period/operator 敏感性分析 |
| Gate engine | frozen gate policy + evidence | PASS/BLOCK/INCONCLUSIVE + reasons | hard gate 先行；逐 stratum 判定；不修改阈值 |
| Reporter | 全部 manifest、结果和 gate decision | HTML/Markdown/JSON release dossier | 每个数字回链到 query、metric revision 和输入 digest |

## 4. 端到端数据流

```text
Registry revisions + risk assessment + baseline release
                         |
                         v
             Plan compile / review / freeze
                         |
                         v
      Resolve aliases -> immutable policy/runtime digests
                         |
                         v
  Preflight -> calibration check -> watchdog proof -> READY
                         |
                         v
 Randomized trial queue -> setup evidence -> policy episode
                         |                         |
                         |             policy_action -> safety/postprocess
                         |                         |
                         |             command_action -> robot state/ack
                         v                         v
          events.jsonl + timeseries.parquet + camera.mp4
                         |
                         v
               reset evidence / next trial
                         |
                         v
    finalize -> artifact checksums -> manifest root -> seal
                         |
                         v
 schema/time/completeness validation -> prelabel -> human review
                         |
                         v
 freeze annotation set -> metrics -> baseline comparison
                         |
                         v
 hard gate -> statistical gate -> release dossier -> archive
```

失败路径也必须落盘：Session Preflight 失败进入 session 状态且不创建 trial 队列；trial setup 失败、执行中止、watchdog stop、collector 不完整分别成为 event/trial/session 状态。只有完成 seal 和 validation 的 session 可进入统计。

## 5. 状态机

### 5.1 Session 状态

```text
DRAFT -> FROZEN -> PREFLIGHT -> READY -> RUNNING -> FINALIZING -> SEALED
                    |           |        |  ^                        |
                    v           v        v  |                        v
                 ABORTED      ABORTED  PAUSED                   ANNOTATING
                                                                     |
                                                                     v
                                                               REVIEWED
                                                                     |
                                                                     v
                                                               REPORTED
                                                                     |
                                                                     v
                                                         GATE_DECIDED -> ARCHIVED
```

- `DRAFT -> FROZEN`：plan 和所有 registry 引用解析为 digest 并批准。冻结后变更必须新建 session。
- `PREFLIGHT -> READY`：硬件/标定/时钟/存储/watchdog/E-stop 证据齐全。
- `READY -> RUNNING`：只有 orchestrator 可激活机器人和策略 adapter。
- `RUNNING -> PAUSED`：安全停止、数据质量告警或人工暂停；恢复需要新 preflight event，进行中的 trial 保留原终态。
- `FINALIZING -> SEALED`：关闭 writer、生成 checksum/manifest 根、写 seal。失败则 session 不得统计。
- `REVIEWED -> REPORTED`：annotation set 已冻结；报告后修改标注会产生新的 report revision。

### 5.2 Trial/Episode 状态

```text
PLANNED -> SETUP -> READY -> EXECUTING -> TERMINATING -> RESETTING -> RECORDED
              |       |          |              |                         |
              v       v          v              v                         v
           INVALID  INVALID   ABORTED/STOPPED  ABORTED              ANNOTATED
                                                                         |
                                                                         v
                                                                      REVIEWED
```

- trial 是计划、随机化、复位和标注的审计单元；episode 是 `EXECUTING` 内的策略时序。正常情况下一个 trial 对应一个 episode。
- `SETUP/READY` 失败可标 `invalid_setup` 且没有 episode；一旦策略获得控制权，超时、接管、安全停止均为有效 outcome，除非冻结计划中的外因规则匹配。
- 不原地“重跑”同一个 trial。任何重试新建 trial，并通过 `retries_trial_id` 关联。

## 6. 数据契约

### 6.1 通用约束

- ID 使用全局唯一、不可复用的 UUIDv7/ULID；`episode_index` 仅用于 session 内展示。
- 所有 schema 带 `schema_version`；所有 registry 引用使用 `{id, revision, digest}`。
- UTC 使用 RFC 3339，在线顺序使用 `t_monotonic_ns + clock_id + seq`。不得用文件 mtime 作为真实采集开始时间。
- 向量字段必须由 schema 给出 dtype、shape、单位、坐标系、关节/维度名；缺失值与 padding 有显式 mask。
- `unknown` 与 `not_applicable` 分开；禁止用默认值填补历史未知版本。

### 6.2 Session 最小 schema

| 字段 | 类型/约束 | 含义 |
|---|---|---|
| `schema_version` | string | session 契约版本 |
| `session_id`, `campaign_id` | ID | 连续执行批次和评测 campaign |
| `plan_ref` | `{id,revision,digest}` | 冻结评测计划 |
| `status` | enum | 第 5.1 节状态 |
| `site_id`, `operator_ids` | string/list | 场地和操作者审计 ID |
| `started_at_utc`, `ended_at_utc` | timestamp/null | 墙钟范围 |
| `clock_sync` | object | clock IDs、方法、offset/drift/uncertainty |
| `registry_snapshot_ref` | artifact ref | 本 session 使用的完整 registry 快照 |
| `policy_bundle_ref` | artifact/ref | 模型结构、config、weight/checkpoint digest、训练 lineage |
| `software_manifest_ref` | artifact ref | 执行/评测代码 commit、dirty patch、依赖 lock、容器 digest、驱动/runtime |
| `hardware_manifest_ref` | artifact ref | robot/end-effector/sensors/compute 序列号、固件、健康状态 |
| `calibration_bundle_ref` | artifact ref | 标定 revision、参数、误差、有效期和校验结果 |
| `safety_profile_ref` | artifact/ref | 限位、watchdog、stop/recovery 参数 digest |
| `collector_config_ref` | artifact/ref | streams、采样、编码、QoS、必需性 |
| `random_seed` | integer/string | trial 队列生成种子 |
| `planned_trials`, `recorded_trials` | integer | 配额与完成数 |
| `artifact_manifest_ref`, `seal_digest` | artifact/digest/null | 制品索引和最终根校验 |

### 6.3 Trial 与 Episode 最小 schema

`trial.json`：

| 字段 | 类型/约束 | 含义 |
|---|---|---|
| `schema_version`, `trial_id`, `session_id`, `ordinal` | required | 契约和主键 |
| `task_ref`, `scenario_ref`, `perturbation_refs` | revisioned refs | 任务、场景和扰动 |
| `stratum_key` | object | 用于门禁的任务/扰动/风险分层键 |
| `randomization` | object | seed、factor values、queue position、candidate/baseline block |
| `status` | enum | 第 5.2 节状态 |
| `setup_started_at`, `execution_started_at`, `execution_ended_at`, `reset_ended_at` | timestamps/null | SOP 阶段边界 |
| `episode_ids` | list | 0 或 1 为常规值；多段必须写原因 |
| `validity` | `valid/invalid_setup/invalid_logging/invalid_external/pending` | 是否进入统计分母 |
| `invalid_reason_code`, `invalid_evidence_refs` | nullable | 必须匹配冻结排除规则 |
| `autonomy_outcome` | `success/partial/failure/timeout/safety_stop/human_takeover/not_evaluable/pending` | 自主任务结果；无效 trial 使用 `not_evaluable` |
| `safety_outcome` | `no_event/near_miss/limit_stop/collision/estop/pending` | 按 `safety_profile` 严重度优先级汇总最严重事件；完整多事件事实保留在 event 流 |
| `failure_stage`, `intervention_event_ids` | nullable/list | 失败阶段和接管证据 |
| `setup_evidence_refs`, `reset_evidence_refs` | list | 初态/末态和复位验收 |
| `retries_trial_id` | ID/null | 若为重试，指向原 trial |

`episode.json`：

| 字段 | 类型/约束 | 含义 |
|---|---|---|
| `schema_version`, `episode_id`, `trial_id`, `episode_index` | required | 主键与展示 index |
| `agent_ref` | policy bundle/ref | 真正控制该 episode 的策略 |
| `started_monotonic_ns`, `ended_monotonic_ns`, `clock_id` | required | 策略控制时间边界 |
| `end_reason` | `task_terminal/timeout/safety_stop/takeover/operator_abort/external_abort/runtime_error` | 时序结束原因 |
| `first_seq`, `last_seq` | integer | 对应时序数据范围 |
| `stream_refs`, `event_refs` | lists | 观测/动作/视频/事件 |
| `is_truncated` | bool | 区分正常终止和截断 |

`end_reason` 到 trial 结果的冻结映射如下：

| `end_reason` | 默认 `validity` | 默认 `autonomy_outcome` | 说明 |
|---|---|---|---|
| `task_terminal` | `valid` | 由 rubric 判 `success/partial/failure` | 终态判定与时序结束原因分开 |
| `timeout` | `valid` | `timeout` | 是否作为行政截尾、失败或复合终点由统计计划冻结 |
| `safety_stop` | `valid` | `safety_stop` | 同时写安全事件；不得因保护正确触发而删 trial |
| `takeover` | `valid` | `human_takeover` | 人工完成部分不得计自主成功 |
| `operator_abort` | `valid` | `failure` | 策略获得控制后、未命中预注册外因规则的人工中止 |
| `runtime_error` | `valid` | `failure` | 若错误属于被冻结的 policy/runtime bundle；只有预注册且有证据的外部基础设施故障可改为 `invalid_external/not_evaluable` |
| `external_abort` | `invalid_external` | `not_evaluable` | 必须命中预注册外因规则；否则按 `valid/failure` 处理 |

`ABORTED/STOPPED` 是 trial/episode 状态，不直接决定是否进入统计分母。完整中止原因保存在 `end_reason` 和 event 流。

### 6.4 Event 最小 schema

| 字段 | 类型/约束 | 含义 |
|---|---|---|
| `schema_version`, `event_id`, `event_type` | required | 事件身份与版本化 taxonomy |
| `session_id`, `trial_id`, `episode_id` | IDs/null | 所属范围 |
| `seq`, `t_monotonic_ns`, `clock_id`, `t_utc` | required/UTC optional | 稳定排序和跨系统关联 |
| `source` | `{component,id,revision}` | watchdog/controller/operator/collector 等 |
| `severity` | `info/warning/safety_stop/critical` | 严重度由 safety profile 定义 |
| `payload` | schema-bound object | 测量值、阈值、单位、坐标系、reason code |
| `action_taken`, `action_result` | nullable | safe-stop/estop/takeover 等及执行结果 |
| `related_event_ids`, `artifact_refs` | lists | 因果链和视频/日志证据 |
| `ingested_at_utc` | timestamp | 与源事件时间分开 |

安全事件 payload 至少记录 trigger、measured value、threshold/ref、机器人 mode、command ID、stop latency 和是否需要人工复位。

### 6.5 Artifact 最小 schema

| 字段 | 类型/约束 | 含义 |
|---|---|---|
| `schema_version`, `artifact_id`, `artifact_type` | required | video/timeseries/config/model/log/report 等 |
| `uri`, `media_type`, `size_bytes` | required | 物理位置和格式 |
| `digest` | `{algorithm:"sha256",value}` | 内容身份；目录使用 manifest 根 digest |
| `producer` | `{component,version,run_id}` | 生成者 |
| `session_id`, `trial_id`, `episode_id` | nullable IDs | 作用域 |
| `created_at_utc` | timestamp | 生成时间 |
| `time_span` | `{clock_id,start_ns,end_ns}`/null | 时序制品覆盖范围 |
| `schema_ref`, `codec` | nullable | Arrow/JSON schema、视频编码等 |
| `retention_class`, `security_class` | enum | 生命周期和访问控制 |
| `supersedes_artifact_id` | ID/null | 派生修正版关系；不覆盖原件 |

### 6.6 Annotation 最小 schema

| 字段 | 类型/约束 | 含义 |
|---|---|---|
| `schema_version`, `annotation_id` | required | 标注身份 |
| `target` | `{type,id}` | trial/episode/event/time_span |
| `rubric_ref` | revisioned ref | 成功/阶段/安全判定口径 |
| `annotator` | `{type:human/model/rule,id,version}` | 来源；人可使用受控匿名 ID |
| `labels` | schema-bound object | validity/outcome/failure stage/attempt 等 |
| `evidence` | list | artifact ID + camera/stream + 时间段/event ID |
| `confidence`, `comment` | nullable | 辅助信息，不能代替证据 |
| `status` | `proposed/reviewed/adjudicated/rejected` | 流程状态 |
| `created_at_utc`, `supersedes_annotation_id` | timestamp/ID/null | 追加式修订链 |
| `reviewer_ids`, `adjudication_reason` | list/null | 复核和裁决审计 |

### 6.7 高频时序最小列

LeRobot/Parquet frame 层至少扩展为：

```text
session_id, trial_id, episode_id, seq, clock_id, t_monotonic_ns, t_utc,
observation_timestamp_ns, observation.*, state.*,
policy_request_id, policy_start_ns, policy_end_ns, policy_action.*,
postprocess_flags, safety_filter_flags, command_id, command_sent_ns,
command_action.*, controller_ack_ns, intervention_state,
queue_latency_ns, inference_latency_ns, command_latency_ns
```

相机可继续用 MP4，但必须有逐帧时间/序号映射、掉帧统计和 artifact digest。`action` 的语义必须在 adapter schema 中明确是策略输出、后处理命令还是控制器目标，禁止模糊复用。

## 7. 执行 SOP

### 7.1 准备与冻结

1. 评测负责人创建 campaign，指定候选/基线、risk tier、mandatory strata、主要/次要指标、分母、无效规则、超时/停止规则、统计方法和样本预算。
2. registry owner 固定 robot/task/scene/object/perturbation/policy/calibration/safety revisions；plan compiler 解析 alias 并生成 digest。
3. 独立审阅者核对数据泄漏、训练/评测对象重叠、场景可执行性和安全边界；批准后状态置 `FROZEN`。
4. 生成随机化/分块队列；候选和基线尽可能共享匹配初始条件，执行顺序交错。操作员可盲化时隐藏模型名。

输出：`plan.snapshot.json`、`registry.snapshot.json`、policy/software/hardware/safety manifests、随机化队列和审批记录。

### 7.2 预检与校准

1. 记录硬件序列号、固件、计算环境、可用磁盘、温度/电量等健康状态。
2. 完成相机/手眼/末端/力传感器校准检查；与允许误差比较，记录原始测量和 calibration digest。
3. 检查坐标系、单位、动作方向、夹爪语义、限位、控制频率和 policy action shape。
4. 验证时钟 offset/drift、相机帧时间、状态/动作 topic freshness 和 collector 写盘。
5. 通过受控故障注入验证 heartbeat timeout、过期动作、限位、safe-stop/E-stop 和本地事件日志。

任一预检 hard gate 不通过，session 进入 `ABORTED`，不得通过“先跑后补配置”继续。

### 7.3 随机化、执行与终止

1. orchestrator 领取下一个不可挑选的 trial；操作员按 scene/perturbation revision 布置并上传初态证据。
2. 自动/人工检查初态容差，满足后进入 `READY`；不满足按冻结规则重置，失败则 `invalid_setup`。
3. collector 先开始并确认 streams healthy，watchdog armed 后才允许 policy active。
4. episode 开始，记录 observation/state/policy action/command action/ack/latency/video/events。操作员不得口头改变超时或成功规则。
5. 达到 task terminal、timeout、接管、安全停止或 runtime error 时终止；立刻写 `end_reason`。接管和安全停止不删除 trial。
6. collector 刷盘并写临时 checksum；数据不完整时保留 trial 和错误事件，由 validity 流程判定。

### 7.4 复位

1. 机器人进入已定义 safe pose，清除/确认停止原因；安全 stop 后必须执行安全 profile 的人工恢复程序。
2. 恢复物体、容器、相机、背景、扰动设备；记录复位后照片/位姿/传感器检查。
3. 若设备维修、重新标定、安全参数或容器发生变化，关闭当前 session 并新建 session；不得混在同一版本快照中。

### 7.5 Seal、标注与复核

1. 完成计划队列或按预定义停止规则结束后，finalize Parquet/MP4，计算每个 artifact digest 和根 manifest，写 `seal.json`。
2. validator 检查 schema、ID、校验和、视频可解码/帧映射、时间单调性/漂移、必需 streams 和 trial 边界。
3. 规则/VLM 生成 proposed annotation；发版关键标签按 risk tier 分配人工单审或双人盲审。
4. 分歧由独立 reviewer 裁决；冻结 annotation set digest。标注改变需产生新 revision 并重算下游结果。

### 7.6 指标、报告与发版

1. metric engine 只读取 sealed session、通过验证的 trial 和冻结 annotation set；输出逐 trial 指标及 denominator audit。
2. 按 task、perturbation type/level、ID/OOD、robot/site 等计划字段分层，计算 `k/n`、区间、连续指标和安全事件。
3. comparator 使用冻结 baseline release 和 matched key 比较，输出效应量、区间、缺失配对与顺序/操作者敏感性。
4. gate engine 先执行 hard gate，再执行 statistical gate，输出逐条证据和最终 `PASS/BLOCK/INCONCLUSIVE`。
5. reviewer 签署 release dossier；只有 PASS 对应的不可变 policy version 才可被 production alias 指向。

## 8. 发版门禁

### 8.1 门禁配置输入

每个 release plan 必须冻结：

- `risk_tier` 及理由，危险源和关键任务/扰动 strata。
- baseline release ID/policy digest、适用机器人/场地边界。
- 每个 stratum 的 hard requirements、主要指标、方向、统计方法、置信水平、非劣/改善 margin、最小信息量/功效规则。
- 多指标/多分层的错误率控制或层级测试顺序。
- 无效 trial、缺失流、早停、异常值、接管和安全 stop 的处理规则。

这些参数由 safety owner、任务 owner、统计 reviewer 和 release owner 基于风险与生产基线批准；查看候选结果后不得放宽。新风险等级或新任务没有可靠基线时，先建立 baseline campaign，不能套用其他任务阈值。

### 8.2 Hard gate

Hard gate 是非补偿式布尔条件，任一 mandatory 条件失败即 `BLOCK`：

| 类别 | 示例条件 | 证据 |
|---|---|---|
| 身份/冻结 | 候选权重、代码、容器、硬件、标定、安全配置均解析到 digest；执行中无未声明漂移 | manifests、registry snapshot、events |
| 预检/安全链 | watchdog、safe-stop/E-stop、限位和必要传感器自检通过 | fault-injection/preflight events |
| 严重安全事件 | 不出现计划定义的不可接受严重度/类型；任何保护失效直接阻断 | safety events、视频、controller log |
| 数据完整性 | session sealed；必需 artifact checksum/schema/time sync/视频/动作链/标注齐全 | validator report、seal |
| 协议合规 | trial 次序/随机化/复位/超时/停止规则无不可接受偏差 | trial/event audit |
| 覆盖 | 每个 mandatory stratum 达到冻结的最小有效 trial/匹配对数；不足不能判通过 | quota table |
| 关键能力底线 | 风险定义的关键任务绝对底线满足 | 分层指标及冻结阈值 |
| 复核 | 关键标签无未裁决分歧；annotation set 已冻结 | review audit |

安全阈值不硬编码在指标程序中，而在 `safety_profile` 与 gate plan 中引用。可接受的计划偏差需要事前分类；事后例外只能形成显式 `BLOCK_WITH_WAIVER_REQUEST`，不能把自动 gate 结果改写为 PASS。

### 8.3 Statistical gate

Hard gate 全部通过后才计算：

1. 二元成功/失败指标报告 `k/n` 和区间。样本小或失败极少时使用精确二项区间；一般比例可用 Wilson。绝对底线用预先指定的单侧下界。
2. 候选-基线优先按相同 task/scene/seed/object instance 配对。二元结果使用计划指定的配对差异方法或 exact/McNemar 类检验；无法配对时使用两比例差与相应区间，并说明更强的混杂风险。
3. 非劣门禁形式为：候选相对基线的效果差单侧下界不低于该 stratum 冻结 margin。margin 由该任务的风险与 baseline 波动确定，不在本文给数值。
4. TTS、路径长度、jerk、延迟等连续量事前指定总体（全 trial、成功 trial 或截尾估计）、方向和统计量；优先报告配对差和 bootstrap/稳健区间。只看成功样本时必须同时展示成功率，避免幸存者偏差。
5. 碰撞/保护失效等事件除 hard gate 外，可对事件率使用单侧上界；零观测事件不等于真实风险为零。
6. 每个 mandatory task × perturbation stratum 独立出 `PASS/BLOCK/INCONCLUSIVE`。总 gate 只有全部 mandatory strata PASS 才 PASS；任何 BLOCK 则 BLOCK；无 BLOCK 但有信息不足则 INCONCLUSIVE。
7. 汇总同时给 macro（各任务等权）和按计划权重的结果；micro 结果只作诊断。汇总不覆盖分层门禁。
8. 多个主要指标/分层需预先指定层级顺序或多重比较控制；探索性结果必须标为 exploratory，不能反向成为发版依据。

### 8.4 回归检测

- 基线固定为当前已批准 release digest，不使用动态 `latest`。
- 候选和基线共享 frozen plan、matched initial-condition keys 和交错随机执行；设备维修/重新标定后分 session，并在比较中建模/分层。
- 每项回归记录 `metric_spec_revision`、candidate/baseline query、effect、interval、decision 和 missing-pair 原因。
- 监控历史趋势可用控制图/层级模型作诊断，但正式发版仍按本次冻结 gate policy 判定，不能事后选择最有利窗口。

## 9. 报告与可追溯性

Release dossier 至少包含：

1. 候选、基线和完整 policy/software/hardware/calibration/safety manifests。
2. plan/registry/session/annotation/metric artifact digests 和数据 lineage 图。
3. 计划数量、执行数量、有效/无效及排除原因，逐 trial 清单。
4. 按 task/perturbation 的 k/n、区间、连续指标、效应量和样本不足提示。
5. 安全事件、接管、watchdog stop、协议偏差和未解决风险。
6. 每个 hard/statistical gate 的配置、证据、结果和 reviewer。
7. 适用边界、已知缺口、decision revision、审批和可选 waiver 请求。

报告中的每个表格行应带可机器解析的 `metric_id + metric_spec_revision + input_manifest_digest + stratum_key`。Markdown/HTML 是视图，JSON decision 和冻结 manifest 才是自动化输入。

## 10. 数据保留与治理

- `raw-critical`：原始视频、状态/动作、策略动作、command、controller/watchdog/安全事件和 frozen manifests。按最高风险等级设置不可变/WORM 保留，不允许只保留抽帧。
- `derived-review`：抽帧、预标注、人工标注、指标明细。可重算，但标注修订和裁决需长期保留审计链。
- `release-record`：seal、报告、gate decision、审批/waiver、发布 alias 变化，至少覆盖该模型生产生命周期和组织规定的追责窗口。
- `scratch`：解码缓存和临时可视化，短期自动清理，不得被报告引用为唯一证据。
- 生命周期由 plan 的 `retention_policy_ref` 按风险、隐私、合同和存储成本决定；本文不设统一天数。
- 热/温/冷迁移只能改变 URI，不改变 artifact ID/digest；迁移后抽样校验可读性。删除需审批、依赖检查和 tombstone，legal hold 优先。
- 视频可能包含人员/场地信息，按 security class 最小授权、加密、记录访问审计；导出前脱敏，但原始证据受控保留。

## 11. Genie02 迁移映射

| 分类 | 内容 |
|---|---|
| 可直接复用 | LeRobot Parquet/MP4 与 `meta/info/tasks/episodes`；state/action/policy_action/intervention/collector policy；现有 session/episode 验证；GSR、成功 TTS、jerk 平滑度和跳过原因；Markdown/SVG；attempt_eval 抽帧/VLM 校验/复核提示 |
| 第一优先扩展 | plan/registry freeze；全局 trial/episode ID；validity/outcome/safety 三列；policy/software/hardware/calibration/safety manifests；artifact checksum/seal；事件流和独立 watchdog；policy_action-command_action-state 与分阶段 latency；人工复核 revision；分层区间、基线比较和 gate JSON |
| 历史无法补回 | 未记录的权重/checkpoint digest、执行 commit/容器/runtime、硬件序列号/固件、标定、安全阈值、实际 command/ack/时延、时钟误差、场景随机化、碰撞/力/急停结构化事件、标注者和 rubric revision。标记 unknown 并补测，不允许推断填充 |

建议迁移顺序：先把现有 Genie02 读取器注册为 `adapters/datasets/lerobot_genie02`，保持指标结果不变；再新增 sidecar schema、seal 和 annotation；最后接入在线 event/watchdog 与统计门禁。现有历史数据只能用于能力趋势和迁移验证，缺少 hard-gate 证据时不得追认成完整发版评测。
