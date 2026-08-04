# VLA 真机评测指标与统计框架

## 1. 使用规则

本模块规定“测什么、如何测、如何汇总、何时不能下结论”。每个指标必须注册 `metric_id`、spec revision、实现 revision、适用任务/机器人、输入流、单位、分母、方向、缺失策略和校验样例。统计独立单元默认是独立复位后的 `trial`，不是帧。

优先级定义：

- **P0**：发版或结论有效性所必需；缺失可触发 BLOCK/INCONCLUSIVE。
- **P1**：强烈推荐的诊断与回归指标；在具备相应传感器/任务条件时启用。
- **P2**：研究或优化指标；不能替代 P0 结果。

本框架不是安全标准或认证方案。所有阈值、风险等级、暴露边界、非劣 margin 和最低样本量由任务风险、机器人厂商约束、生产基线和正式风险分析决定，并在看结果前冻结。下文没有来源的数值/流程均是工程建议。

## 2. 统一数据集与分母

每个 trial 的三个正交字段：

```text
validity: valid | invalid_setup | invalid_logging | invalid_external | pending
autonomy_outcome: success | partial | failure | timeout | safety_stop | human_takeover | not_evaluable | pending
safety_outcome: no_event | near_miss | limit_stop | collision | estop | pending
```

`ABORTED` 属于 trial/episode 状态，不自动等于 invalid；具体原因使用合法 `end_reason`（如 `operator_abort` 或 `external_abort`）。episode 开始后，timeout、策略导致的 safety stop、E-stop 和人工接管原则上是 valid 的模型表现；只有冻结排除规则明确命中且证据齐全的非模型外因才可 `invalid_external`。任何重试创建新 trial，并以 `retries_trial_id` 关联。

`safety_outcome` 是按 `safety_profile` 严重度优先级得到的最严重摘要，不是唯一安全事实；同一 trial 的 collision、E-stop、safe-stop 等完整多事件序列必须保留在 event 流。

报告必须公开以下流转：

```text
planned -> attempted -> valid
                    -> invalid_setup
                    -> invalid_logging
                    -> invalid_external

valid -> success | partial | failure | timeout | safety_stop | human_takeover
invalid_* -> not_evaluable
```

主要自主成功率分母为全部 valid trials。接管后由人完成不能计自主成功；删掉 intervention frames 也不能改变 trial 终态。成功条件下指标用 `valid & success`，必须明确标注并与全体成功率并列。安全暴露分母还应给自主运行小时、运动距离或机器人动作数。

## 3. 指标字典

“陷阱”列是实现和解释时的最低审计要求。方向中的 ↑/↓ 表示通常越高/越低越好，`gate` 表示按非补偿规则判定，`context` 表示无通用单调方向。

| 名称 | 定义/公式 | 数据源/传感器 | 单位 | 聚合/分母 | 方向 | P0/P1/P2 | 陷阱 |
|---|---|---|---|---|---|---|---|
| 严格任务成功率 | \(\hat p=\sum I(outcome=success)/n_{valid}\)；成功需满足冻结 rubric 全部必要条件 | 终态、事件、视频、人工/规则标注 | % 与 `k/n` | task/scene/OOD 分层；全部 valid | ↑ | P0 | 接管/安全 stop/timeout 不能排除；必须报区间 |
| Partial progress | \(P=\sum w_j I(g_j)/\sum w_j\)，按先决依赖计分 | 子目标 annotation、状态、视频 | 0--1 或 % | valid trial 均值/分位数；逐任务 macro | ↑ | P0（长程）/P1（短程） | 不能替代 full success；权重不得事后改 |
| 子目标完成率 | 完成子目标 j 的 eligible valid trials / 有机会执行 j 的 valid trials | 事件、阶段标注 | % 与 `k/n` | subgoal × task | ↑ | P1 | 后置步骤的 eligibility 要冻结；重复完成不重复计分 |
| 首次失败阶段 | 第一个未满足/不可恢复的 stage ID | 盲审标注、事件 | 类别 | valid 非成功 trial 的计数/比例 | context | P1 | 症状不一定是根因；保留 unknown |
| 指令全约束满足率 | 对象/属性/位置/关系/顺序/数量/否定/stop 全部满足 | 终态、视频、任务 parser/rubric | % 与 `k/n` | valid trials，按约束类型分层 | ↑ | P0 | 做对动作但选错对象不能成功；VLM 只能预标 |
| 原子指令约束满足率 | \(\sum I(c_m)/\#eligible\ constraints\) | 同上 | % | constraint type × task；macro 优先 | ↑ | P1 | 多约束 trial 非独立，不把约束数当 trial 数 |
| 成功 TTS | \(t_{first\ sustained\ success}-t_{control\ enabled}\) | 单调时钟、terminal detector/标注 | s | `valid & success` 的 median/IQR/p90；同时报 success rate | ↓ | P0 | 仅成功样本有幸存者偏差；起止事件必须统一 |
| 全体 time-to-success / 竞争终止 | 固定观察窗可形成行政截尾；失败、接管、安全 stop、runtime error 按预注册 estimand 作为吸收/竞争事件；可报告 cumulative incidence、cause-specific 量或复合终点，满足假设时才用 RMST | trial 边界、end_reason、成功时刻 | s、概率 | 全部 valid；task/stratum 分层 | 依 estimand | P1 | 不能把所有未成功 trial 当普通右截尾；不能伪造为成功时间 |
| 吞吐 | 固定窗口内严格成功数 / 墙钟；另给自主运行时吞吐 | session/trial 时间、复位时间 | success/h | campaign/session | ↑ | P1 | 是否包含复位/故障时间必须明确；任务混合需冻结 |
| 末端路径长度 | \(L=\sum_t \lVert x_t-x_{t-1}\rVert\)；旋转另算 geodesic | 机器人 state/外部追踪 | m、rad | 每 valid trial；成功条件与全体分别 | ↓ | P1 | 命令≠实际轨迹；米和弧度不可相加 |
| 关节路径长度 | \(\sum_t \lVert W(q_t-q_{t-1})\rVert\)，W/关节范围冻结 | 编码器 state | rad 或归一化量 | 同上 | ↓ | P1 | 跨机器人权重和自由度不同，不宜直接横比 |
| 参考路径效率 | \(E_L=L^*/\max(L,L^*)\)，仅在 L* 可验证时 | state、规划器/示教 reference | 0--1 | valid success；同时给原始 L | ↑ | P1 | 不可靠的“最短路径”会误导；失败处理须另报 |
| 动作总变差 | \(TV=\sum_t \lVert W(a_t-a_{t-1})\rVert\) | policy_action 与 command_action | action unit | 每 trial，两个动作层分别 | ↓ | P1 | 动作尺度/频率敏感；不能混 policy 与 command |
| 冗余/反向动作 | 重复切换、方向反转、夹爪切换或无进展命令数 | action/state、task phase | count 或 /min | valid trial；单位成功量并列 | ↓ | P2 | 阈值与去抖窗须冻结；探索动作未必错误 |
| 终点位姿误差 | \(\lVert x_T-x^*\rVert\) 与旋转 geodesic \(\theta(R_T,R^*)\) | state/外部追踪、目标 pose | m、deg/rad | eligible trials；median/p95/max | ↓ | P1 | 终点误差不是轨迹跟踪误差；坐标系/标定需记录 |
| 轨迹跟踪误差 | 时间/弧长对齐后的 position/orientation RMSE、p95、max | state 与 reference trajectory | m、deg/rad | 有参考的 eligible trials | ↓ | P1/P2 | 对齐方法影响结果；仅目标任务不适用 |
| 峰值/分位速度 | \(v=dx/dt\)，关节/平移/旋转分开 | state + monotonic timestamp | m/s、rad/s | 每 trial p95/max，再跨 trial | gate/↓ | P0（限值）/P1 | 不均匀采样、噪声和单位；frame 不独立 |
| 峰值/分位加速度 | \(a=dv/dt\) | 同上 | m/s²、rad/s² | 同上 | gate/↓ | P0/P1 | 差分放大噪声；滤波参数必须版本化 |
| Jerk RMS/能量 | \(j=da/dt\)；RMS 或 \(\int\lVert j\rVert^2dt\)，各空间分开 | state 或 command + monotonic time | m/s³ 等；能量单位 | valid autonomous segment；成功/全体并列 | ↓ | P1 | 时长、频率、坐标表示敏感；不能跨量纲求和 |
| 抖振带内能量 | 稳态段速度/动作在冻结频带 \([f_1,f_2]\) 的 PSD 积分或高通 RMS | state/action，高频时间戳 | signal² 或 RMS unit | phase/trial 后跨 trial | ↓ | P1 | 频带需低于 Nyquist；接触动作与抖振要区分 |
| 抓取保持成功率 | 规定保持时长/运动后无掉落且滑移未超限 | 视频、外部追踪、触觉/夹爪 state | % 与 `k/n` | 抓取成功且进入 hold 的 eligible trials；另报端到端成功 | ↑ | P1 | 条件分母会隐藏抓取失败，必须并列端到端率 |
| 峰值滑移/滑移率 | 物体相对夹爪位姿变化的峰值/速率 | 外部视觉、触觉 | mm、mm/s、deg | hold/transport phase | ↓ | P1 | 无传感器不可从开度强推；遮挡需缺失标记 |
| 峰值接触力/力矩 | contact phase 的 \(\max \lvert F-F_0\rvert\)、\(\max\lvert\tau-\tau_0\rvert\) | 校准 F/T、触觉或控制器估计 | N、Nm | 每 event/trial p95/max；超限 incidence | gate/↓ | P0（风险任务）/P1 | 带宽、零偏、滤波、饱和；估计值不等于传感实测 |
| 冲量/力冲击 | \(J=\int_{contact}\lvert F-F_0\rvert dt\)，另给 force rise rate | F/T + time | N·s、N/s | contact event/trial | ↓ | P1 | 接触窗定义和采样带宽影响大 |
| ID 成功率 | ID strata 严格成功 `k/n` + CI | trial metadata、outcome | % | task/axis 分层，macro | ↑ | P0 | ID 边界和训练重叠需声明 |
| OOD 成功率 | 每个单轴/组合 OOD 的严格成功 `k/n` + CI | scenario factors、outcome | % | OOD axis/level/task | ↑ | P0 | 不把所有 OOD 混成一个总分 |
| 泛化保持率 | \(R_{gen}=p_{OOD}/p_{ID}\)，并报 \(p_{OOD}-p_{ID}\) | paired ID/OOD strata | ratio、百分点 | task/axis；配对/分层区间 | ↑ | P1 | ID≈0 时比值不稳定；比值不能替代绝对成功率 |
| 扰动层 Full SR | 最终严格成功 / 所有分配到该扰动 stratum 的 valid trials | assignment、perturbation event、outcome | % 与 `k/n` | type × level × phase；planned-stratum/ITT-style | ↑ | P0 | canonical gate 指标；扰动是否成功施加另记，不能事后切换到 eligible 分母 |
| 恢复状态率 | 在 eligible perturbations 中，恢复窗内重回可继续状态且满足冻结恢复条件的比例 | 扰动/恢复事件、state、视频 | % 与 `k/n` | type × level × phase | ↑ | P1 | 分母须排除扰动前已失败；恢复定义不得事后修改 |
| 恢复时间 | \(t_{recovered}-t_{perturbation}\)；固定恢复窗可行政截尾，安全 stop/takeover 等按竞争终止处理 | monotonic event timestamps | s | eligible perturbations；预注册 time-to-event estimand | ↓ | P1 | 成功恢复与最终任务成功需分开；不能把竞争终止当普通删失 |
| 碰撞 trial incidence | 至少一次 collision 的 valid trials / valid trials | 碰撞检测、F/T、控制器、视频复核 | % 与 `k/n` | severity/object class/task 分层 | ↓/gate | P0 | 去重窗、允许接触和碰撞必须区分；零观测≠零风险 |
| 碰撞事件暴露率 | collision event 数 / 自主小时或运动距离 | 事件流、active time/state | event/h 或 event/km | campaign/stratum + 单侧上界 | ↓/gate | P0 | 同一碰撞弹跳不能重复计数；暴露定义一致 |
| 近碰 incidence | 最小距离/预测 TTC 等进入冻结 near-miss 区间且未接触 | 人/障碍追踪、区域传感器、state | %、event/h、m 或 s | valid trials 与 exposure 双分母 | ↓/gate | P0（人机/移动）/P1 | 主观“差点”不可用；传感器盲区和误差要计入 |
| 超限 incidence | 位置/速度/力/工作空间/温度等越过 safety profile 阈值 | controller/watchdog/sensors | % 与 count | limit type × severity；valid trial/exposure | ↓/gate | P0 | software clamp 与真实越限分开；阈值需引用 revision |
| E-stop incidence | E-stop 被触发的 trial/事件率，区分 automatic/manual | E-stop circuit/controller/event | %、count、event/h | cause/severity/task | ↓/gate | P0 | 触发成功不代表策略安全；不可将 trial 设 invalid |
| Safe-stop 成功率 | 触发后在 profile 规定时间/距离内进入目标安全状态 | watchdog、controller state | % 与 `k/n` | trigger type；fault-injection 与实际事件分开 | ↑/gate | P0 | 框架测试不等于功能安全认证；fail-to-log 不得阻碍停止 |
| Stop latency/distance | trigger timestamp 到安全状态；期间移动距离 | watchdog/controller/state | ms、m/rad | trigger type 的 p50/p95/max | ↓/gate | P0 | 多时钟需校准；日志延迟不等于物理停止延迟 |
| 人工接管率 | `human_takeover` valid trials / valid trials；另给 event/h | operator event、intervention state | %、event/h | reason/task/phase | ↓ | P0 | 安全人员正确接管可是安全成功，但自主结果仍失败 |
| 介入占空比 | 人工控制时长 / episode 时长 | frame intervention + source policy | % | takeover trials/全 valid 分别 | ↓ | P1 | 删介入帧后不能把 trial 记自主成功；帧标志需事件化 |
| E2E latency | \(t_{controller\ ack}-t_{observation\ available}\)，链路各阶段另算 | source timestamps、request/command IDs | ms | 先 per trial p50/p95/p99，再分层汇总 | ↓/gate | P0 | 混用时钟、均值掩盖尾延迟、帧伪重复 |
| Deadline miss rate | latency > frozen deadline 的 commands / eligible commands | 同上 + control deadline | % 与 `k/n` | trial/session/task mode | ↓/gate | P0 | deadline 随控制模式不同；丢失命令不能从分母消失 |
| Action staleness | \(t_{execute/sent}-t_{source\ observation}\) | observation source time、command time | ms | command/trial p50/p95/p99/max | ↓/gate | P0 | 不能用文件写入时刻代替 source timestamp |
| 推理/排队/后处理延迟 | 各阶段结束减开始；安全过滤单列 | runtime instrumentation | ms | trial/session percentiles | ↓ | P1 | 只报 inference 会漏掉队列和传输瓶颈 |
| 资源水位 | CPU/GPU、RAM/VRAM、温度、功耗、queue depth 的 peak/percentile | OS/runtime/telemetry | %、GB、°C、W、count | trial/session；与 latency 对齐 | context/gate | P1 | 采样本身有开销；均值掩盖 OOM/thermal throttle |
| 运行可靠性 | inference error、process restart、dropped command、watchdog timeout / trial 或 hour | runtime/watchdog events | %、event/h | reason/component/version | ↓/gate | P0 | 基础设施原因与策略原因分列，但都要审计 |
| 必需流完整率 | 实际有效 sample/预期 sample；必需 artifact present/expected | collector manifest、seq | % | stream/trial/session | ↑/gate | P0 | 插值不能恢复安全证据；预期频率要考虑设计容差 |
| 掉帧/序号缺口率 | missing sequence IDs / expected IDs | camera/stream seq map | % | stream/trial/session | ↓ | P0/P1 | 编码帧率与采集 source FPS 不同；重复帧另计 |
| 时钟质量 | offset、drift、sync uncertainty、timestamp non-monotonic count | clock sync service | us/ms、ppm、count | clock pair/session | gate | P0 | UTC 对齐不能替代单调时钟；误差会污染 latency |
| 动作链关联率 | 可由 request ID 关联 observation→policy→command→ack 的命令比例 | timeseries IDs | % | trial/session | ↑/gate | P0 | 依靠邻近 timestamp 猜关联不够可靠 |
| 标注完整/分歧率 | reviewed labels/required；双标 disagreement/dual-labeled | annotation revisions | % 与 count | label type/risk tier | ↑ / ↓ | P0 | 自动/VLM proposed 不等于 reviewed；一致不等于正确 |

## 4. “如何测”的实现规范

### 4.1 成功、partial 与指令遵循

每个 task registry 必须包含：成功终态、禁止条件、子目标 DAG、子目标权重、阶段 eligibility、terminal 持续时间、timeout、可用传感器、证据视角、模糊案例和裁决示例。严格成功建议由确定性状态规则优先判定，视频人工复核作为补充；规则或 VLM 首先只生成 `proposed`。

指令先映射为冻结约束集合：`object_identity`、`attribute`、`target`、`spatial_relation`、`order`、`quantity`、`negative_constraint`、`stop_condition`。报告原子约束满足率和所有必要约束同时满足率；不得用平均原子得分替代严格成功。

### 4.2 TTS 与截尾

统一起点是机器人/策略实际获得控制权的 `control_enabled`，不是文件创建、相机启动或推理预热时间。终点是成功条件首次连续满足 rubric 规定的稳定窗。固定观察窗结束且没有提前终止可作为行政截尾；timeout 是否作为失败或行政截尾取决于冻结 estimand。takeover、safety stop、明确失败和 runtime error 是吸收性失败或竞争事件，不能按普通非信息删失处理。

报告三项：

1. `valid & success` 的 TTS median/IQR/p90 和样本量；
2. 全 valid trial 的 cumulative incidence、cause-specific 结果或预注册复合终点；仅在删失假设成立时使用 Kaplan-Meier/RMST；
3. timeout、明确失败、takeover、safety stop、runtime error 等终止数量。

如业务需要“失败代价时间”，可另算 `penalized_time=min(TTS, tau)` 或失败赋 \(\tau\)，但名称必须带 `penalized`，不得称真实 TTS。

### 4.3 路径、轨迹与平滑度

计算前先完成：选择 `state` 或 `command` 层、定义坐标系和关节顺序、验证单调时钟、标记人工控制段、处理缺口、确定重采样频率和滤波器 revision。旋转使用 SO(3) geodesic/log-map，避免直接对四元数符号或 rot6d 分量做欧氏差。

速度/加速度/jerk 在相同物理空间分别计算。默认先产出逐 trial 的 RMS、p95、max、能量和有效样本数，再跨 trial 聚合。抖振只在任务定义的稳态/保持段计算，并冻结频带。过滤人工段的数值可以作为“自主段控制质量”，但该 trial 的成功、接管和安全分母仍保持原样。

### 4.4 抓取与接触

将 attempt、contact、closure、lift、hold、transport、release 标成阶段事件。至少报告端到端抓取任务成功、进入 hold 后的条件保持成功、掉落/滑移事件和最大接触载荷。F/T 数据要保存原始/校准值、zeroing、量程、采样率、滤波和饱和 flag；视频不足以测量瞬态冲击。

### 4.5 泛化与扰动恢复

每个 OOD trial 记录变化轴、强度及哪些因素仍是 ID。泛化结果至少提供 ID 绝对成功率、OOD 绝对成功率、百分点差和保持比；任何汇总都保留单轴与组合 OOD 分层。

扰动事件必须记录 planned/actually_applied、类型、强度、时刻、阶段和验收证据。canonical 扰动层 Full SR 使用所有分配到该 stratum 的 valid trials。恢复 eligibility 为：扰动实际施加，施加前策略仍在有效推进且未出现不可逆失败；恢复状态率和恢复时间只在该冻结子集中计算，不能替代 canonical Full SR。

### 4.6 安全事件

安全 event payload 至少包含：trigger、measured value、threshold/ref、单位、坐标系、source、severity、robot mode、request/command ID、action taken/result、stop latency、人工复位和证据引用。相近采样触发按冻结 debounce/merge window 合并为一个物理事件。

碰撞须区分任务允许接触、非期望物体接触和不同严重度；近碰须用距离/TTC/禁区进入等可测定义；超限须区分 `command_clamped`、真实 state 越界和保护失效。所有严重安全项先进入 hard gate，事件率只用于补充趋势，不能由高任务成功率抵消。

### 4.7 延迟、陈旧度与资源

每个 policy request 和 command 建立链路：

```text
observation_source -> observation_available -> queue_enter -> inference_start
-> inference_end -> postprocess_end -> safety_filter_end -> command_sent
-> controller_ack -> state_response
```

各节点记录 `clock_id` 和 `t_monotonic_ns`；跨时钟差只在 offset/drift uncertainty 通过预检时计算。动作陈旧度优先以 source timestamp 到 controller execute/ack 计算。percentile 先按命令生成逐 trial 指标，再以 trial/session 聚类 bootstrap；同时保留全局命令分布用于运行诊断。

资源采样与 request ID/时间轴对齐，至少覆盖 CPU/GPU/RAM/VRAM、温度、队列深度和错误/重启；功耗按硬件能力选配。deadline miss 必须把超时、丢失和无法关联的 eligible command 按冻结规则纳入审计，不能只分析成功返回。

### 4.8 失败 taxonomy 与数据质量

推荐版本化一级失败类：`perception_grounding`、`planning_reasoning`、`reach_navigation`、`grasp_contact`、`transport_manipulation`、`place_release`、`instruction_sequence`、`recovery`、`timeout_stall`、`safety`、`runtime_infrastructure`、`unknown`。另存 `failure_stage`、observable symptom、primary/secondary cause 和 evidence；评审不能仅看结局倒推模型内部根因。

validator 在指标前检查 schema、digest、必需 artifact、视频解码/逐帧映射、sequence、时间单调性/同步误差、trial 边界、动作链、传感器范围和标注 revision。不可恢复的关键数据缺失标 `invalid_logging` 并触发数据门禁审查；插值结果必须另存派生流和 mask，不得覆盖 raw。

## 5. 实验设计

### 5.1 随机化、配对与分块

候选和基线使用相同 frozen plan。以 `task × scene × object instance × initial pose × perturbation` 形成 matched key；在 operator/session/time block 内随机模型顺序并交错执行，记录 seed、计划顺序、实际顺序和偏差原因。禁止操作员基于前一结果挑选下一模型。

设备维修、固件/标定/安全配置变化后开启新 session/block。物理初态无法完全复现时记录实际位姿和误差，分析中按 matched pair 比较并把 session/operator/block 作为分层或模型因素。

### 5.2 盲评、双标与裁决

标注页面隐藏 policy 名称、版本、候选/基线身份和无关性能元数据。关键 success、partial、failure stage 和 safety severity 按 risk tier 双人独立标注；报告 raw agreement 和适当的一致性统计，分歧进入第三方 adjudication。最终统计只读取冻结的 adjudicated/reviewed annotation set digest。

如果机器人动作风格或视频水印暴露模型，报告 `blinding_limited`，但仍隐藏显式版本信息。自动规则/VLM 只能提议标签，不能单独裁决短时碰撞、力冲击或发版关键安全事件。

## 6. 统计分析计划

### 6.1 二元比例与区间

所有比例给 `k/n`。默认双侧 Wilson 置信区间；绝对底线/安全事件率门禁可使用预注册单侧界。Wilson 区间中心与半宽可按：

\[
\frac{\hat p+z^2/(2n)}{1+z^2/n}
\ \pm\ 
\frac{z}{1+z^2/n}\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}
\]

极小 n、`k=0`、`k=n` 或稀有安全事件优先用预先指定的精确二项（Clopper-Pearson）区间。Jeffreys 区间 `Beta(k+1/2,n-k+1/2)` 可作为事前选择的替代方案，报告其贝叶斯含义；不得按结果挑选更有利方法。零碰撞只表示 `k=0/n`，仍必须给单侧上界。

### 6.2 配对模型比较与效应量

二元 matched pairs 建立 2×2 表，主要效应量为候选减基线的配对成功率差（百分点）及区间；exact McNemar 使用 discordant pair 判断差异。报告匹配总数、双方成功、仅候选成功、仅基线成功、双方失败和缺失配对原因。

连续量先计算 pair difference，再报告 median/mean difference、业务单位区间和相对变化。bootstrap 必须以 matched pair/trial 为重采样单元；同一 trial 内帧不得独立重采样。可补充 Wilcoxon/稳健方法或含 task/session/operator 的混合效应模型，但需预注册并检查模型假设与收敛。

优先报告可解释效应量：成功率百分点差、泛化保持差、TTS 秒差或 RMST 差、路径/jerk/latency 百分比变化、事件率差。相对比在 baseline 为 0 或接近 0 时不稳定，应标 unavailable。

### 6.3 Macro、micro 与多重比较

- `macro`：每个 task/stratum 先求指标，再等权平均；作为跨任务主汇总。
- `weighted macro`：按冻结的生产/风险权重平均；权重与适用边界进入 plan。
- `micro`：合并 trial 后计算；只作诊断并公开每层 n，避免样本多的简单任务主导结论。

mandatory strata 分别给 `PASS/BLOCK/INCONCLUSIVE`，总均值不覆盖分层门禁。多个主要指标/分层使用预注册层级测试顺序，或 Holm 等 family-wise error 控制；大规模探索性切片可用 FDR，但必须标记 exploratory，不能事后升级为发版依据。

### 6.4 TTS、连续量与缺失

TTS 只对满足预注册行政截尾定义的 trial 做右截尾；明确失败、takeover、safety stop 和 runtime error 按吸收性失败/竞争事件处理，并报告 cumulative incidence、cause-specific 或复合终点结果。连续量按计划指定分析总体：全部 valid、valid autonomous segment、或 valid success。只分析 success 时必须并列 success `k/n`，必要时做“全 valid + missing reason”敏感性分析。

异常值只能按传感器故障/协议定义处理；保留原值、reason、排除前后结果。传感器不具备时为 `not_applicable`，本应具备却缺失为 `missing/invalid_logging`，两者不能混合。对缺失安全证据不得用统计插补获得 PASS。

### 6.5 样本不足与决策

计划冻结每个 mandatory stratum 的最小 valid trials、最小 matched pairs、最大 invalid 比例、置信水平、margin 和停止规则。结果出现以下任一情况即 `INCONCLUSIVE`（除非已触发安全/能力 BLOCK）：

- 有效 trial 或匹配对不足；
- 区间跨越绝对底线、非劣 margin 或安全上限，无法作方向结论；
- 关键任务/扰动层缺失，或数据/标注分歧未解决；
- 成功全为 0/1 且信息量不足，模型比较不可稳定估计；
- 计划偏差、顺序/操作员混杂或时钟/传感器问题使结果不可解释。

`p>0.05`、零观测事件、预算耗尽或“平均值看起来更好”均不构成 PASS。任何 mandatory BLOCK 优先于统计汇总；无 BLOCK 但有任一 mandatory INCONCLUSIVE，则总结果 INCONCLUSIVE。

## 7. 输出与审计格式

每个聚合指标记录：

```json
{
  "metric_id": "task.strict_success_rate",
  "metric_spec_revision": "...",
  "implementation_revision": "...",
  "input_manifest_digest": "sha256:...",
  "annotation_set_digest": "sha256:...",
  "stratum_key": {},
  "population": "valid_trials",
  "numerator": 0,
  "denominator": 0,
  "estimate": null,
  "unit": "proportion",
  "interval": {"method": "wilson", "level": 0.95, "lower": null, "upper": null},
  "missing_reasons": {},
  "decision": "PASS|BLOCK|INCONCLUSIVE|OBSERVE"
}
```

发布报告至少包含：分母流转表、逐 task/scene/OOD/perturbation 的 `k/n+CI`、成功与 partial、TTS 截尾结果、路径/控制质量、安全事件和暴露、延迟 p50/p95/p99、deadline miss/action staleness、资源与运行错误、指令约束、失败 taxonomy、数据质量、候选-基线配对效应、macro/weighted/micro、多重比较方法及全部 INCONCLUSIVE 原因。

## 8. Genie02 的接入顺序

1. 保留现有 GSR、成功 TTS 和 jerk smoothness 的数值，注册为带 revision 的兼容指标；新增 `validity/autonomy_outcome/safety_outcome` sidecar 和分母审计。
2. GSR 增加逐任务 `k/n`、Wilson/精确区间、macro/micro；TTS 增加预注册的行政截尾、吸收/竞争事件与复合终点处理；jerk 明确动作层、坐标空间、采样/滤波和有效段。
3. 在线补采 `policy_action -> command_action -> state/ack`、request/command ID、多阶段 timestamps、碰撞/近碰/超限/E-stop/takeover/safe-stop events 和资源 telemetry。
4. 增加 rubric/subgoal/failure taxonomy、盲评双标与裁决；再接入配对 comparator 和 hard/statistical gate。

历史数据缺少某项 P0 证据时应标 `unknown/not_available` 并补测，不能从 MP4、文件名或默认配置反推成已测量事实。
