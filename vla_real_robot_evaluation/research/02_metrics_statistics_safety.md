# VLA 真机评测的指标、统计与安全惯例调研

> 调研日期：2026-08-04。本稿不新增网络检索，事实依据限于本项目已核对的论文/官方资料。`[PAPER]` 表示论文明确报告，`[OFFICIAL]` 表示官方技术资料，`[INFERENCE]` 表示跨工作归纳，`[ENG]` 表示本文提出的工程建议。`[ENG]` 不是行业标准，也不构成机器人安全认证。

## 1. 结论摘要

1. `[PAPER]` 代表性 VLA 真机论文仍以任务成功率为主，常见每任务/条件 5--20 次；大型内部评测可达到数千次 trial。样本规模差异很大，因此必须同时给逐层 `k/n`，不能只给跨任务平均值。
2. `[PAPER]` 长程和灵巧任务已经普遍加入 partial progress、子步骤或 rubric：pi0/pi0.5、OpenVLA、RDT-1B、GR00T N1 都提供了不同形式的分项判定。完整成功率仍须并列报告，不能用平均进度替代。
3. `[INFERENCE]` 学术论文较少系统报告碰撞、近碰、力超限、急停、接管、动作陈旧度和资源水位；这些是发版评测必须补齐的工程维度，而不是可以从 success rate 推断的性质。
4. `[OFFICIAL]` NIST/SEMATECH 的比例区间资料支持 Wilson 类区间；小样本、全成功或零事件时正态近似不可靠，可采用精确二项等方法。样本不足应给 `INCONCLUSIVE`，而不是把“没有发现差异”写成“等效”。
5. `[INFERENCE]` 公平比较的强惯例是共享初始条件、候选/基线配对和交错执行；成功判定应盲化模型身份，关键或模糊标签双人独立标注、分歧裁决。
6. `[ENG]` 评测结果应由任务能力、效率、控制质量、泛化/恢复、安全、系统实时性、指令遵循和数据质量共同组成。数据与严重安全项采用非补偿式门禁，不应折成一个可被成功率抵消的总分。

## 2. 既往工作实际测了什么

| 类别 | 论文中的常见做法 | 可复用点 | 已知缺口 |
|---|---|---|---|
| 任务成功 | RT-1/RT-2/Octo/OpenVLA 等以 rollout success 为主；DROID 给逐任务完整成功条件 | 每个任务预先冻结可观察终态；报告逐任务 `k/n` | 很多工作没有公开统一几何阈值、标注者和取消规则 |
| Partial / 子目标 | pi0/pi0.5 报 task progress；RDT-1B 报 Pick/Turn/Get/Pour/Place；GR00T N1 使用 0.5 分或限时完成比例 | 长程任务分阶段、保留 full success | 不同论文 rubric 不同，分数不可跨任务直接横比 |
| 泛化 | RT-1/RT-2/OpenVLA 按物体、位置、背景、环境、语言等分轴；pi0.5 强调全新家庭 | 单轴 OOD 用于归因，组合 OOD 用于生态有效性 | “OOD/zero-shot”边界常不一致，需声明目标域数据和微调范围 |
| 扰动/鲁棒 | RT-1 测干扰物、遮挡和背景；其他工作常测相机位移、新包装、新场景 | 扰动类型与等级要受控并分层 | 很少显式区分“扰动后仍成功”和“检测并恢复” |
| 长程效率 | pi0/pi0.5 给任务持续时间或超时；多数工作仍主要报成功/进度 | 固定任务超时，记录实际完成时间 | 对失败 trial 的 TTS 截尾处理很少被完整报告 |
| 系统实时性 | RT-X/OpenVLA 等报告运行频率或部署配置；部分工作报告推理延迟 | 控制频率与动作后处理必须冻结 | p95/p99、deadline miss、动作陈旧度和资源水位普遍缺失 |
| 安全与介入 | 部分模型卡声明不适合人机交互；pi0/pi0.5 区分人工高层控制等 oracle 条件 | 人工帮助必须作为实验条件显式分开 | 碰撞、近碰、力限、E-stop、接管原因和停止性能通常未统一报告 |

因此，“既往论文常报”不能等同于“部署评测足够”。框架应兼容论文式成功率，同时增加安全、实时性和数据有效性指标。

## 3. 指标分类与测量惯例

### 3.1 成功、partial 与子目标

- **严格成功**：在冻结的超时内，所有必要目标、顺序、对象/位置和约束均满足，且无禁止事件。由可计算终态、视频或两者联合判定。报告 `k/n` 和区间。
- **Partial**：对预定义子目标 \(g_j\) 赋权 \(w_j\)，计算 \(P=\sum_j w_j I(g_j)/\sum_j w_j\)。有前置依赖时，后置子目标只有在前置条件满足后才得分；禁止根据候选结果事后增加容易得分的步骤。
- **子目标完成率**：每个子目标独立给 `完成 trial 数/有机会执行该子目标的 valid trial 数`。同时报告首次失败阶段，避免平均 progress 掩盖稳定卡点。
- **长程任务**：同时报告 full success、partial progress、完成子目标数、失败阶段和超时。重复完成同一子目标不应重复计分。

论文中的 partial rubric 是任务内诊断工具，不是跨任务统一量纲。跨任务汇总时应先对每个任务标准化，再进行任务等权 macro 汇总。

### 3.2 时间、路径、动作与轨迹质量

- **TTS（time to success）**：从 `episode control enabled` 到首次持续满足成功条件的时间。仅看成功 trial 会产生幸存者偏差，应并列给成功条件下 TTS，以及预注册终点下的全体 time-to-event 结果。
- **截尾与竞争终止**：只有固定观察窗结束且此前没有吸收性/竞争终止时，才可按预注册 estimand 视为行政右截尾。明确任务失败、takeover、safety stop、runtime error 是吸收性失败或竞争事件，不得当作普通非信息删失；应预注册 cumulative incidence、cause-specific estimand 或复合终点。把失败统一赋为 timeout 只能标为惩罚式诊断量。
- **路径效率**：末端或关节路径长度 \(L=\sum_t d(q_t,q_{t-1})\)。有可验证参考路径 \(L^*\) 时可给 \(L^*/\max(L,L^*)\)；无可靠参考时给原始路径长度及成功条件下相对 baseline 的配对差。
- **动作效率**：动作变化总量、有效命令数、反向/重复动作数、夹爪切换次数及单位成功的动作量。应分别基于 `policy_action` 和 `command_action` 计算，以暴露后处理或 safety filter 的影响。
- **轨迹误差**：对具备时序参考的任务，用同坐标系、时间对齐后的 RMSE/最大误差；仅有目标位姿时报告终点位置/姿态误差，不把“到终点”误称为全轨迹跟踪。
- **速度/加速度/jerk**：用真实单调时间差分，单位分别为 position/s、position/s²、position/s³；非均匀采样需要重采样或适合非均匀时间的差分。平移、旋转、关节和夹爪分别报告，禁止无量纲直接相加。
- **抖振**：在冻结频带和滤波器后，测稳态段高频速度/动作 RMS、带内能量或零交叉率。频带必须低于 Nyquist，跨控制频率比较需统一物理频带。

jerk 很容易被控制频率、坐标表示、滤波和短轨迹支配。现有 Genie02 jerk 能量可保留为同硬件同频率下的版本回归量，但跨机器人、跨动作空间横比需要重新定义归一化和带宽。

### 3.3 抓取稳定性与接触质量

- **保持成功率**：抓取后在规定姿态/运动和保持时长内未掉落、未超滑移阈值。
- **滑移**：使用外部视觉、触觉或物体/夹爪相对位姿估计峰值滑移和滑移速率；没有相应传感器时标记 unavailable，不能从夹爪开度单独断言滑移。
- **抓取裕量**：物体进入稳定抓持区的几何/力学裕量，或夹爪闭合后稳定时间；定义随夹具与传感器变化。
- **峰值力与冲击**：力/力矩传感器或控制器估计给出峰值、超阈持续时间和冲量 \(J=\int |F-F_{baseline}|dt\)。传感器带宽、零偏、滤波和饱和必须记录。

### 3.4 泛化、扰动与恢复

- **泛化保持率**：对同任务的 OOD 与 ID 成功率定义 \(R_{gen}=p_{OOD}/p_{ID}\)，并同时报告绝对差 \(p_{OOD}-p_{ID}\) 和两侧区间。ID 接近 0 时比值不稳定，应标记 unavailable/INCONCLUSIVE，不能只报比值。
- **扰动层 Full SR（canonical gate）**：所有分配到该扰动 stratum 的 valid trials 中，最终严格成功的比例。采用 planned-stratum/ITT-style 分母，扰动未成功施加或策略在施加前失败仍保留在该分层的审计与主结果中。
- **恢复状态率**：分母仅为“扰动已按协议成功施加且系统在扰动前未失败”的 eligible trials；分子为在恢复窗内重新达到冻结的“可继续任务”状态的 trial。
- **扰动分层**：按类型、强度和施加阶段报告；无扰动基线与扰动条件尽量共享初态 key 并配对。

### 3.5 安全

安全指标至少包含碰撞、近碰、位置/速度/力/工作空间超限、E-stop、独立 safe-stop、人工接管和保护失效。每个事件必须有预先定义的 trigger、严重度、去重/合并时间窗、传感器证据、动作结果和复位要求。

同一事件同时报告两种暴露分母：`发生事件的 valid trial / valid trial`，以及 `事件数 / 自主运行小时（或运动距离）`。trial 率体现任务风险，暴露率便于不同任务时长比较。零次观测不等于零风险，需报告事件率单侧上界。

近碰必须是可测的代理量，例如人与机器人、末端与禁入区域或物体与障碍物的最小距离低于冻结阈值，不能仅由操作员主观写“差点撞到”。接管既是自主能力失败结果，也是安全人员采取的措施；两种事实应分列。

`[ENG]` 可引用适用法域、机器人厂商安全手册和项目风险分析来制定阈值，但本项目现有资料没有核验具体安全标准条款。任何标准名称、等级或“合规/认证”结论必须由安全负责人基于正式版本另行核对；本文框架本身不等于认证。

### 3.6 系统实时性和资源

时间链路至少分为 observation available、queue、inference、postprocess/safety filter、command sent、controller ack 和 state response。主要报告端到端延迟与各阶段延迟的 p50/p95/p99、最大值、样本数和 deadline miss rate。动作陈旧度定义为命令发送/执行时刻减去生成该动作所依据 observation 的 source timestamp。

资源指标包括 CPU/GPU 利用率、显存/内存峰值、温度、功耗（可得时）、队列深度、掉帧/丢包、推理错误和重启。高频帧不是独立样本；跨模型区间应以 trial/session 为聚类单元，不能把数百万帧当作数百万次独立实验。

### 3.7 指令遵循、失败分类与数据质量

- **指令遵循**：把对象、属性、目标位置、空间关系、顺序、数量、否定约束和停止条件拆成原子约束，报告每项满足率与 all-constraints success。仅完成动作但选错对象不能算严格成功。
- **失败 taxonomy**：建议至少包含 perception/grounding、planning/reasoning、reach/navigation、grasp/contact、transport/manipulation、place/release、instruction/sequence、recovery、timeout/stall、safety、runtime/infrastructure；同时记录首次失败阶段、主要根因、可观测症状和证据。根因未知时保留 `unknown`，不要强行归因。
- **数据质量**：schema/校验和、必需流完整率、序号缺口、视频可解码率、掉帧率、时钟 offset/drift、不合理值、动作链关联率、标注完成率、双标一致率和未裁决分歧数。数据质量失败先影响 validity/gate，而不是用插值悄悄修补。

## 4. Trial 有效性和介入口径

| 情形 | `validity` | 自主结果/状态 | 是否进入主要分母 |
|---|---|---|---|
| 正常执行并成功/失败/partial | `valid` | 对应 outcome | 是 |
| 到冻结 timeout 未成功 | `valid` | `timeout` | 是；按预注册 estimand 作为行政截尾、吸收性失败或复合终点处理 |
| 策略导致碰撞、限位或 watchdog stop | `valid` | `safety_stop` 或 failure | 是，另记 safety outcome |
| 人工为避免风险而接管 | `valid` | `human_takeover` | 是；接管后数据不得算自主成功 |
| 已执行策略后因非预定义原因人工中止 | 通常 `valid` | `failure`，`status=ABORTED`、`end_reason=operator_abort` | 是；不得事后删除不利样本 |
| 初态无法满足容差且策略未获得控制 | `invalid_setup` | `not_evaluable`；无 episode | 否，但计入无效审计 |
| 必需日志损坏，无法判断主要结果 | `invalid_logging` | `not_evaluable` | 否，但数据 gate 可能 BLOCK |
| 预先列明且有证据的外部故障 | `invalid_external` | `not_evaluable`，`end_reason=external_abort` | 否；重试必须新建 trial 并关联 |

`aborted` 是执行/状态语义，不应成为随意排除分母的理由。是否排除只由冻结的 `validity` 规则决定。帧级 intervention 可用于切分自主与人工控制轨迹，但不能把介入帧删掉后把整个 trial 重新称为自主成功。

报告必须给 planned、attempted、valid、invalid（按原因）、timeout、aborted、safety stop、takeover 和重试数量的流转表，使分母可审计。

## 5. 实验设计与标注

1. **随机化/分块**：在 task × scene × object instance × perturbation level 内随机候选/基线顺序；按 session/operator/time block 交错，记录 seed 和实际顺序。若设备维修或重新标定，结束当前 block/session。
2. **配对**：候选和基线共享 matched initial-condition key，包括初始位姿、对象实例、场景和扰动脚本。物理世界不能完全复现时记录布置误差，并在分析中保留 session/block 因子。
3. **盲评**：标注界面隐藏模型名、版本和候选身份，视频外观可能泄漏时记录盲化限制。操作员若无法盲化，至少不能自行挑选下一模型。
4. **双标与裁决**：关键成功/失败阶段/安全标签由两名标注者独立判断；报告原始一致率及类别适合的一致性统计。分歧由第三人按同一 rubric 裁决。Kappa 低或高都不能代替 rubric 正确性审查。
5. **预注册口径**：执行前冻结主要指标、层级、超时、排除规则、异常值处理、停止规则、区间方法和多重比较策略。

## 6. 统计惯例与推荐方法

### 6.1 单模型估计

- 二元比例始终报告 `k/n`、点估计和区间。常规采用 Wilson；极小样本、`k=0`/`k=n` 或安全稀有事件采用精确二项。Jeffreys 区间可作为预先指定的贝叶斯替代，但不能看到数据后挑最窄方法。
- 连续量报告 `n`、中位数与 IQR/p90/p95；均值/标准差在分布适合且便于对照时并列。对 trial 内帧聚合后再跨 trial 分析。
- TTS 用成功条件下分布加预注册的全体 time-to-event 分析。仅行政截尾且假设成立时使用 Kaplan-Meier/RMST；takeover、安全 stop、明确失败和 runtime error 按竞争/吸收事件使用 cumulative incidence、cause-specific estimand 或复合终点，不得静默当普通 timeout。
- 稀有安全事件报告 exposure、事件数及单侧上界；多个事件可落在同一 trial 时，同时给 event count 和 event-trial incidence。

### 6.2 候选与基线比较

- 二元配对结果给配对风险差，以及 discordant pairs；检验可用 exact McNemar。未配对时给两比例差/比值与区间，并明确更高混杂风险。
- 连续配对结果给每对差值的中位数/均值、置信区间和效应量；可采用以 matched pair 或 trial 为重采样单元的 bootstrap。长尾严重时用稳健或秩方法作为预注册分析。
- 多任务/多 session 可使用含 task、session、operator/block 的分层或混合效应模型；模型假设、收敛和小样本稳定性必须诊断，不能用复杂模型制造确定性。
- 效应量优先用业务可解释的绝对量：成功率百分点差、TTS 秒差、路径长度百分比变化、事件率差；标准化效应量只作补充。

### 6.3 汇总、多重比较与决策

- **Macro**：先算各 task/stratum 指标，再等权平均，回答“典型任务表现”；**micro**：合并所有 trial，容易被 trial 多的简单任务主导，只作诊断；生产权重汇总必须在计划中冻结权重。
- mandatory stratum 逐一门禁，汇总值不能抵消关键任务或严重安全失败。
- 多个主要指标/分层使用预先指定的层级检验，或 Holm 等 family-wise error 控制；探索性扫描可用 FDR，但必须标为 exploratory。
- 非劣比较用候选-基线效应的预注册单侧界与 margin；“p>0.05”不证明非劣或等效。
- 没达到最低有效 trial/匹配对数、区间跨越决策界、关键分层缺失或模型无法稳定估计时，结果为 `INCONCLUSIVE`。不得因为预算耗尽自动 PASS，也不得用更多帧数替代独立 trial。

## 7. 常见陷阱

- 只报平均成功率，不报任务清单、`k/n`、分层和区间。
- 把接管、安全 stop、超时和不利的 aborted trial 改成 invalid 后重跑。
- 只比较成功 trial 的速度/平滑度，导致低成功模型因“幸存样本少”看起来更高效。
- 用欧氏差直接处理四元数/旋转 6D，或把米、弧度、夹爪量纲相加。
- 以命令轨迹代替实际状态轨迹，或混淆 `policy_action` 与 `command_action`。
- 不冻结滤波/重采样参数，导致 jerk 与抖振可被后处理调参改善。
- 以 `0 collisions observed` 宣称零风险，或以框架检查表宣称符合某项认证。
- 对视频关键安全事件仅用低频抽帧/VLM 判断。
- 从百分比的小数位反推试验数；从论文总体 trial 数推断每任务重复数。

## 8. 来源与证据边界

本稿的论文实例和试验数量来自本项目已完成的 [`01_real_world_precedents.md`](./01_real_world_precedents.md)，其中列出 RT-1、RT-2、Open X/RT-X、Octo、OpenVLA、pi0、pi0.5、RDT-1B、GR00T N1、SmolVLA 和 DROID 的论文/项目来源与证据标签。系统与统计资料来自 [`03_system_practice.md`](./03_system_practice.md)，主要包括：

1. NIST/SEMATECH, *Confidence intervals for proportions*: https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm
2. Google Research, *RLDS Dataset Format*: https://github.com/google-research/rlds
3. Hugging Face, *LeRobotDataset v3.0*: https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3
4. ROS 2 Design, *Managed nodes*: https://design.ros2.org/articles/node_lifecycle.html

NIST 比例区间资料支持比例不确定性处理；RLDS/LeRobot/ROS 资料支持数据终止语义、数据层和运行生命周期。它们均不构成 VLA 真机安全认证规范。本文未核验任何具体机器人安全标准条款，相关阈值与合规声明均需项目安全负责人另行完成风险分析和正式核对。
