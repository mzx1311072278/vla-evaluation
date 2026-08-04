# VLA 真机评测既有惯例

> 调研日期：2026-08-04。本文只把论文、项目页、官方仓库/模型卡作为事实来源；`[PAPER]`、`[CODE]`、`[INFERENCE]`、`[UNKNOWN]` 分别表示论文事实、官方代码事实、本文归纳、来源未报告。论文中的仿真结果与真机结果分开记录。任务数量、试验数量和“task”的定义在不同论文中并不等价，不能直接横向相加。

## 1. 快速结论

| 工作 | 真机评测主形态 | 真机规模与主要泛化轴 | 评测协议中最值得复用的做法 |
|---|---|---|---|
| RT-1 | Everyday Robots 移动操作机器人 | 超过 3,000 条真机 rollout；seen/unseen 指令、干扰物、背景、新厨房、长程 | 将单轴 robustness 与多轴真实厨房场景分开；长程同时报告规划与执行成功 |
| RT-2 | 同类 7-DoF 移动操作机器人 | 约 6,000 条轨迹；seen、未见物体/背景/环境、语义与推理 | OOD 设 easy/hard；语义能力采用同初态 A/B 测试 |
| Open X / RT-X | 6 种真机机器人 | 3,600 次试验；ID 正迁移、RT-2 OOD、跨机器人技能迁移 | 保留各机器人原评测协议和本地基线，同时固定训练混合物 |
| Octo | WidowX、UR5、RT-1 Robot；另在新机器人上微调 | 零样本每机器人 2 任务 x 10 次；微调域每域 20 次 | 分开报告 out-of-box 与 target-data fine-tuning；显式测试新输入/动作空间 |
| OpenVLA | WidowX、Google Robot、Franka | 17 x 10、12 x 5；微调共 129 次 | 相同初态的 A/B；把视觉、运动、物理、语义、语言 grounding 分栏；给逐任务 rubric |
| pi0 | 7 类单臂/双臂/移动平台 | 通常每任务每方法 10 次；短程、灵巧、语言、5--20 分钟长程 | 长程用预定义分项 rubric，同时保留 full success；区分 autonomous HL 与 human HL |
| pi0.5 | 双臂移动操作机器人，新 mock home 与真实家庭 | 标准比较为 4 任务 x 10 次；真实家庭与 mock home；位置数/数据组分消融 | 全部环境在训练外；交错执行模型；公开取消试验规则；报告语言选对率与动作成功率 |
| RDT-1B（扩展） | ALOHA 双臂 | 7 任务；每条件 8 或 25 次 | 对物体、场景、指令、few-shot、精细操作分别设专门任务和子步骤指标 |
| GR00T N1（扩展） | GR-1 人形机器人 | 通常每任务 10 次，机械装箱 5 次；seen/unseen object、关节物体、工业、协作 | 预训练零样本与 post-training 分开；低数据 10% 与全数据并列 |
| SmolVLA（扩展） | SO-100 / SO-101 低成本机械臂 | 4 个真机数据集；ID/OOD 位置、同步/异步速度 | 分项得分、固定时限吞吐和成功率并报；论文未给主表每任务试验数 |
| DROID（扩展） | 标准化 Franka Panda | 6 任务 x ID/OOD x 每方法 10 次 | 50/50 co-training、相同初态分布、实验室/办公室/家庭分层、逐任务明确成功条件 |

`[INFERENCE]` 代表性论文没有形成统一的真机 benchmark。相对稳定的惯例是：成功率为主、每任务 5--20 次较常见；大规模封闭评测会用数千 rollout；OOD 常拆成物体、位置、背景/场景、指令/语义和组合变化；长程任务逐步计分，但完整成功率仍需单独保留。

## 2. 必选代表工作

### 2.1 RT-1

来源：[论文，§6、Appendix D，尤其 pp.8--11、23--28，Tables 2--3、11](https://arxiv.org/abs/2212.06817)；[官方项目页](https://robotics-transformer1.github.io/)（访问日期均为 2026-08-04）。

| 字段 | 可核验信息 |
|---|---|
| 机器人/形态 | `[PAPER]` Everyday Robots 移动操作机器人；训练集来自 13 台机器人。主评测在一个训练“classroom”和两个真实办公室厨房进行。 |
| 任务数量与类型 | `[PAPER]` 主文称 seen 评测“超过 200”条指令：36 pick、35 knock、35 upright、48 move、18 drawer open/close、36 drawer pick/place；21 条新组合指令；30 个干扰物任务、22 个背景任务；15 条长程指令，平均 9.6 步、2.4 个操作技能。Appendix D.1 又按完整任务字典报告 744 seen、53 held-out，粒度与主文不一致，应分别保留，不能声称 744 次独立重复。 |
| 环境/物体 | `[PAPER]` 厨房柜台、抽屉、食品/饮料/餐具等；变化包括物体位置、时段、机器人底座位置、台面纹理、照明、厨房几何。 |
| ID/OOD | `[PAPER]` seen 指令仍随机化初态；unseen 是已见技能与物体的新组合；干扰物为 0--5、9、9 且遮挡；背景为原环境、花纹桌布、新厨房；L1/L2/L3 依次叠加新布局/光照、未见干扰物、新对象/新位置/新任务设置。 |
| 每条件 rollout | `[PAPER]` 总计超过 3,000 条真机 rollout。来源未报告 seen/unseen/robustness 的固定每任务重复数；跨形态 bin-picking 补充实验为 72 次抓取；长程表对应 15 条指令。 |
| 成功判定 | `[PAPER]` 以自然语言任务完成的 success rate 为主；Appendix 给出任务集合，但来源未报告统一逐任务几何阈值或人工标注细则。长程分别报告 planning success 与 execution success。 |
| 人工介入 | `[UNKNOWN]` 标准 rollout 中的接管、复位、超时和异常剔除规则来源未报告。论文 model card 明确“not suitable for interaction with humans”。 |
| 核心指标 | `[PAPER]` seen/unseen/干扰物/背景成功率；L1--L3 成功率；长程规划/执行成功率；推理延迟。仿真仅用于 real-to-sim checkpoint selection，另有混入仿真训练数据的独立消融，不是主真机结果。 |
| 基线与公平性 | `[PAPER]` Gato、BC-Z、BC-Z XL 都用 RT-1 的同一机器人数据训练；Gato 参数缩到 37M、RT-1 为 35M，以满足真机频率。此比较主要隔离架构，不是原论文系统的直接比较。 |
| 公开数据/代码 | `[PAPER/PROJECT]` 项目页公开论文与视频；来源未报告可复现主结果所需的完整数据、权重和官方训练/评测代码。 |
| 已知局限 | `[PAPER]` 背景成功率仅 59%；任务仍只覆盖操作空间的一小部分；未在当前研究设置之外验证。 |

### 2.2 RT-2

来源：[论文，§4、Appendix F--H，尤其 pp.6--9、21--26，Tables 4--6](https://arxiv.org/abs/2307.15818)；[官方项目页](https://robotics-transformer2.github.io/)（访问 2026-08-04）。

| 字段 | 可核验信息 |
|---|---|
| 机器人/形态 | `[PAPER]` 默认 7-DoF 移动操作机器人；55B 模型经云 TPU 以 1--3 Hz 控制，5B 约 5 Hz。Language-Table 是单独的仿真结果。 |
| 任务数量与类型 | `[PAPER]` seen 套件沿用 RT-1 的 200+ 任务；未见物体/背景/环境合计 280+ 任务，主要是 pick/place；另有 symbol、数学/Logo/营养/颜色/多语言、人物识别等 emergent 任务。 |
| 环境/物体 | `[PAPER]` 办公室厨房、厨房水槽、视觉差异更大的办公室桌面；常见包装食品、玩具、器皿、办公物体、图片和符号。 |
| ID/OOD | `[PAPER]` 未见物体、未见背景、未见环境各分 easy/hard；hard 进一步增加难抓物体、视觉差异和新对象。语义测试要求机器人数据中没有相应概念/组合。 |
| 每条件 rollout | `[PAPER]` 全部比较约 6,000 条评测轨迹。Appendix F.2：一般 OOD 指令各运行 1--5 次；emergent 指令各 5 次。来源未给各大类精确总分配。 |
| 成功判定 | `[PAPER]` success rate；来源未报告各任务统一成功阈值和超时。 |
| 人工介入 | `[UNKNOWN]` 接管、救援、异常剔除和人工判定协议来源未报告。 |
| 核心指标 | `[PAPER]` seen success；六种 easy/hard OOD success 及平均；symbol/reasoning/person recognition success。Language-Table Table 1 是仿真，不能并入真机均值。 |
| 基线与公平性 | `[PAPER]` RT-1、VC-1、R3M、MOO 使用完全相同的机器人数据。emergent 测试用 A/B 框架，四个模型依次在完全相同条件执行。模型规模和 web 预训练量仍不匹配，是能力比较而非严格等算力消融。 |
| 公开数据/代码 | `[PROJECT]` 项目页公开论文和视频；来源未报告主模型权重、训练代码和完整机器人数据的公开下载。 |
| 已知局限 | `[PAPER]` 不会凭 web 数据获得机器人数据中没有的新运动；对新动力学、按特定部位抓取、工具使用、折毛巾等精细动作和多层推理失败明显。 |

### 2.3 Open X-Embodiment / RT-X

来源：[论文，§IV--VI，Fig.4、Tables I--II](https://arxiv.org/abs/2310.08864)；[官方项目页](https://robotics-transformer-x.github.io/)；[官方数据/代码仓库](https://github.com/google-deepmind/open_x_embodiment)（访问 2026-08-04）。

| 字段 | 可核验信息 |
|---|---|
| 机器人/形态 | `[PAPER]` 数据仓库覆盖 22 种 embodiment、21 家机构；论文训练实验当时使用 9 种 embodiment，真机评测覆盖 6 种机器人。 |
| 任务数量与类型 | `[PAPER]` 数据集有 527 skills、160,266 tasks。小数据域：Kitchen Manipulation、Cable Routing、NYU Door Opening、AUTOLab UR5、Robot Play；大数据域：Bridge/WidowX 与 RT-1/Google Robot；另复用 RT-2 OOD 和 Bridge 技能到 Google Robot 的 emergent skill 测试。 |
| 环境/物体 | `[PAPER]` 多机构、多相机、多控制方式的真实操作环境；从家居 pick/place 到擦拭、组装、线缆、开门。 |
| ID/OOD | `[PAPER]` ID 正迁移；未见物体/背景/环境；以及“技能在 WidowX 数据中出现、在 Google Robot 数据中未出现”的跨 embodiment 技能迁移。论文没有测试全新机器人 embodiment。 |
| 每条件 rollout | `[PAPER]` 总计 3,600 次真机评测，跨 6 种机器人。各域固定重复数来源未报告。 |
| 成功判定 | `[PAPER]` success rate；小数据域沿用各原论文的评测与机器人，成功定义并未被统一成一个 rubric。 |
| 人工介入 | `[UNKNOWN]` 来源未报告。 |
| 核心指标 | `[PAPER]` 各域成功率、跨域平均；RT-2 generalization；emergent skills；3--10 Hz 真机运行频率。 |
| 基线与公平性 | `[PAPER]` 每域比较原作者方法和只用该域训练的 RT-1；所有 RT-X 评测使用同一机器人数据混合物。RT-2-X 容量远大于 RT-1-X；大数据域结果显示小模型欠拟合，因此不能把容量效果误归因于数据混合。 |
| 公开数据/代码 | `[CODE]` 官方仓库公开标准化数据访问、数据集说明与使用代码；项目页说明发布 RT-1-X 模型。完整 RT-2-X 权重不可公开获取。 |
| 已知局限 | `[PAPER]` 未覆盖感知/执行模态差异很大的机器人；未测试新机器人；没有给出何时会正迁移的判据。 |

### 2.4 Octo

来源：[论文，§IV、Appendix F，Tables I--II、VI--VII](https://arxiv.org/abs/2405.12213)；[官方项目页](https://octo-models.github.io/)；[官方仓库](https://github.com/octo-models/octo)（访问 2026-08-04）。

| 字段 | 可核验信息 |
|---|---|
| 机器人/形态 | `[PAPER]` out-of-box：WidowX 250、UR5、RT-1 Robot；微调评测再覆盖 Franka、ViperX、ALOHA 双臂等，共 9 个真机 setup、4 家机构。 |
| 任务数量与类型 | `[PAPER]` 零样本每机器人选择 2 个预训练内语言任务，含 pick/place、擦桌、开关抽屉。6 个新域微调：peg insertion、coffee、baking、pickup、Coke、双臂拔笔帽，各约 100 条 target demos。 |
| 环境/物体 | `[PAPER]` tabletop、咖啡机、烤面包机、插销、抽屉；重装机器人后产生相机/背景/光照变化。 |
| ID/OOD | `[PAPER]` 零样本任务在训练混合物内，但初始位置、光照、干扰物、相机和背景变化；WidowX 另测 novel objects、novel environments、novel skills。Berkeley Coke 的新 ViperX embodiment 是有 115 条 target demonstrations 后微调，不是 zero-shot。 |
| 每条件 rollout | `[PAPER]` 零样本每任务 10 次，即每机器人 20 次；微调每域 20 次；WidowX 每个泛化轴 2 任务合计 20 次；消融 4 任务合计 40 次。 |
| 成功判定 | `[PAPER]` task success rate；部分任务给操作目标，但来源未报告统一几何阈值、超时和人工判分协议。 |
| 人工介入 | `[UNKNOWN]` 来源未报告。 |
| 核心指标 | `[PAPER]` zero-shot success、goal-image 相对 language 的成功率、微调成功率、不同泛化轴成功率。 |
| 基线与公平性 | `[PAPER]` zero-shot 比 RT-1-X/RT-2-X；微调比同 target data 的 scratch ResNet+Transformer 和 VC-1。所有 Octo 微调使用相同 recipe/hyperparameters。WidowX 的 RT-2-X 数字取自 RT-X 论文，RT-1 Robot 由 RT-2-X 作者代跑，地点与执行方并非全部统一。 |
| 公开数据/代码 | `[CODE]` 公开模型、完整训练/微调代码和 OXE 数据工具。 |
| 已知局限 | `[PAPER]` 对未见技能（flip、精密插入）零样本退化大；腕相机和语言条件数据占比不足；只训练/评估单臂和双臂操作，不含导航/移动操作。 |

### 2.5 OpenVLA

来源：[论文，§5、Appendix B，尤其 pp.7--9、21--31，Tables 4、6--7](https://arxiv.org/abs/2406.09246)；[官方项目页](https://openvla.github.io/)；[官方仓库](https://github.com/openvla/openvla)（访问 2026-08-04）。

| 字段 | 可核验信息 |
|---|---|
| 机器人/形态 | `[PAPER]` direct：BridgeData V2 WidowX、Google mobile manipulator；adaptation：Franka-Tabletop、Franka-DROID。 |
| 任务数量与类型 | `[PAPER]` WidowX 17 任务：5 visual、2 motion、3 physical、4 semantic、3 language grounding；Google 12 任务：5 ID、7 OOD；Franka-Tabletop 6 任务，Franka-DROID 1 个擦桌任务。 |
| 环境/物体 | `[PAPER]` Bridge sink、Google 平台/抽屉、Franka tabletop、DROID 桌面；对象、相机、sink、照明无法完全复刻 Bridge 训练场景，因此 17 个任务都带自然 distribution shift。 |
| ID/OOD | `[PAPER]` 视觉（背景/干扰物/颜色）、运动（位置/朝向/高度）、物理（尺寸/形状）、语义（新对象/指令/互联网概念）、语言 grounding；Franka OOD 分别替换对象、桌布、干扰物、位置/朝向。 |
| 每条件 rollout | `[PAPER]` WidowX：17 x 10 = 170/方法；Google：12 x 5 = 60/方法；Franka-Tabletop 每任务 10--12 ID + 5--6 OOD；Franka-DROID 18 ID + 12 OOD；adaptation 总计 129/方法。 |
| 成功判定 | `[PAPER]` direct 通常 0/1；困难任务允许 0.5，并逐任务写明“接近正确物体、接触、完成抓取但未放置”等条件。Franka-DROID 每次 0/1/2 分，对应扫入 0、1--2、3 个物体。完整成功率与部分进度应分报。 |
| 人工介入 | `[UNKNOWN]` 接管和异常取消规则来源未报告。 |
| 核心指标 | `[PAPER]` mean success rate ± standard error；逐任务成功数；ID/OOD 分栏；微调成功率与训练显存/参数量。 |
| 基线与公平性 | `[PAPER]` direct 比 RT-1-X、Octo、RT-2-X；所有评测以相同任务、相同一组机器人/物体初态做 A/B。adaptation 比同 target data 的 Diffusion Policy、matched DP、Octo、OpenVLA scratch。完整 DP 有历史、proprioception、action chunk 和绝对坐标，matched DP 才匹配 OpenVLA I/O，二者同时报告以揭示控制栈差异。 |
| 公开数据/代码 | `[CODE]` 权重、训练/微调代码、HF 集成公开；训练依赖公开 OXE 组成数据。真机硬件场景和逐次初态 manifest 未完整公开。 |
| 已知局限 | `[PAPER]` 不带 action chunking/temporal smoothing 时精细动作不如 Diffusion Policy；Bridge 环境复刻误差使“ID/OOD”边界不纯；每项仅 5--12 次时不确定性较大。 |

### 2.6 pi0

来源：[论文，§VI、Appendix E，尤其 pp.7--12、16--17](https://arxiv.org/abs/2410.24164)；[官方项目页](https://www.pi.website/blog/pi0)；[官方 openpi 仓库/模型卡](https://github.com/Physical-Intelligence/openpi)（访问 2026-08-04）。

| 字段 | 可核验信息 |
|---|---|
| 机器人/形态 | `[PAPER]` 7 类配置：UR5e、双 UR5e、Franka、双臂 Trossen、双臂 ARX/AgileX、Mobile Trossen/ARX、Mobile Fibocom；覆盖单臂、双臂、全向/非全向移动操作。 |
| 任务数量与类型 | `[PAPER]` out-of-box 5 项（折衣、easy/hard bussing、装购物袋、取吐司）；语言评测 3 项；新灵巧任务 5 项；复杂 5--20 分钟多阶段任务 7 项。论文另称 fine-tuning 覆盖 20+ 任务。 |
| 环境/物体 | `[PAPER]` 厨房/桌面/洗衣/烘干机、微波炉、纸巾架、抽屉、可变形衣物、蛋、纸盒等；预训练私有混合物约 10,000 小时，并含 OXE、Bridge、DROID。 |
| ID/OOD | `[PAPER]` out-of-box 任务在预训练中；新任务按 easy/hard 与预训练相似度分层，评测混合 seen/unseen bowl/container；复杂任务有预训练内和完全新任务。其主线不是像 OpenVLA 那样逐个单变量 OOD。 |
| 每条件 rollout | `[PAPER]` 主要结果均为每任务每方法 10 次；fine-tuning 比较 1/5/10 小时数据。复杂 laundry 明确 5 件物品 x 每件 2 次。 |
| 成功判定 | `[PAPER]` full success=1，按任务预定义部分进度；Appendix E 给逐任务 rubric，例如 bussing=正确分类数/总数，folding 分抓取/展平/折叠/堆放，packing eggs=6 个蛋+关盒共 7 分。 |
| 人工介入 | `[PAPER]` `pi0-human` 条件由人类专家提供中间语言命令，是显式 oracle/辅助条件；`pi0-HL` 由高层 VLM 自动给命令，属 autonomous。标准 out-of-box 与复杂任务中的救援/异常剔除来源未报告。 |
| 核心指标 | `[PAPER]` normalized task progress、full/partial task score、language-following accuracy；长程还隐含最长步数/约 5 分钟超时。 |
| 基线与公平性 | `[PAPER]` OpenVLA、Octo、pi0-small、scratch、ACT、Diffusion Policy。OpenVLA/Octo 用同混合物，但因时间未训练到 pi0 的 700k steps；论文另给 160k-step pi0 parity 版本。新任务基线用相同 target data；模型规模/预训练仍不完全匹配。 |
| 公开数据/代码 | `[CODE]` openpi 公开 pi0/pi0.5 模型代码、checkpoint/模型卡及推理/微调示例；论文的大规模私有机器人数据不公开，公开成分仅为 OXE/DROID/Bridge 等。 |
| 已知局限 | `[PAPER]` 绝对表现随任务难度变化，许多复杂任务仍只有部分完成；论文承认任务覆盖/数据选择规律和跨机器人通用性仍未解决。 |

### 2.7 pi0.5

来源：[论文，§V、Appendix B--C，尤其 Fig.7--13](https://arxiv.org/abs/2504.16054)；[官方项目页](https://www.pi.website/blog/pi05)；[官方 openpi 仓库/模型卡](https://github.com/Physical-Intelligence/openpi)（访问 2026-08-04）。

| 字段 | 可核验信息 |
|---|---|
| 机器人/形态 | `[PAPER]` 两类双臂移动操作平台：双 6-DoF 臂、平行夹爪、前后和双腕 4 相机、全向底盘、升降机构，18/19 维状态动作；端到端 50 Hz，无额外轨迹规划/碰撞检测。 |
| 任务数量与类型 | `[PAPER]` 标准定量 4 项：dishes in sink、items in drawer、laundry in basket、make bed；真实家庭另以厨房/卧室任务验证；语言测试在 2 场景从 5 个干扰对象中选择指定对象。 |
| 环境/物体 | `[PAPER]` 所有评测环境都未进训练：mock kitchens/bedrooms 与 3 个真实家庭的厨房和卧室；训练移动操作数据约 400 小时、约 100 个家庭。 |
| ID/OOD | `[PAPER]` 任务类型可在训练中出现，但场景、布局、背景、对象、配置全新；语言实验再分 seen category 与 unseen category；位置数量从 3/12/22/53/82/104 做扩展消融。 |
| 每条件 rollout | `[PAPER]` 默认每任务每 policy 10 次；标准比较共 40 次/policy，策略交错执行。Appendix 说明 4 任务在真实与 mock 的合计 12 个位置开展不同实验，但并非每个图都覆盖全部位置。 |
| 成功判定 | `[PAPER]` 以 rubric 总分百分比：dishes 8 分、drawer 4、laundry 3、bed 5；语言另报选对对象的 language following rate 和放到目标位置的 task success。 |
| 人工介入 | `[PAPER]` 标准 pi0.5 只接收一个高层人类指令，子任务由模型自主生成；`human HL` 是专家 oracle 基线，不能与 autonomous 结果混称。部分 episode 因机器人故障、时间等取消并剔除，论文控制样本量接近并用双侧不等样本 t-test；取消数未逐条件公开。 |
| 核心指标 | `[PAPER]` task progress、language-following rate、task success；长程时长约 2--5 分钟；消融随训练位置数和数据组分变化。 |
| 基线与公平性 | `[PAPER]` pi0、pi0-FAST+Flow、不同训练混合消融、implicit/no HL、GPT-4 HL、human HL。比较固定对象集合、交错执行模型以控制环境漂移，并尽量控制每个模型看到的唯一训练样本数。 |
| 公开数据/代码 | `[CODE]` openpi 提供 pi0.5 代码/checkpoint/模型卡；约 400 小时家庭数据与完整评测初态/取消清单未公开。 |
| 已知局限 | `[PAPER]` 新抽屉把手、物理难开的柜门、遮挡造成的部分可观测、高层策略反复开关抽屉；只处理相对简单 prompt，短上下文限制跨房间记忆。取消试验剔除也使复现者必须预注册 censoring 规则。 |

## 3. 近期扩展

### 3.1 RDT-1B

来源：[论文，§5、Appendix H，pp.7--10、24--26，Tables 1、3](https://arxiv.org/abs/2410.07864)；[官方项目页](https://rdt-robotics.github.io/rdt-robotics/)；[官方仓库](https://github.com/thu-ml/RoboticsDiffusionTransformer)（访问 2026-08-04）。

`[PAPER]` 在静态 Mobile ALOHA 双臂机器人上测试 7 项：洗杯（1 seen+2 unseen cup，各 8 次）、倒水（3 unseen room，各 8 次）、左手 1/3 和右手 2/3 倒水（各 8 次）、5-shot handover、1-shot fold shorts、robot-dog joystick（后三项各 25 次）。物体随机 3--10 cm、robot dog 最远 50 cm。指标既有全任务 success，也有 Pick/Turn/Get/Pour/Place、correct hand、correct amount 等子步骤；主基线 ACT、OpenVLA、Octo、scratch。`[UNKNOWN]` 人工介入与异常剔除未报告。`[PAPER]` 其“zero-shot”是在大规模预训练+目标 ALOHA 6K+ episode fine-tuning 后，对特定未见对象/房间/词语为零样本，不是新机器人零样本。`[CODE]` 仓库公开代码和模型资源。

### 3.2 GR00T N1

来源：[论文，§4.2--4.4、Appendix Table 5，pp.11--15、26](https://arxiv.org/abs/2503.14734)；[官方仓库](https://github.com/NVIDIA/Isaac-GR00T)（访问 2026-08-04）。

`[PAPER]` 真机为 GR-1 人形机器人，pre-training 直接测两项：左右手交接后上架、未见对象到未见容器，各 5 对象 x 3 次，0.5 表示抓对但未放入。Post-training benchmark 包括 5 个 pick/place、3 个 articulated、3 个 industrial、2 个 multi-agent coordination；通常每任务 10 次，Pack Machinery 因时间只做 5 次并按 30 秒内放入 5 个零件比例计分。对比 Diffusion Policy 的 10%/全数据，任务数据由人类遥操作 15 分钟--3 小时采集。`[UNKNOWN]` 其他任务 partial scoring 的完整逐步 rubric、接管/取消规则未在正文统一列出。`[CODE]` 模型和代码公开。仿真 RoboCasa/DexMimicGen/GR-1 Tabletop 的每项 100 trials 是独立结果，不应并入真机。

### 3.3 SmolVLA

来源：[论文，§4，pp.8--12，Tables 3--5、Fig.5](https://arxiv.org/abs/2506.01844)；[官方发布说明](https://huggingface.co/blog/smolvla)；[LeRobot 官方仓库](https://github.com/huggingface/lerobot)（访问 2026-08-04）。

`[PAPER]` SO-100 测 pick-place、stacking、双物体 sorting，SO-101 测更小 Lego 的 pick-place；每目标数据集 5 个初始位置 x 每位置 10 条 demonstration。真机评分为 0.5 抓取+0.5 放置，sorting 为四个 0.25 子项。SO-101 的 OOD 是训练未见位置；SO-101 embodiment 不在预训练，但仍使用该机器人 target data 微调，不能称 zero-shot embodiment transfer。主表每任务 rollout 数 `[UNKNOWN]` 来源未报告，尽管百分比以 5% 为粒度；异步速度实验明确 10 次、5 个位置，并加 60 秒固定时限吞吐。ACT 和 pi0 为基线；所有数据集与 LeRobot 代码公开。仿真 LIBERO/Meta-World 的 10 trials/task 与真机结果严格分开。

### 3.4 DROID 数据论文中的真机先例

来源：[论文，§V、Appendix E](https://arxiv.org/abs/2403.12945)；[官方项目页](https://droid-dataset.github.io/)；[官方仓库](https://github.com/droid-dataset/droid)（访问 2026-08-04）。

`[PAPER]` DROID 本身是数据集而非 VLA，但给出高度可复用的真机协议：同一 Franka Panda 硬件栈，在实验室/办公室/家庭 4 个地点做 6 任务，各有 ID 和 OOD，且每 task setting、每 method 10 次 A/B。OOD 单独改变新包装、干扰物、camera shift 等；策略看到相似的初始物体位置分布。比较 in-domain only、50/50 混入 DROID、50/50 混入 OXE 的同一 Diffusion Policy。Appendix E 对每项给完整成功条件，例如 apple 在 pot 且 lid 盖好、drawer 关闭且 eraser 在内、Cook Lentils 三阶段全完成。`[UNKNOWN]` 接管/异常剔除规则未报告。该设计是“数据贡献”公平比较的优秀模板。

## 4. 跨工作归纳

1. `[INFERENCE]` **先定义泛化单位再统计。** RT-1 的“任务”可指 noun-verb 组合，pi0 的一个“任务”可持续 5--20 分钟并包含几十个行为；任务数不可直接横比。
2. `[INFERENCE]` **单变量 OOD 与组合 OOD 都需要。** RT-1/RT-2/OpenVLA 先控制物体、位置、背景、环境、指令，再进入真实厨房的多轴组合；pi0.5 直接强调 unseen homes，生态有效但难做因果归因。
3. `[INFERENCE]` **同初态 A/B 和交错执行是最强公平性惯例。** RT-2 emergent、OpenVLA、DROID 用相同初态；pi0.5 进一步交错策略执行以控制照明、机器人温度和场景漂移。
4. `[INFERENCE]` **5--10 次适合快速比较，不足以证明高可靠。** 论文常见 5--20 次/任务；当目标成功率约 50% 时，n=10 的二项置信区间非常宽。发布/采购门槛应提高重复数并报告 Wilson 区间，而不是只报点估计。
5. `[INFERENCE]` **partial score 与 full success 必须并列。** pi0/pi0.5/OpenVLA/RDT/GR00T 用分项分数诊断进度；若只报平均 progress，可能掩盖“从未完整完成”的策略。
6. `[INFERENCE]` **人工帮助是一个实验变量。** pi0-human、pi0.5 human-HL 是 oracle 条件；自主结果必须排除人在 rollout 中选择子任务、扶正物体或接管。安全停止仍应记录，并区分策略导致与基础设施导致。
7. `[INFERENCE]` **zero-shot 标签需写清 target-data 边界。** Octo 的 ViperX、SmolVLA 的 SO-101、RDT 的 unseen scene 都有不同程度的目标机器人/任务 fine-tuning；只能把具体未见变量称为 zero-shot。
8. `[INFERENCE]` **仿真只做补充。** RT-2 Language-Table、OpenVLA LIBERO、GR00T 三套仿真、SmolVLA LIBERO/Meta-World 的结果不等于真机；可用于回归和 checkpoint selection，最终门槛仍以真机为准。

## 5. 未能核实与复现风险

- RT-1、RT-2、RT-X 未公开完整逐 rollout manifest、人工成功标注协议、异常/超时/接管规则；RT-1 主文和 Appendix 的任务计数粒度不一致。
- RT-X 只给 3,600 次总量，未给每个域/任务的固定重复数；不同实验室沿用本地成功标准。
- Octo 的成功阈值和人工判分细则不完整；部分 RT-2-X 比较来自别处或由另一团队代跑。
- pi0 的私有 10,000 小时混合物、pi0.5 的家庭数据和完整评测初态不公开；无法从公开仓库复刻论文数据条件。
- pi0.5 剔除了基础设施/时间等取消 episode，但未逐条件给取消数；复现时必须公开原始分母和 censoring 明细。
- SmolVLA 主真机结果表未明确每任务 rollout 数；不能由百分比粒度反推后当作事实。
- 以上论文普遍没有统一报告碰撞、力限、急停、接管次数、复位时间和每小时有效任务数；工程评测必须补齐这些运营与安全指标。
