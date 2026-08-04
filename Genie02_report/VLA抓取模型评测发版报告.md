# VLA抓取模型评测发版报告

文档编号：EM-OP-RELEASE-20260707-001

版本：V1.1

日期：2026-07-07

评测类型：☑ VLA抓取策略评测 □ RL 模型单独评测 □ 联合评测

## 1. 报告概述

本报告仅使用 `local_zqyh_2cm_mixed_ee_rot6_right_arm_only` 这一数据集及其派生报告作为评测结果来源，保证模型、数据集、任务和指标口径统一。报告中缺失的模型架构、默认训练配置等背景信息，仅参考 `Evo-RL` 中 Pi0.5 / LeRobot 相关实现，不引入其他评测目录的结果。

评测任务：

> Place the medicine in front of your arm into the basket.

适用任务边界：Genie02 右腕相机、右臂末端位姿控制的桌面级单物体抓取与放置任务。

不适用场景：柔性物体抓取、精密装配、多步骤复杂装配、强 OOD 泛化与碰撞/力控闭环安全验证。

## 2. 依据与数据来源

| 来源 | 路径 | 用途 | 口径 |
| --- | --- | --- | --- |
| 主评测报告 | `report/report.md` | GSR、TTS、平滑度、Episode 明细 | 仅 local 数据集 |
| 主评测指标 | `report/metrics_core.json`、`report/episode_metrics.csv` | 汇总指标与逐条 episode 指标 | 仅 local 数据集 |
| 主评测数据集 | `local_zqyh_2cm_mixed_ee_rot6_right_arm_only/` | LeRobot 原始数据、右臂动作/状态/视频/介入标记 | 仅 local 数据集 |
| Evo-RL 代码 | `../Evo-RL` | Pi0.5 架构、LeRobot 策略接口、默认训练配置 | 仅补充背景，不作为评测结果 |

Evo-RL 当前本地信息：

*   代码路径：`../Evo-RL`
    
*   分支：`main`
    
*   commit：`dc67326`
    
*   Pi0.5 实现：`../Evo-RL/src/lerobot/policies/pi05/`
    

### 2.1 核心结论摘要

| 项 | 结论 |
| --- | --- |
| 发版建议 | 建议暂缓生产发版 |
| 主要依据 | local 数据集原始 GSR 为 30.8%，剔除疑似异常 Ep 9 后为 25.0% |
| 关键风险 | 成功样本少、Ep 9 疑似误标、缺少 OOD/鲁棒性/安全测试 |
| 下一步准入条件 | 完成 Ep 9 核验、补齐模型权重与训练信息、补测不少于 30 条同口径真机 rollout |

## 3. 公共依赖与环境

### 3.1 运行环境

| 项 | 当前记录 |
| --- | --- |
| 报告生成目录 | `.` |
| 报告生成 Python | 3.13.9 |
| OS / Kernel | Linux 6.17.0-35-generic |
| LeRobot / Evo-RL 代码 | `../Evo-RL` |
| 主数据 codebase\_version | `v3.0` |
| 主数据 robot\_type | `genie02` |
| PyTorch / CUDA / cuDNN | 未在 `local_zqyh_2cm_mixed_ee_rot6_right_arm_only` 数据集中记录 |

### 3.2 配套模块

| 模块 | 当前记录 |
| --- | --- |
| VLA 策略类型 | LeRobot Pi0.5 / VLA 系列，具体由 collector policy 标识推断 |
| 主评测策略标识 | `zqyh_2cm_mixed_ee_pi05_stage2_acp` |
| 数据后端 | LeRobot parquet + MP4 |
| 评测脚本 | `genie02_eval_report.py`、`genie02_episode_metrics.py`、`genie02_metrics_core.py`、`genie02_markdown_report.py` |
| 日志/数据采集 | `data/chunk-000/*.parquet`、`videos/observation.images.right_wrist/chunk-000/*.mp4` |

### 3.3 硬件平台

| 项 | 当前记录 |
| --- | --- |
| 机器人 | Genie02 |
| 控制对象 | 右臂 + 右夹爪 |
| 相机 | `observation.images.right_wrist` |
| 图像尺寸 | 480 × 640 × 3 |
| 采样频率 | 30 FPS |
| 右臂状态 xyz 范围 | x=\[0.3134,0.5481\]，y=\[-0.4024,-0.0873\]，z=\[1.0219,1.2311\] |
| 夹爪状态范围 | \[-0.7850, 0.0000\] |
| 力传感器 / 标定方式 / 工控机 | 未在 local 数据集中记录 |

### 3.4 模型输出后处理与安全策略

local 数据集中记录的是采集后的 action/state/video 数据，未包含完整在线控制配置。因此本节只记录可由 local 数据直接确认的信息：

| 项 | 当前记录 |
| --- | --- |
| 动作表示 | 10 维右臂 EE rot6d + 右夹爪 |
| 状态表示 | 10 维右臂 EE rot6d + 右夹爪 |
| 策略动作字段 | `action`、`complementary_info.policy_action` |
| 遥操/介入字段 | `complementary_info.is_intervention`、`complementary_info.collector_policy_id` |

## 4. 模型版本与清单

### 4.1 模型版本标识

| 项 | 当前记录 |
| --- | --- |
| 模型名称 | Genie02 VLA 右臂抓取/放置策略 |
| 策略标识 | `zqyh_2cm_mixed_ee_pi05_stage2_acp` |
| 采集/执行来源 | `complementary_info.collector_policy_id` |
| 是否包含策略优化阶段 | 未在 local 数据集中记录；策略名包含 `stage2_acp`，仅作为命名信息，不作为训练结论 |
| 发版建议 | 建议暂缓生产发版；进入整改、补测与复核流程 |

### 4.2 组件清单

| 组件类型 | 名称 | 版本 / 路径 |
| --- | --- | --- |
| 数据集 | `local_zqyh_2cm_mixed_ee_rot6_right_arm_only` | `local_zqyh_2cm_mixed_ee_rot6_right_arm_only/` |
| 数据元信息 | LeRobot info/stats/tasks | `meta/info.json`、`meta/stats.json`、`meta/tasks.parquet` |
| 动作数据 | LeRobot parquet | `data/chunk-000/*.parquet` |
| 视频数据 | 右腕 MP4 | `videos/observation.images.right_wrist/chunk-000/*.mp4` |
| 评测脚本 | Genie02 report scripts | `genie02_eval_report.py` 等 |
| 模型配置/权重 | 未在 local 数据集中记录 | 需发版前补齐 |

## 5. 数据溯源

### 5.1 数据集信息

| 名称 | 数据类型 | 用途 | 路径 | 关键统计 |
| --- | --- | --- | --- | --- |
| `local_zqyh_2cm_mixed_ee_rot6_right_arm_only` | LeRobot parquet + MP4 | 本报告唯一评测数据源 | `local_zqyh_2cm_mixed_ee_rot6_right_arm_only/` | 13 episodes，18015 frames，30 FPS |

主评测数据字段：

*   时间与索引：`timestamp`、`frame_index`、`episode_index`、`index`、`task_index`
    
*   图像：`observation.images.right_wrist`，480 × 640 × 3
    
*   状态：`observation.state`，10 维右臂 EE + 夹爪
    
*   动作：`action`，10 维右臂 EE + 夹爪
    
*   策略动作：`complementary_info.policy_action`
    
*   介入：`complementary_info.is_intervention`
    
*   来源策略：`complementary_info.collector_policy_id`
    

10 维动作/状态定义：

1.  `right_ee.x`
    
2.  `right_ee.y`
    
3.  `right_ee.z`
    
4.  `right_ee.rot6d_0`
    
5.  `right_ee.rot6d_1`
    
6.  `right_ee.rot6d_2`
    
7.  `right_ee.rot6d_3`
    
8.  `right_ee.rot6d_4`
    
9.  `right_ee.rot6d_5`
    
10.  `right_gripper.pos`
    

### 5.2 数据质量记录

| 项 | 数值 |
| --- | --- |
| Episode 总数 | 13 |
| 总帧数 | 18015 |
| 原始标注成功 / 失败 | 4 / 9 |
| policy 帧数 | 17883 |
| human intervention 帧数 | 132 |
| human intervention 帧占比 | 0.73% |
| 存在遥操介入的 episode | 5、6、7 |
| 单 episode 中位时长 | 29.267 s |
| 单 episode 平均时长 | 46.159 s |
| 疑似异常 episode | Ep 9：success 但仅 0.200 s / 7 帧 |

重要说明：Ep 9 在原始 LeRobot 元数据中标注为 `success`，但仅 0.2 s，不符合正常任务完成时间。本文同时给出“按原始标注”和“剔除 Ep 9”两种口径；不直接篡改原始标注。

### 5.3 数据预处理流程

| 项 | 当前记录 |
| --- | --- |
| 时间同步 | 按数据集时间戳，30 FPS |
| 图像预处理 | 数据集保存为 480×640；Pi0.5 默认模型输入 224×224 为 Evo-RL 背景信息 |
| 轨迹过滤 | 平滑度计算时过滤 `complementary_info.is_intervention != 0` 的帧 |
| 异常轨迹处理 | 报告中标注 Ep 9 为疑似异常；核心 GSR 默认仍保留原始标注 |
| 数据增强 | 未在 local 数据集中记录 |
| 训练/验证/测试划分 | `splits.train=0:13`；未见独立验证/测试划分 |

## 6. 模型架构信息

本节为 Evo-RL 背景补充，不作为 local 数据集的实测结果。

Evo-RL 中 Pi0.5 说明：π₀.₅ 是基于 OpenPI 适配的 Vision-Language-Action 模型，面向 open-world generalization。与 π₀ 相比，π₀.₅ 使用 `time_mlp_*` 进行 AdaRMS conditioning，tokenizer 最大长度 200，使用离散状态输入，并移除 π₀ 中的 state embedding 层。

### 6.1 总体结构

1.  Vision Encoder：编码 RGB 图像。
    
2.  Language Encoder：编码任务指令。
    
3.  State Encoder：编码机器人状态。
    
4.  Policy / Action Expert：基于 Pi0.5 flow matching 生成动作 chunk。
    
5.  Action Post-processing：由部署系统完成动作反归一化、滤波、限幅和控制接口转换；local 数据集中未记录具体在线参数。
    

### 6.2 输入定义

| 输入类型 | 输入内容 | 维度 / 格式 | 来源 |
| --- | --- | --- | --- |
| RGB 图像 | 右腕相机 | 480×640×3 采集；模型侧 224×224 为 Evo-RL 默认背景 | `observation.images.right_wrist` |
| 语言指令 | 单任务文本 | string | `meta/tasks.parquet` |
| 机器人状态 | 右臂 EE + rot6d + 夹爪 | 10 维 | `observation.state` |
| 深度图 | 未使用 | N/A | local 数据集中无记录 |
| 力 / 力矩 | 未使用 | N/A | local 数据集中无记录 |

### 6.3 输出定义

| 输出类型 | 含义 | 维度 | 后处理 |
| --- | --- | --- | --- |
| 右臂末端位姿 | `right_ee.xyz + right_ee.rot6d_0..5` | 9 | local 数据集中未记录在线后处理参数 |
| 右夹爪 | `right_gripper.pos` | 1 | local 数据集中未记录在线后处理参数 |
| 策略动作 | `action` / `complementary_info.policy_action` | 10 | 用于评测统计 |

## 7. 训练方案与超参数

local 数据集中没有完整训练日志，因此本节仅记录 Evo-RL Pi0.5 默认配置作为背景，不声明当前模型实际训练一定使用这些超参数。

| 项 | Evo-RL Pi0.5 默认 / 背景 |
| --- | --- |
| 策略类型 | Pi0.5 / LeRobot |
| VLM 主干 | PaliGemma / Gemma 系列 |
| 动作专家 | Gemma expert |
| 默认状态/action padding | `max_state_dim=32`，`max_action_dim=32` |
| 默认 optimizer | AdamW |
| 默认 LR | `2.5e-5` |
| 默认 betas | `(0.9, 0.95)` |
| 默认 weight decay | `0.01` |
| 默认 grad clip | `1.0` |
| 默认 scheduler | warmup + cosine decay |
| 默认 warmup / decay | 1000 / 30000 steps |
| 默认 decay LR | `2.5e-6` |
| 当前模型实际训练日志 | 未在 local 数据集中记录 |

## 8. VLA 测评方案

### 8.1 指标定义

| 指标 | 定义 | 本次是否可得 |
| --- | --- | --- |
| GSR | 成功 episode 数 / episode 总数 | 可得 |
| TTS | 成功 episode 的平均耗时 | 可得 |
| 平滑度 | `S=log10(E+1)`，`E=Σ||jerk||²·Δt` | 可得 |
| 遥操介入率 | human intervention 帧数 / 总帧数 | 可得 |
| 碰撞/异常检测率 | 异常能否检测并安全停止 | 不可得 |
| OOD 泛化 | 空间/实例/组合泛化 | 不可得 |
| 鲁棒性 | 干扰恢复、光照/遮挡扰动 | 不可得 |

### 8.2 评测分层覆盖情况

| 层级 | 说明 | 本次覆盖 |
| --- | --- | --- |
| L1 分布内测试 | 与 local 数据集同任务同视角 | 部分覆盖 |
| L2 空间泛化 | 物体位置/姿态变化 | 未形成独立分层 |
| L3 实例泛化 | 同类未见物体 | 未覆盖 |
| L4 组合泛化 | 未见物体 + 未见场景 | 未覆盖 |
| L5 鲁棒性 | 遮挡、光照、干扰 | 未覆盖 |
| L6 安全测试 | 碰撞、力超限、异常停机 | 未覆盖 |

## 9. 测评结果汇总

### 9.1 `local_zqyh_2cm_mixed_ee_rot6_right_arm_only`

| 指标 | 按原始标注 | 剔除 Ep 9 疑似异常后 |
| --- | --- | --- |
| Episode 总数 | 13 | 12 |
| 成功数 | 4 | 3 |
| 失败数 | 9 | 9 |
| GSR | 30.8% | 25.0% |
| TTS（成功） | 43.133 s | 57.444 s |
| 右臂平滑度均值 | 6.518877 | 6.756118 |
| 右臂平滑度最小/最大 | 3.671991 / 8.547116 | 5.864266 / 8.547116 |

Episode 明细：

| Episode | 结果 | 时长(s) | 右臂平滑度 | 有效帧 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 0 | failure | 91.333 | 6.587643 | 2741 |  |
| 1 | success | 118.333 | 6.830845 | 3551 |  |
| 2 | failure | 16.200 | 6.191095 | 487 |  |
| 3 | failure | 7.267 | 5.864266 | 219 |  |
| 4 | failure | 93.667 | 6.545043 | 2811 |  |
| 5 | failure | 104.567 | 6.849841 | 3096 | 有遥操介入 |
| 6 | failure | 59.967 | 6.684299 | 1750 | 有遥操介入 |
| 7 | failure | 19.200 | 8.547116 | 537 | 有遥操介入 |
| 8 | success | 24.733 | 6.243798 | 743 |  |
| 9 | success | 0.200 | 3.671991 | 7 | 疑似异常：success 时长低于 1s |
| 10 | failure | 30.567 | 6.249944 | 918 |  |
| 11 | success | 29.267 | 6.303955 | 879 |  |
| 12 | failure | 4.767 | 8.175571 | 144 |  |

### 9.2 额外可计算诊断指标

| 指标 | 数值 | 说明 |
| --- | --- | --- |
| 遥操介入帧占比 | 0.73% | 132 / 18015 |
| 介入 episode | 5、6、7 | 均为 failure |
| action-state xyz 平均误差 | 0.0123 m | `||action_xyz - state_xyz||` |
| action-state xyz P95 | 0.0418 m | 95 分位 |
| policy\_action-state xyz 平均误差 | 0.0213 m | 原始 policy action 对状态 |
| policy\_action-state xyz P95 | 0.0475 m | 95 分位 |
| policy\_action-state xyz 最大误差 | 1.2885 m | 可能包含填充值/异常动作 |
| 夹爪状态范围 | \[-0.7850, 0.0000\] | `right_gripper.pos` |

## 10. 风险分析

1.  成功率不足：按原始标注 GSR 为 30.8%，剔除 Ep 9 后为 25.0%，未达到生产发版要求。
    
2.  成功样本不足：剔除 Ep 9 后有效成功样本仅 3 条，统计置信度低。
    
3.  Ep 9 异常：0.2 s 的 success 不符合任务语义，需回看视频和标注流程。
    
4.  泛化/鲁棒/安全测试缺失：没有 OOD、干扰恢复、碰撞/力控闭环数据，暂不足以支撑生产准入判断。
    
5.  训练与部署信息缺失：local 数据集中未记录权重路径、训练日志、部署环境和在线后处理参数。
    

## 11. 发版结论

结论：**建议暂缓生产发版**。

建议状态：暂缓正式发版，完成数据核验、补测与准入复核后再提交发版评审。

理由：

*   local 数据集按原始标注 GSR 为 30.8%，剔除疑似异常 success 后为 25.0%。
    
*   Ep 9 存在明显标注/切片异常风险。
    
*   当前评测没有覆盖 OOD 泛化、扰动鲁棒性、碰撞/力控安全测试，生产准入证据不足。
    
*   local 数据集中未记录完整模型权重、训练日志和部署后处理参数。
    

## 12. 下一步建议

1.  回看 Ep 9 视频与 parquet 边界，确认是否为误标；若误标，修正 LeRobot episode metadata。
    
2.  明确 `zqyh_2cm_mixed_ee_pi05_stage2_acp` 对应的模型权重路径、训练步数和 checkpoint 选择依据。
    
3.  至少补充 30 条同模型、同任务、同数据口径的真机 rollout，重新统计 GSR/TTS/平滑度。
    
4.  把介入率、action-state 跟踪误差加入常规报告，帮助定位失败原因。
    
5.  增加最小 OOD 分层：位置扰动、姿态扰动、实例变化各不少于 10 条。
    
6.  增加安全验收：急停、碰撞检测、力阈值、遥操接管恢复。
    
7.  对失败样本按原因分类：未到达、抓取失败、放置失败、夹爪异常、控制抖动、感知失败。
    

## 13. 附录

### 13.1 关键文件

*   `report/report.md`
    
*   `report/metrics_core.json`
    
*   `report/episode_metrics.csv`
    
*   `report/smoothness_curve.svg`
    
*   `local_zqyh_2cm_mixed_ee_rot6_right_arm_only/meta/info.json`
    
*   `local_zqyh_2cm_mixed_ee_rot6_right_arm_only/meta/stats.json`
    
*   `local_zqyh_2cm_mixed_ee_rot6_right_arm_only/meta/tasks.parquet`
    
*   `../Evo-RL/src/lerobot/policies/pi05/README.md`
    
*   `../Evo-RL/src/lerobot/policies/pi05/configuration_pi05.py`
    

### 13.2 评测公式

GSR：

```text
GSR = N_success / N_total
```

TTS：

```text
TTS = mean(t_end - t_start) over successful episodes
```

平滑度：

```text
S = log10(E + 1)
E = Σ ||j_k||² * Δt
j_k ≈ (x_k - 3x_{k-1} + 3x_{k-2} - x_{k-3}) / Δt³
```

### 13.3 未记录项

以下内容未在 local 数据集或 Evo-RL 背景中找到可直接引用的同口径实测值，发版前需补齐：

*   `zqyh_2cm_mixed_ee_pi05_stage2_acp` 对应模型权重路径
    
*   checkpoint 训练步数、训练日志、训练起止时间与选择依据
    
*   部署机 PyTorch / CUDA / cuDNN 精确版本
    
*   在线控制后处理参数、限幅参数、retarget 频率
    
*   工控机与 GPU 型号
    
*   力传感器型号与力控阈值
    
*   标定方式与标定版本
    
*   独立验证集 / 测试集划分
    
*   OOD、鲁棒性、安全测试结果
