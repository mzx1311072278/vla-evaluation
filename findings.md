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
