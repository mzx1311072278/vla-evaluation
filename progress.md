# 分析进度

## 2026-07-31

- 已确认目标 Parquet 存在，大小约 7.9 MB。
- 已确认当前工作目录不是 Git 仓库。
- 已创建分析计划与发现记录。
- 已枚举数据集伴随文件，确认存在信息、统计、任务、episode 元数据及右腕相机视频。
- 捆绑 Python 读取 Parquet 失败：`ModuleNotFoundError: pyarrow`；下一步检查项目已有环境与读取实现。
- 已读取 `meta/info.json`、`meta/stats.json` 的首段和项目 LeRobot 读取逻辑。
- 已确认数据集规模、10 维动作/状态名称、右腕视频规格和补充字段。
- 系统 Python 同样没有 `pyarrow`，但项目 requirements 锁定了对应版本；继续查找已有虚拟环境或 Parquet CLI。
- 未发现项目虚拟环境、`parquet-tools` 或 `duckdb` CLI；常见 `python3.10/3.13` 命令也不存在。
- 已检索旧报告：提取了 xyz 单位、动作/策略动作关系和在线后处理不可恢复的边界；同时确认旧报告部分统计并非当前 60-episode 文件。
- 已在临时目录成功安装项目锁定版本 `pyarrow==21.0.0`，不会修改项目环境。
- 已成功读取主 Parquet；首次报告因展开全部 episode 统计而被截断，现改为紧凑聚合输出。
- 已完成逻辑列、物理列、null、向量范围、索引连续性、rot6d 几何性质及向量差异验证。
- 直接文本查看二进制文件和 `ffprobe` 路径不可用；已记录并改用 PyArrow/元数据。
- 已验证物理存储占比、视频时间窗映射、episode 时长范围和任务文本。
- 所有分析阶段完成，准备提交逐字段与逐元素说明。

## 2026-08-07：Web 报告内容对齐

- 曾从已验证的报告页美化版本创建临时分支，随后按用户要求切回 `feature/vla-eval-web-vlm-api-backend`；功能实现统一留在 API backend 分支。
- 建立 Markdown 章节/公式与真实 Web HTML 的自动差异检查。
- 首次检查结果：缺失 33 个章节，公式渲染钩子为 0，反馈环按预期失败。
- 确认旧发版报告包含旧数据数值，后续只复用章节结构和可验证定义，不搬运旧结果。
- codegraph 不可用，已切换为 `rg` 加逐文件读取。
- 当前进入章节数据来源追踪阶段。
- 已完整读取 13 章原始发版文档和运行时 Markdown 生成器。
- 确认两者不是同一模板：原始文档包含大量人工/外部背景内容，运行时生成器只有 5 章。
- 初步完成真实来源分类和页面内容优先级判断；下一步追踪现有加载器、profile/provenance、数据集检查结果和环境信息。
- 已查询当前演示 Job、Dataset 和输出目录，确认 VLM 关闭且只有 4 个主报告产物。
- 已确认 native/LeRobot 共用的 session、episode、metrics 加载接口，可以直接用于 Web 报告视图模型。
- 已确认 profile 将 VLM 汇总定义在 `attempt_eval/` 子目录，旧的根目录假设不能作为真实接口。
- 已列出当前没有真实来源的训练、部署、安全和发版门禁字段。
- 已完成原始 13 章逐章来源矩阵，明确动态、版本化定义、部分可用和不可用四类。
- 已确定 Web 打开报告时只读取持久化产物，不重算指标；共享指标定义同时服务 Markdown 与 Web。
- 已确定无 release-gate/审批来源时不生成发版建议，改为显示“未配置自动发版判定”。
- 已确认公式可用本地语义化 HTML 表达，不引入外部 CDN；下载接口需支持 profile 中真实的 `attempt_eval/` 嵌套路径。
- 用户已确认采用“证据优先的完整报告”方案，内容架构设计阶段完成。
- 已完成 `docs/superpowers/plans/2026-08-07-evidence-first-report.md`，按共享指标定义、真实来源视图、嵌套产物、页面渲染和真实验收五个任务拆分。
- 已完成共享指标定义、严格持久化报告视图、嵌套产物下载契约和九章节 Web 报告实现。
- 真实任务 `d9338238-e7b7-4559-870a-7b33153b9823` 浏览器验收通过：4 个 Episode、2 成功 / 2 失败、GSR 50.0%、成功 TTS 2.500 s、平滑度约 0.081，VLM 本次未启用且无执行产物。
- 真实页面包含九个预期章节和 5 条结构化公式，未出现旧报告的 `30.8%`、`Ep 9` 或 `dc67326`。
- 页面列出的 5 个下载入口均返回 200、`Content-Disposition: attachment`，响应字节与落盘文件一致。
- `tests/e2e/test_visual_layout.py` 在 1440x1000 和 390x844 视口通过，覆盖章节可见性、公式容器和页面横向溢出。
- 最终执行 `.venv/bin/pytest -q`：全量通过，1 个既有跳过项；执行 `.venv/bin/ruff check Genie02_report vla_eval tests` 与 `git diff --check`：均通过。

## 2026-08-07：受控本机数据集导入

- 功能在 `feature/vla-eval-web-vlm-api-backend` 实现，`main` 未修改。
- 增加白名单 `LocalSource` 配置、本机路径安全解析、rsync 本机传输、共享发布流程、Web 表单和 Worker 调度；现有 SSH 导入保持兼容。
- Docker transfer Worker 容器内新增 `/mnt/vla-datasets:/mnt/vla-datasets:ro`，部署文档说明了 Worker 主机路径、SMB/NAS 挂载和复制进 inbox 后评测的语义。
- 本机运行配置新增来源 `this-mac`，根目录为 `/Users/xueyg/Downloads/fangdianlang_data`；现有 session secret 和 `remote_sources` 未改变。
- 安装并校验 GNU rsync 3.4.4；数据盘可用空间 656 GiB，Redis 健康，Web、transfers Worker、evaluations Worker 均从当前 worktree 重启。
- 通过真实 HTTPS 登录和导入表单提交来源 `this-mac`、相对路径 `fangdianlang_good_only_ee`。原目标名已存在，为避免覆盖，验收目标使用 `fangdianlang_good_only_ee_local`。
- 导入任务 ID：`d0ab02ee-ad2e-4ec6-a196-0e1b59d51cd1`；生成数据集 ID：`a43a792b-10a5-48da-ada2-3de5f812dc03`。
- 任务最终状态 READY、进度 100%；数据集 kind 为 `lerobot`，Episode 数 199，大小 3,762,623,032 字节。
- 来源、目标和数据库持久化指纹一致：`d1db953119b7edd15335f34573e66b327808290fb449566b4232363d6f59d912`，证明复制结果完整且来源保持不变。
- 聚焦回归覆盖配置、路径安全、共享导入、Web 和 Worker 调度，全部通过。
- 最终执行 `.venv/bin/pytest -q`：全量通过，1 个既有跳过项；执行 `.venv/bin/ruff check Genie02_report vla_eval tests`、`git diff --check` 和 Compose YAML/挂载断言：均通过。

## 2026-08-07：数据集与评测任务列表管理

- 功能继续在 `feature/vla-eval-web-vlm-api-backend` 实施，`main` 未修改，无关未跟踪 `uv.lock` 未纳入提交。
- 实施提交：`37d333f` 列表参数契约、`73ac002` 归档持久化表、`707797d` 数据集归档/恢复、`d72f2cc` 评测任务归档/恢复、`de92af4` 搜索排序与组合筛选、`2dd1153` 响应式工具栏与页面操作；设计/计划提交为 `48d142f` 和 `4569316`。
- 数据集归档复用 `Dataset.status = ARCHIVED`，并保存 `previous_status`/`archived_at`/`archived_by`；恢复仅接受完整且原状态为 `READY` 或 `PREFLIGHT_FAILED` 的快照。
- 新增 `evaluation_job_archives` 表：`evaluation_job_id` 主键并对任务 `ON DELETE CASCADE`，`archived_at` 为 UTC，`archived_by` 对用户 `ON DELETE SET NULL`；任务原始状态、参数、来源和输出目录不变。
- 两个列表默认排除已归档记录，`archived=1` 同时显示活动和已归档记录；支持不区分大小写的包含搜索、`newest`/`oldest`/`name_asc`/`name_desc` 稳定排序，评测页的原状态和数据集筛选可与新条件组合。
- 列表与详情页已增加 Lucide 归档/恢复图标、归档标记、原生确认框和窄屏堆叠工具栏；活动任务不渲染归档提交按钮。
- 重启真实 Web 和两个 RQ worker 后，`https://127.0.0.1:8443/health` 返回 `{"status":"ok"}`；`transfers` 和 `evaluations` worker 均为 `idle`，Redis 原进程未重启。
- SQLite 启动初始化已创建新表，真实元数据检查确认 3 列、主键与两条设计外键均存在。
- 真实数据集 `a43a792b-10a5-48da-ada2-3de5f812dc03` 可逆验收通过：归档后默认搜索消失、开启已归档后出现，恢复后状态回到 `READY`；29 个文件的大小/mtime 清单和指纹 `d1db953119b7edd15335f34573e66b327808290fb449566b4232363d6f59d912` 前后一致。
- 真实成功任务 `d9338238-e7b7-4559-870a-7b33153b9823` 可逆验收通过：归档前后报告 HTML 和当时页面列出的 4 个下载响应字节/`Content-Disposition` 一致，恢复后归档记录已删除。
- 真实 HTTPS 验收对数据集和评测列表都执行了部分关键词搜索和四种排序；自动化 Chromium 在 1440x1000 和 390x844 视口确认工具栏、控件、操作区无页面级横向溢出或内容越界。
- 最终只运行一次全量验证：`.venv/bin/pytest -q` 100% 通过，1 个既有跳过项；`.venv/bin/ruff check Genie02_report vla_eval tests` 输出 `All checks passed!`；`git diff --check` 退出 0。
- 既有依赖仍输出 `StarletteDeprecationWarning`（FastAPI TestClient 的 httpx 兼容过渡），本次功能无新增警告或失败。

## 2026-08-07：平滑度趋势图优化

- 功能继续在 `feature/vla-eval-web-vlm-api-backend` 实施，`main` 未修改，无关未跟踪 `uv.lock` 未纳入修改。
- 已确认旧 SVG 固定为 820px，且每个柱最少 10px、间隔 8px；199 条数据所需宽度远超画布，导致后续 Episode 被裁切并伴随标签重叠。
- 新生成的 `smoothness_curve.svg` 使用固定画布折线散点图，保留全部有效 Episode，横轴最多 12 个标签，并使用成功、失败和遥操介入三种可区分标记。
- Web 报告直接从现有 `episode_metrics.csv` 构建交互趋势，不要求历史任务重新评测；默认显示全部数据，可切换 100/50/25 条窗口并拖动起始位置。
- 鼠标悬停或键盘聚焦可读取 Episode、六位平滑度、结果和遥操介入；页面同时显示中位数、最近秩 P90、最大值及平滑度最高的 10 个 Episode。
- 折线图纵轴按当前最小值和最大值自动留出 10% 边距，避免 4.5 至 5.0 一类窄范围数据被零基线压缩；平滑度计算公式和 `metrics_core.json` 未改变。
- 历史 SVG、CSV 和其他报告下载接口保持不变；未来评测生成新版 SVG，历史任务页面也可通过 CSV 获得新版交互总览。
- 真实任务 `d7a7e869-4897-4211-8ead-558e0162226a`（199 Episode）验收：默认 199 个点、10 个横轴刻度、中位数 4.833、P90 4.894、最大值 4.976，异常表 10 行。
- 区间验收切换为 50 条并拖到末尾后准确显示 Episode 149–198；悬停 Episode 149 显示 `S 4.896401`，与持久化 CSV 一致。
- 1440x1000 桌面宽度无页面横向溢出；390x844 手机宽度下图表和异常表分别局部滚动，页面主体无横向溢出；浏览器无新增 error/warn 日志。
- 最终执行 `.venv/bin/pytest -q`：100% 通过，1 个既有跳过项；`.venv/bin/ruff check Genie02_report vla_eval tests` 与 `git diff --check` 均通过。

## 2026-08-07：报告状态标题优化

- 将报告中偏技术化的“本次证据状态”改为“本次评测数据与产物状态”，不改变任何指标、数据来源或状态判定逻辑。
- 报告页自动化测试同时校验新标题存在且旧标题不再出现。

## 2026-08-07：平滑度趋势图横轴修复

- 修复 50 条等分窗口中横轴末端两个 Episode 标签重叠：根因为旧算法在步长刻度后额外追加窗口终点。
- 横轴现在基于当前滑动窗口，在首尾之间均匀生成最多 10 个刻度；滑块移动后标签与范围文字同步更新。
- 新增 199 条数据的 Playwright 回归验收，覆盖 50 条窗口从 `Episode 0–49` 滑到 `Episode 149–198`、首尾刻度更新和相邻标签间距。

## 2026-08-07：项目 README

- 新增根目录 `README.md`，以“本机开发”和“Ubuntu 4090 Docker Compose”两条路径介绍项目、环境、配置、启动、数据导入、评测、VLM 与部署验收。
- 文档明确无默认账号、本地路径的 Worker 语义、共享数据范围、归档行为和密钥安全边界。
- 已使用隔离的临时数据目录实际验证配置加载、建库、创建用户、Redis smoke、Uvicorn 启动、`/health` 和登录页；当前开发机无 Docker/4090，容器 GPU 验收命令保留给 Ubuntu 宿主机执行。
