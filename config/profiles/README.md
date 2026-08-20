# ⚠️ 这是模板目录，不是运行目录

**app/config/profiles/这里面的 profile 不会生效。改这里 = 白改。**

## 两个目录的分工

| 目录 | 角色 | 生效？ |
|---|---|---|
| `app/config/profiles/`（本目录） | git 仓库内的**模板库**，随代码提交 | ❌ 不生效 |
| `data/profiles/`（`/czj/code/vla-evaluation/data/profiles`） | **运行时目录**，服务实际读取 | ✅ 生效 |

运行中的 Web / 评测 Worker 通过环境变量 `VLA_EVAL_PROFILES_ROOT` 指向
`data/profiles`（见 `deploy/start-all.sh` 的 `PROFILES_ROOT` 默认值）。

## 为什么分成两个目录（不要合并）

**1. 本机配置不属于版本库。** 运行副本携带每台机器各不相同的值
（model_path 指向本机模型、API 端点、采样调优）。代码目录会被
git pull / 重新部署 / 容器重建整体重写——即使整台机器只有你一个人
有写权限，一次代码更新照样会覆盖这些本机改动。模板（默认值，随代码
分发）与运行副本（本机事实，操作员维护）因此分开存放。类比 nginx 的
`nginx.conf.example` 与 `/etc/nginx/nginx.conf`。

**2. 外部数据进不来 profiles 所在的角落。** 他人数据集只能通过服务的
transfer 管道进入：staging → 指纹/结构校验 → inbox，全程被当作"数据"
对待，外部无法直接向 `data/` 任意写文件。profiles 位于 `data/` 内
操作员管理的独立子目录（目录 700、文件 600），与接收外来数据的
inbox/staging 角色分离；服务每次读取 profile 还会验证它是常规文件、
整条路径无符号链接（`tasks.py` 的 `_trusted_profile_path`）。

**3. 一套布局两种环境通用。** 共享存储部署上，`data_root_boundary`
模式要求服务读取的路径落在 data_root 内（祖先权限由存储平台负责）；
个人电脑上把 profiles 放代码目录虽然技术上也能通过校验，但保持同一
布局能让运维手册、部署脚本和 provenance 记录在两种环境完全一致。

代价是模板更新后要手工同步到运行目录——这是用少量维护成本换
"代码变更"与"配置变更"互不干扰，不是历史遗留，**不要合并成单目录**。

## 正确的改法

1. **调整某台机器的评测配置**（如 model_path、采样参数）：
   改 `data/profiles/` 下的文件。改动即时生效（每次评测任务加载），无需重启服务。

2. **新增模型/新增一套通用配置**：先在两处各放一份并保持一致——
   模板写到本目录（进版本库），运行副本放到 `data/profiles/`。

3. **代码更新带来新模板，要同步到运行目录**：
   ```bash
   cp app/config/profiles/*.yaml data/profiles/
   ```
   注意：会覆盖运行副本中已修改的 `model_path` 等字段，同步前先备份：
   ```bash
   cp -a data/profiles data/profiles.backup
   ```

## 常见踩坑

- 只改了本目录 → 服务完全无感知，评测仍用 `data/profiles` 的旧配置
- 只把新文件放进本目录 → Web 页面上看不到这个 profile
- `backend: api` 的 profile 不能设 `model_family`；`backend: local` 的
  `model_family` 必须与 checkpoint `config.json` 的 `model_type` 一致
  （合法值：`qwen2_5_vl` / `qwen3_vl` / `qwen3_5`）

## 验证服务实际用的是哪个目录

```bash
tr '\0' '\n' < /proc/<uvicorn_pid>/environ | grep VLA_EVAL_PROFILES_ROOT
```
