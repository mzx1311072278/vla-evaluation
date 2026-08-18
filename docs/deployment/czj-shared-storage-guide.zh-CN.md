# VLA 评测系统服务器使用手册（`/czj` 共享存储）

本文适用于当前服务器部署：

- 代码：`/czj/code/vla-evaluation/app`
- 配置：`/czj/code/vla-evaluation/config/app.yaml`
- Conda 环境：`/czj/envs/vla-eval`
- 运行数据：`/czj/code/vla-evaluation/data`
- 待导入数据集：`/czj/code/vla-evaluation/datasets`
- 评测配置：`/czj/code/vla-evaluation/data/profiles`
- Web 端口：服务器 `8000`，Mac 隧道端口 `18000`
- Redis：`127.0.0.1:6379`

## 1. 日常启动

系统需要保持四个终端：服务器三个，Mac 一个。

### 服务器终端 1：Web

```bash
conda activate /czj/envs/vla-eval
cd /czj/code/vla-evaluation/app
export VLA_EVAL_CONFIG=/czj/code/vla-evaluation/config/app.yaml
export VLA_EVAL_PROFILES_ROOT=/czj/code/vla-evaluation/data/profiles

uvicorn vla_eval.server:create_app_from_env \
  --factory --host 0.0.0.0 --port 8000
```

成功标志：

```text
Uvicorn running on http://0.0.0.0:8000
```

### 服务器终端 2：数据导入 Worker

```bash
conda activate /czj/envs/vla-eval
cd /czj/code/vla-evaluation/app
export VLA_EVAL_CONFIG=/czj/code/vla-evaluation/config/app.yaml

python -m vla_eval.cli worker --queue transfers
```

成功标志：

```text
Listening on transfers...
```

### 服务器终端 3：评测 Worker

```bash
conda activate /czj/envs/vla-eval
cd /czj/code/vla-evaluation/app
export VLA_EVAL_CONFIG=/czj/code/vla-evaluation/config/app.yaml
export VLA_EVAL_PROFILES_ROOT=/czj/code/vla-evaluation/data/profiles

CUDA_VISIBLE_DEVICES=0 \
python -m vla_eval.cli worker --queue evaluations
```

成功标志：

```text
Listening on evaluations...
```

启动前用 `nvidia-smi` 确认 GPU。若 GPU 0 被占用，将 `CUDA_VISIBLE_DEVICES=0`
改成空闲的 GPU 编号。

### Mac 终端 4：SSH 隧道

如果 Mac 的 `~/.ssh/config` 已配置服务器别名 `czj`：

```bash
ssh -o ExitOnForwardFailure=yes \
  -N -L 18000:127.0.0.1:8000 czj
```

终端没有输出并一直占用是正常现象。浏览器打开：

```text
http://127.0.0.1:18000/login
```

如果没有 SSH 别名，使用完整登录参数：

```bash
ssh -o ExitOnForwardFailure=yes \
  -N -L 18000:127.0.0.1:8000 \
  -p <SSH端口> <用户名>@<服务器IP或域名>
```

隧道失败时增加 `-v` 查看原因：

```bash
ssh -v -o ExitOnForwardFailure=yes \
  -N -L 18000:127.0.0.1:8000 czj
```

## 2. 日常停止和重启

在对应终端按 `Ctrl+C` 正常停止。修改代码或配置后，建议重启三个服务器进程。

检查是否有重复进程：

```bash
ps aux | grep -E '[u]vicorn|[v]la_eval.cli worker'
```

正常情况下只应有：

- 一个 Uvicorn Web 进程；
- 一个 `--queue transfers` Worker；
- 一个 `--queue evaluations` Worker。

若原终端已经丢失，先用上面的命令找到 PID，确认后正常停止：

```bash
ps -fp <PID>
kill <PID>
```

不要同时保留新旧 Worker，否则旧 Worker 可能抢到新任务。

## 3. 系统健康检查

```bash
conda activate /czj/envs/vla-eval
cd /czj/code/vla-evaluation/app
export VLA_EVAL_CONFIG=/czj/code/vla-evaluation/config/app.yaml

python -m vla_eval.cli smoke
```

成功标志：

```text
smoke ok
```

也可以在 Web 启动后检查：

```bash
curl -fsS http://127.0.0.1:8000/health
```

成功输出：

```json
{"status":"ok"}
```

## 4. 导入数据集

### 4.1 放置原始数据

每个数据集放在独立目录中，例如：

```text
/czj/code/vla-evaluation/datasets/zqyh_barcode_multi_ae
```

Web 导入页面选择：

- 数据源：`server-datasets`
- 根目录：`/czj/code/vla-evaluation/datasets`
- 相对路径：`zqyh_barcode_multi_ae`
- 目标名称：建议与数据集目录名一致

“相对路径”只填写根目录下面的部分，不能填写完整绝对路径。

### 4.2 导入结果

导入过程由 transfer Worker 完成：连接、传输、验证、预检、发布。成功数据集位于：

```text
/czj/code/vla-evaluation/data/inbox/<目标名称>
```

同名目标已经存在时，系统会故意拒绝覆盖：

```text
FileExistsError: import target already exists
```

这不是服务故障。进入 Web 的“数据集”页面，直接使用已有数据集，不要重复导入。

如果目录已存在但 Web 列表没有记录，可执行：

```bash
python -m vla_eval.cli scan-datasets
```

然后刷新数据集页面。

## 5. 创建评测任务

1. 打开 Web 的“数据集”页面；
2. 选择状态为 `READY` 的数据集；
3. 创建评测任务；
4. 选择 `genie02-full` 使用 Qwen2.5-VL，选择 `genie02-qwen3-vl` 使用
   Qwen3-VL-8B-Instruct，或选择已配置好的 API profile；
5. 在 evaluation Worker 终端观察运行日志；
6. 完成后从“评测任务”进入报告页面。

评测输出位于：

```text
/czj/code/vla-evaluation/data/runs/<评测任务ID>
```

## 6. 评测 Profile 和模型

共享存储部署必须使用：

```bash
export VLA_EVAL_PROFILES_ROOT=/czj/code/vla-evaluation/data/profiles
```

Web 和 evaluation Worker 都要设置这个变量：Web 用它展示并校验可选 profile，
evaluation Worker 用它执行评测。transfer Worker 不读取 profile。

初始化或代码更新后同步 profile：

```bash
mkdir -p /czj/code/vla-evaluation/data/profiles

cp /czj/code/vla-evaluation/app/config/profiles/*.yaml \
  /czj/code/vla-evaluation/data/profiles/

chmod 700 /czj/code/vla-evaluation/data/profiles
chmod 600 /czj/code/vla-evaluation/data/profiles/*.yaml
```

本地 VLM 的实际模型路径分别在以下文件中配置：

```text
/czj/code/vla-evaluation/data/profiles/genie02-full.yaml
/czj/code/vla-evaluation/data/profiles/genie02-qwen3-vl.yaml
```

查看当前配置：

```bash
grep -nE 'model_family|model_path' \
  /czj/code/vla-evaluation/data/profiles/genie02-{full,qwen3-vl}.yaml
```

### 任务级摄像头选择和 4090 冒烟测试

新建评测时，页面会展示数据集导入检查得到的摄像头列表。选择会保存到任务快照中，Worker 执行时使用快照，不会因为页面刷新或数据集目录后来增加视角而改变任务含义。

- 不选择时默认分析数据集全部摄像头，但单个任务最多允许 3 路；数据集超过 3 路时必须明确勾选不超过 3 路。
- 同一 Episode 的多路图片在一次 VLM 请求中联合分析。每路独立按当前抽帧上限采样，默认最多 16 帧，因此三路最多 48 张图片。
- 本地 VLM 在 Processor 后读取真实输入 token 数，并在 `input_tokens + max_new_tokens` 超过 checkpoint 的 context limit 时跳过生成；Episode 结果会标记 `context_length_exceeded`。
- 本地后端记录每个 Episode 的 CUDA allocated/reserved 峰值；API 后端的 token 和 CUDA 观测字段为 `null`。

先准备只包含 1 个 Episode 的测试数据集，勾选 3 路摄像头运行一次。检查：

```bash
docker compose logs --tail=200 evaluation-worker
jq '{camera_keys, sampled_frame_count_by_camera, input_token_count, context_token_limit, cuda_peak_memory_allocated_bytes, cuda_peak_memory_reserved_bytes}' \
  /czj/code/vla-evaluation/data/runs/<评测任务ID>/attempt_eval/episode_results/episode_000.json
```

确认三路都出现在 `camera_keys`，每路抽帧数量符合预期，且 CUDA 峰值字段为非负整数后，再去掉 `limit` 扩大评测范围。上下文预算通过只表示 token 数安全；显存还会受到视觉编码、Prefill 激活和 CUDA workspace 影响，必须以这次真实 smoke test 的峰值为准。

查找服务器上的模型：

```bash
find /czj -type d \( \
  -name 'Qwen2.5-VL-7B-Instruct' -o \
  -name 'Qwen3-VL-8B-Instruct' \
\) 2>/dev/null
```

将 profile 中旧的 `/srv/vla-eval/...` 改成查到的实际 `/czj/...` 路径，然后重启
evaluation Worker。

如果使用 `genie02-api.yaml`，需要修改其中的 `base_url`、`model`，并在 evaluation
Worker 启动前设置对应 API Key：

```bash
export VLA_EVAL_VLM_API_KEY='<API Key>'
```

不要把 API Key 写入 Git、profile YAML 或终端截图。

## 7. 当前服务器配置示例

`/czj/code/vla-evaluation/config/app.yaml` 的关键内容如下。保留服务器已有的
`session_secret`，不要用示例值覆盖：

```yaml
data_root: /czj/code/vla-evaluation/data
storage_trust_mode: data_root_boundary
database_url: sqlite:////czj/code/vla-evaluation/data/db/app.sqlite3
redis_url: redis://127.0.0.1:6379/0
session_secret: "<保留服务器现有值或环境变量占位符>"

local_sources:
  server-datasets:
    roots:
      - /czj/code/vla-evaluation/datasets

remote_sources: {}
```

修改该文件后至少重启 Web 和两个 Worker，保证三个进程读取同一份配置。

## 8. 目录与权限

推荐目录结构：

```text
/czj/code/vla-evaluation/
├── app/                 # Git 代码仓库
├── config/
│   └── app.yaml         # 私有运行配置
├── datasets/            # 待导入的原始数据集
├── data/
│   ├── db/              # SQLite 数据库
│   ├── inbox/           # 已发布的数据集
│   ├── staging/         # 导入中间目录
│   ├── runs/            # 评测结果
│   ├── profiles/        # 运行时评测配置
│   └── credentials/     # 凭据（如使用）
├── models/              # 模型（也可在其他指定 /czj 路径）
└── logs/
```

应用管理的目录不能被 group/other 写入：

```bash
chmod 700 \
  /czj/code/vla-evaluation/data \
  /czj/code/vla-evaluation/data/db \
  /czj/code/vla-evaluation/data/inbox \
  /czj/code/vla-evaluation/data/staging \
  /czj/code/vla-evaluation/data/runs \
  /czj/code/vla-evaluation/data/profiles
```

不要修改共享目录 `/czj` 或 `/czj/code` 的权限。系统通过：

```yaml
storage_trust_mode: data_root_boundary
```

将 `/czj/code/vla-evaluation/data` 作为应用负责的安全边界，但仍严格检查
`data_root` 及其所有子目录。

## 9. 更新服务器代码

PR 合并到 GitHub `main` 后：

```bash
cd /czj/code/vla-evaluation/app
git switch main
git pull --ff-only
git log -1 --oneline
```

如 profile 模板也有更新，再同步到运行目录，并重新确认本地模型路径：

```bash
cp config/profiles/*.yaml /czj/code/vla-evaluation/data/profiles/
chmod 600 /czj/code/vla-evaluation/data/profiles/*.yaml
grep -n model_path /czj/code/vla-evaluation/data/profiles/genie02-full.yaml
```

注意：复制模板会覆盖运行目录中的 profile 修改。同步前可以先备份：

```bash
cp -a /czj/code/vla-evaluation/data/profiles \
  /czj/code/vla-evaluation/data/profiles.backup
```

更新后重启三个服务器进程。通常不需要重新创建 Conda 环境；只有依赖变化时才执行：

```bash
python -m pip install -e '.[dev,gpu,vlm-api]'

python -c "import torch, torchvision, transformers, qwen_vl_utils; \
assert torch.cuda.is_available(); print(transformers.__version__, torch.cuda.get_device_name(0))"
```

## 10. 常见故障

### `KeyError: VLA_EVAL_CONFIG`

当前终端没有配置文件环境变量：

```bash
export VLA_EVAL_CONFIG=/czj/code/vla-evaluation/config/app.yaml
```

或者给 CLI 显式传递：

```bash
python -m vla_eval.cli smoke \
  --config /czj/code/vla-evaluation/config/app.yaml
```

### `address already in use`（端口 8000 被占用）

已经有 Web 进程运行。检查重复进程：

```bash
ps aux | grep '[u]vicorn'
```

确认 PID 后停止旧进程，再启动一个 Web。

### `trusted ... must not be group or other writable`

不要修改 `/czj` 或 `/czj/code` 权限。检查：

- `app.yaml` 是否配置 `storage_trust_mode: data_root_boundary`；
- Worker 是否已重启并加载新配置；
- Web 和 evaluation Worker 是否设置正确的 `VLA_EVAL_PROFILES_ROOT`；
- `data`、`staging`、`inbox`、`runs`、`profiles` 是否为 `700`。

### `OSError: [Errno 22] Invalid argument`，发生在发布阶段

这是旧版本在 GPFS 上调用 `renameat2(RENAME_NOREPLACE)` 的问题。确认服务器 `main`
已经包含 GPFS 兼容修复，然后重启 transfer Worker：

```bash
git log --oneline --all --grep='GPFS'
```

### `import target already exists`

同名数据集已经发布。不要删除现有 inbox 数据，也不要重复导入；直接从“数据集”页面创建评测。

### 评测在 `PREFLIGHT` 立即失败

查看 evaluation Worker，而不是 transfer Worker。常见原因：

- 未设置 `VLA_EVAL_PROFILES_ROOT`；
- Web 与 evaluation Worker 使用了不同的 profiles 目录；
- profiles 仍位于代码目录而不是 `data/profiles`；
- profile 文件权限不安全；
- profile 名称或版本被修改。

### 评测到 VLM 阶段失败

检查：

```bash
nvidia-smi
grep -nE 'model_family|model_path' \
  /czj/code/vla-evaluation/data/profiles/genie02-{full,qwen3-vl}.yaml
```

确认 GPU 可用、模型目录存在且包含 `config.json`、Profile 的 `model_family` 与模型
`model_type` 一致，并且 Conda 环境已安装完整 GPU 依赖。

### `work-horse terminated unexpectedly`，任务约 3 分钟后失败

旧版本使用 RQ 默认的 `180` 秒任务超时，本地 VLM 加载模型和推理通常会超过该时间。
更新到包含评测任务 24 小时超时修复的版本，重启 Web 后重新创建评测任务。旧任务在
Redis 中记录的 `180` 秒不会自动改变。

### 页面只显示通用 `IMPORT_FAILED` 或 `EVALUATION_FAILED`

这是系统故意隐藏服务器内部路径和敏感信息。真实异常位于对应 Worker：

- 导入失败：看 transfer Worker；
- 评测失败：看 evaluation Worker；
- Web 无法访问或提交失败：看 Web 终端。

## 11. 安全注意事项

- 只通过 SSH 隧道访问 Web，不对公网直接开放 `8000`；
- 不修改共享 `/czj`、`/czj/code` 权限；
- 不使用 `chmod -R 777`；
- 不删除已有 `data/inbox` 数据来绕过重名保护；
- 不将密码、session secret、SSH 私钥或 API Key 提交到 Git；
- `unset HTTP_PROXY HTTPS_PROXY ALL_PROXY` 只影响当前终端，不影响其他用户；
- 执行 `kill` 前先用 `ps -fp <PID>` 确认进程身份；
- 不使用 `kill -9`，除非普通 `kill` 无法停止且已确认目标进程。
