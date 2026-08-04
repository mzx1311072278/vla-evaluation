# VLA 单机自动评测网站 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Ubuntu 22.04 + RTX 4090 单机上交付一个组内内部网站，支持受控远程数据导入、Genie02 自动评测、本地 VLM 分析、持久任务队列和可追溯报告。

**Architecture:** FastAPI 负责账号、服务端页面和 API；SQLite 保存业务状态；Redis/RQ 分别运行单并发 Transfer Worker 和 Evaluation Worker。原始数据、模型和产物落在宿主机 `/srv/vla-eval/data`，现有 Genie02 与 attempt_eval 代码通过插件接口复用。

**Tech Stack:** Python 3.11, FastAPI, Jinja2, HTMX, SQLAlchemy 2, SQLite WAL, Redis, RQ, Paramiko, rsync over SSH, PyTorch/Qwen2.5-VL, Docker Compose, Caddy, pytest, Playwright

---

## 文件结构

新增应用包：

```text
vla_eval/
|-- __init__.py                 # 应用版本
|-- config.py                   # YAML/环境变量配置与 RemoteSource
|-- db.py                       # SQLAlchemy engine/session/初始化
|-- models.py                   # User/Dataset/ImportJob/EvaluationJob
|-- security.py                 # 密码、会话用户和 CSRF
|-- queueing.py                 # RQ 队列封装，可在测试中替换
|-- datasets.py                 # 发现、预检、清单指纹
|-- remote.py                   # 远程路径校验、SSH/rsync 参数
|-- import_jobs.py              # 断点拉取、校验、原子发布
|-- profiles.py                 # 版本化评测方案
|-- evaluation.py               # METRICS/VLM/REPORT 编排
|-- cli.py                      # init-db/create-user/scan/smoke
`-- web/
    |-- app.py                  # FastAPI 工厂与中间件
    |-- routes_auth.py          # 登录/退出
    |-- routes_imports.py       # 数据导入页面/API
    |-- routes_datasets.py      # 数据集页面/API
    |-- routes_evaluations.py   # 提交、状态、重试、取消
    |-- routes_reports.py       # 报告与文件下载
    |-- templates/              # Jinja2 页面
    `-- static/app.css          # 操作型界面样式
```

现有代码改造：

```text
Genie02_report/__init__.py
Genie02_report/attempt_eval/__init__.py
Genie02_report/*.py             # 包内相对导入，保留 CLI
Genie02_report/attempt_eval/run_episode_attempt_eval.py
                                # 抽出可调用 run_attempt_evaluation()
```

部署与测试：

```text
config/app.example.yaml
config/profiles/genie02-full.yaml
deploy/Caddyfile
deploy/Dockerfile.web
deploy/Dockerfile.evaluation
deploy/entrypoint.sh
docker-compose.yml
tests/
```

## Task 1: 建立仓库、包和测试基线

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `vla_eval/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: 写入不会提交数据和模型的 `.gitignore`**

```gitignore
.env
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.superpowers/
data/
models/
runs/
*.sqlite3
Genie02_report/attempt_eval/outputs/
Genie02_report/report_*/
Genie02_report/zqyh_*/
vla_real_robot_evaluation/tmp/
```

- [ ] **Step 2: 创建 Python 3.11 项目配置**

`pyproject.toml` 至少固定以下依赖和 pytest 配置：

```toml
[project]
name = "vla-eval"
version = "0.1.0"
requires-python = ">=3.11,<3.12"
dependencies = [
  "fastapi>=0.116,<1",
  "uvicorn[standard]>=0.35,<1",
  "jinja2>=3.1,<4",
  "python-multipart>=0.0.20,<1",
  "sqlalchemy>=2.0,<3",
  "pydantic-settings>=2.10,<3",
  "pyyaml>=6.0,<7",
  "redis>=6.2,<7",
  "rq>=2.5,<3",
  "pwdlib[argon2]>=0.2,<1",
  "itsdangerous>=2.2,<3",
  "paramiko>=4,<5",
  "typer>=0.16,<1",
  "numpy==2.3.5",
  "pandas==2.3.3",
  "pyarrow==21.0.0",
]

[project.optional-dependencies]
gpu = [
  "torch",
  "transformers",
  "accelerate",
  "qwen-vl-utils",
  "opencv-python-headless",
  "pillow",
  "av",
]
dev = ["pytest>=8,<9", "pytest-cov>=6,<7", "httpx>=0.28,<1", "ruff>=0.12,<1", "playwright>=1.54,<2", "fakeredis>=2.30,<3"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --strict-markers"

[tool.ruff]
target-version = "py311"
line-length = 100
```

- [ ] **Step 3: 创建包版本和隔离测试配置**

```python
# vla_eval/__init__.py
__version__ = "0.1.0"
```

```python
# tests/conftest.py
from pathlib import Path

import pytest


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    for name in ("inbox", "staging", "runs", "models", "db"):
        (root / name).mkdir(parents=True)
    return root
```

- [ ] **Step 4: 创建虚拟环境并验证空测试集**

Run: `python3.11 -m venv .venv && .venv/bin/pip install -e '.[dev]' && .venv/bin/pytest`

Expected: `no tests ran`，命令退出码为 5；添加后续首个测试后退出码必须为 0。

- [ ] **Step 5: 初始化 Git，显式排除数据后提交基线**

Run: `git init && git add .gitignore pyproject.toml vla_eval tests docs && git status --short`

Expected: 暂存区不包含 `.mp4`、`.parquet`、模型权重或 `attempt_eval/outputs`。

```bash
git commit -m "chore: initialize vla evaluation web project"
```

## Task 2: 配置、白名单路径和远程数据源

**Files:**
- Create: `vla_eval/config.py`
- Create: `config/app.example.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: 写配置加载和越界路径的失败测试**

```python
from pathlib import Path

import pytest

from vla_eval.config import load_config, resolve_local_dataset_path


def test_resolve_local_dataset_path_rejects_escape(tmp_path: Path):
    root = tmp_path / "inbox"
    root.mkdir()
    with pytest.raises(ValueError, match="outside allowed root"):
        resolve_local_dataset_path(root, "../secret")


def test_load_config_parses_remote_source(tmp_path: Path):
    path = tmp_path / "app.yaml"
    path.write_text(
        "data_root: /srv/vla-eval/data\n"
        "remote_sources:\n"
        "  lab-a:\n"
        "    host: 10.0.0.8\n"
        "    port: 22\n"
        "    username: eval-read\n"
        "    key_path: /run/secrets/lab_a_key\n"
        "    known_hosts_path: /run/secrets/known_hosts\n"
        "    roots: [/data/rollouts]\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.remote_sources["lab-a"].roots == ("/data/rollouts",)
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/pytest tests/test_config.py -v`

Expected: FAIL，提示 `No module named 'vla_eval.config'`。

- [ ] **Step 3: 实现不可变配置模型和安全本地路径解析**

```python
# vla_eval/config.py
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RemoteSource:
    name: str
    host: str
    port: int
    username: str
    key_path: Path
    known_hosts_path: Path
    roots: tuple[str, ...]


@dataclass(frozen=True)
class AppConfig:
    data_root: Path
    database_url: str
    redis_url: str
    session_secret: str
    remote_sources: dict[str, RemoteSource]


def load_config(path: Path) -> AppConfig:
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data_root = Path(raw["data_root"]).expanduser().resolve()
    sources = {
        name: RemoteSource(
            name=name,
            host=str(item["host"]),
            port=int(item.get("port", 22)),
            username=str(item["username"]),
            key_path=Path(item["key_path"]),
            known_hosts_path=Path(item["known_hosts_path"]),
            roots=tuple(str(value) for value in item["roots"]),
        )
        for name, item in (raw.get("remote_sources") or {}).items()
    }
    return AppConfig(
        data_root=data_root,
        database_url=str(raw.get("database_url", f"sqlite:///{data_root / 'db/app.sqlite3'}")),
        redis_url=str(raw.get("redis_url", "redis://redis:6379/0")),
        session_secret=str(raw.get("session_secret", "")),
        remote_sources=sources,
    )


def resolve_local_dataset_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    allowed = root.resolve()
    if candidate != allowed and allowed not in candidate.parents:
        raise ValueError("path is outside allowed root")
    return candidate
```

- [ ] **Step 4: 创建不含真实密钥的示例配置**

`config/app.example.yaml` 固定 `data_root`、Redis、模型路径、一个示例远端源，并用 `${VLA_EVAL_SESSION_SECRET}` 表示运行时密钥。实现 `load_config` 时从同名环境变量覆盖空的 `session_secret`，为空则启动失败。

- [ ] **Step 5: 运行测试和提交**

Run: `.venv/bin/pytest tests/test_config.py -v`

Expected: 2 tests PASS。

```bash
git add vla_eval/config.py config/app.example.yaml tests/test_config.py
git commit -m "feat: add secure application configuration"
```

## Task 3: SQLite 业务模型和持久状态

**Files:**
- Create: `vla_eval/db.py`
- Create: `vla_eval/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: 写数据库初始化和状态默认值测试**

```python
from sqlalchemy import select

from vla_eval.db import create_engine_for_url, init_db, session_scope
from vla_eval.models import Dataset, EvaluationJob


def test_database_persists_dataset_and_job(tmp_path):
    engine = create_engine_for_url(f"sqlite:///{tmp_path / 'app.db'}")
    init_db(engine)
    with session_scope(engine) as session:
        dataset = Dataset(name="run-1", path="/data/run-1", kind="lerobot", status="READY")
        session.add(dataset)
        session.flush()
        session.add(EvaluationJob(dataset_id=dataset.id, profile_name="genie02-full"))
    with session_scope(engine) as session:
        job = session.scalar(select(EvaluationJob))
        assert job.state == "QUEUED"
        assert job.dataset.name == "run-1"
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/pytest tests/test_models.py -v`

Expected: FAIL，缺少 `vla_eval.db`。

- [ ] **Step 3: 实现 SQLAlchemy Base、WAL 和会话事务**

`vla_eval/db.py` 必须提供：

```python
class Base(DeclarativeBase):
    pass


def create_engine_for_url(url: str) -> Engine:
    engine = create_engine(url, connect_args={"check_same_thread": False})
    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def set_sqlite_pragmas(connection, _record):
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
    return engine


@contextmanager
def session_scope(engine: Engine):
    with Session(engine) as session:
        with session.begin():
            yield session
```

- [ ] **Step 4: 实现四个核心模型**

`vla_eval/models.py` 使用 UUID 字符串主键和 UTC 时间，定义：

- `User(username, password_hash, is_admin, active)`
- `Dataset(name, path, kind, status, fingerprint, size_bytes, episode_count, inspection_json)`
- `ImportJob(source_name, remote_path, target_name, state, progress, error_message, dataset_id)`
- `EvaluationJob(dataset_id, profile_name, profile_version, vlm_enabled, state, stage, progress, run_key, output_dir, error_code, error_message, params_json, provenance_json, created_by, cancel_requested)`

所有 JSON 字段使用 SQLAlchemy `JSON`，状态字段使用字符串并由业务层常量约束；关系只保留 `Dataset.evaluation_jobs` 和 `EvaluationJob.dataset`，避免第一版过度建模。

`provenance_json` 固定保存提交时的数据指纹、profile 名称/版本、应用版本、Git SHA、VLM 模型标识、Prompt 版本和完整推理参数；重试沿用该快照，不读取已经变化的默认配置。

- [ ] **Step 5: 运行测试和提交**

Run: `.venv/bin/pytest tests/test_models.py -v`

Expected: PASS。

```bash
git add vla_eval/db.py vla_eval/models.py tests/test_models.py
git commit -m "feat: persist datasets imports and evaluation jobs"
```

## Task 4: 将现有 Genie02 代码变为可安装包并建立回归测试

**Files:**
- Create: `Genie02_report/__init__.py`
- Create: `Genie02_report/attempt_eval/__init__.py`
- Modify: `Genie02_report/genie02_eval_report.py`
- Modify: `Genie02_report/genie02_episode_metrics.py`
- Modify: `Genie02_report/genie02_metrics_core.py`
- Modify: `Genie02_report/genie02_markdown_report.py`
- Modify: `Genie02_report/attempt_eval/run_episode_attempt_eval.py`
- Test: `tests/test_genie02_regression.py`

- [ ] **Step 1: 写现有样例指标回归测试**

```python
import json
from pathlib import Path

import pytest

from Genie02_report.genie02_eval_report import generate_report


SAMPLE = Path("Genie02_report/zqyh_2cm_mixed_ee_rot6_right_arm_only_eval_pi05_stage2_acp")


@pytest.mark.skipif(not SAMPLE.exists(), reason="large local sample is not installed")
def test_existing_lerobot_sample_matches_committed_metrics(tmp_path):
    expected = json.loads(Path("Genie02_report/report_20260708/metrics_core.json").read_text())
    actual = generate_report(SAMPLE, tmp_path)
    assert actual["n_episodes"] == expected["n_episodes"]
    assert actual["n_success"] == expected["n_success"]
    assert actual["gsr"] == pytest.approx(expected["gsr"])
```

- [ ] **Step 2: 运行测试并确认包导入失败**

Run: `.venv/bin/pytest tests/test_genie02_regression.py -v`

Expected: FAIL，包内绝对导入找不到 `genie02_episode_metrics`。

- [ ] **Step 3: 将包内导入改为相对导入，同时保留脚本模式**

每个顶层脚本采用同一模式：

```python
if __package__:
    from .genie02_eval_common import EvaluationError
else:
    from genie02_eval_common import EvaluationError
```

`attempt_eval` 继续使用现有的 `if __package__` 分支。禁止通过修改 `sys.path` 解决导入问题。

- [ ] **Step 4: 运行回归测试和 CLI 帮助**

Run: `.venv/bin/pytest tests/test_genie02_regression.py -v`

Expected: PASS 或仅因本机未安装大样例而 SKIP。

Run: `.venv/bin/python Genie02_report/genie02_eval_report.py -h`

Expected: 显示 `session_dir` 和 `--output-dir`，退出码 0。

- [ ] **Step 5: 提交**

```bash
git add Genie02_report/__init__.py Genie02_report/attempt_eval/__init__.py \
  Genie02_report/*.py Genie02_report/attempt_eval/run_episode_attempt_eval.py \
  tests/test_genie02_regression.py
git commit -m "refactor: package existing genie02 evaluation code"
```

## Task 5: 数据集发现、预检和清单指纹

**Files:**
- Create: `vla_eval/datasets.py`
- Test: `tests/test_datasets.py`

- [ ] **Step 1: 写 LeRobot、Session、越界符号链接测试**

```python
from pathlib import Path

import pandas as pd

from vla_eval.datasets import DatasetKind, inspect_dataset


def test_inspect_lerobot_dataset(tmp_path: Path):
    root = tmp_path / "run"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/info.json").write_text('{"total_episodes": 1, "fps": 30}')
    pd.DataFrame(
        {"episode_index": [0], "length": [1], "episode_success": ["success"], "data/chunk_index": [0], "data/file_index": [0]}
    ).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
    pd.DataFrame({"episode_index": [0], "timestamp": [0.0], "action": [[0.0, 0.0, 0.0]]}).to_parquet(
        root / "data/chunk-000/file-000.parquet"
    )
    result = inspect_dataset(root, allowed_root=tmp_path)
    assert result.kind is DatasetKind.LEROBOT
    assert result.ready is True
    assert len(result.fingerprint) == 64


def test_inspect_rejects_symlink_outside_allowed_root(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "session.json").write_text("{}")
    (root / "leak").symlink_to(Path("/etc/passwd"))
    result = inspect_dataset(root, allowed_root=tmp_path)
    assert result.ready is False
    assert "outside allowed root" in result.errors[0]
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/pytest tests/test_datasets.py -v`

Expected: FAIL，缺少 `vla_eval.datasets`。

- [ ] **Step 3: 实现检查结果和确定性指纹**

```python
class DatasetKind(StrEnum):
    LEROBOT = "lerobot"
    GENIE02_SESSION = "genie02_session"


@dataclass(frozen=True)
class DatasetInspection:
    kind: DatasetKind | None
    ready: bool
    fingerprint: str
    size_bytes: int
    episode_count: int | None
    errors: tuple[str, ...]
```

`inspect_dataset()` 必须：解析所有路径后检查仍位于 `allowed_root`；识别两种格式；读取关键 JSON/CSV/Parquet；检查视频引用；为每个文件记录相对路径、大小和 `mtime_ns`，并对 JSON/CSV/Parquet 元数据文件计算 SHA-256 内容哈希。最终指纹是规范化 manifest JSON 的 SHA-256，避免每次扫描完整读取数百 GB 视频。

- [ ] **Step 4: 运行测试和提交**

Run: `.venv/bin/pytest tests/test_datasets.py -v`

Expected: PASS。

```bash
git add vla_eval/datasets.py tests/test_datasets.py
git commit -m "feat: discover and validate evaluation datasets"
```

## Task 6: 评测方案和 Genie02 插件

**Files:**
- Create: `vla_eval/profiles.py`
- Create: `vla_eval/evaluation.py`
- Create: `config/profiles/genie02-full.yaml`
- Test: `tests/test_evaluation.py`

- [ ] **Step 1: 写方案解析和阶段顺序测试**

```python
from vla_eval.evaluation import EvaluationCallbacks, run_evaluation
from vla_eval.profiles import load_profile


def test_run_evaluation_calls_metrics_then_report(tmp_path, monkeypatch):
    stages = []
    monkeypatch.setattr("vla_eval.evaluation.generate_episode_metrics", lambda dataset, output: [])
    monkeypatch.setattr("vla_eval.evaluation.generate_metrics_core", lambda dataset, output: {"gsr": 1.0})
    monkeypatch.setattr(
        "vla_eval.evaluation.generate_markdown_report",
        lambda dataset, output: (output / "report.md"),
    )
    profile = load_profile("config/profiles/genie02-full.yaml")
    result = run_evaluation(
        dataset_path=tmp_path,
        output_dir=tmp_path / "run",
        profile=profile,
        vlm_enabled=False,
        callbacks=EvaluationCallbacks(
            on_stage=stages.append,
            on_progress=lambda _value: None,
            should_cancel=lambda: False,
        ),
    )
    assert stages == ["METRICS", "REPORT"]
    assert result.metrics["gsr"] == 1.0
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/pytest tests/test_evaluation.py -v`

Expected: FAIL，缺少评测模块。

- [ ] **Step 3: 实现版本化方案**

`genie02-full.yaml` 固定：方案名、版本 `1.0.0`、图像键、VLM 抽帧参数、review policy 和输出文件清单。`Profile` 使用冻结 dataclass；`load_profile()` 对未知字段和非法范围报错。

- [ ] **Step 4: 实现可测试的同步编排函数**

```python
class EvaluationCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class EvaluationCallbacks:
    on_stage: Callable[[str], None]
    on_progress: Callable[[float], None]
    should_cancel: Callable[[], bool]


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, Any]
    report_path: Path
    vlm_summary_path: Path | None


def run_evaluation(dataset_path, output_dir, profile, vlm_enabled, callbacks, resume_from="METRICS"):
    if resume_from == "METRICS":
        callbacks.on_stage("METRICS")
        generate_episode_metrics(dataset_path, output_dir)
        metrics = generate_metrics_core(dataset_path, output_dir)
    else:
        metrics = load_metrics_core(output_dir, load_session(dataset_path))
    if callbacks.should_cancel():
        raise EvaluationCancelled()
    vlm_path = None
    if vlm_enabled and resume_from in {"METRICS", "VLM"}:
        callbacks.on_stage("VLM")
        vlm_path = run_profile_vlm(dataset_path, output_dir / "attempt_eval", profile, callbacks)
    elif vlm_enabled:
        vlm_path = output_dir / "attempt_eval/attempt_summary.json"
    if callbacks.should_cancel():
        raise EvaluationCancelled()
    callbacks.on_stage("REPORT")
    report_path = generate_markdown_report(dataset_path, output_dir)
    return EvaluationResult(metrics, report_path, vlm_path)
```

`run_profile_vlm()` 在同一文件中定义，并在函数体内延迟导入 Task 7 的 `AttemptEvalConfig` 与 `run_attempt_evaluation()`；它把 profile 的抽帧、review 和模型参数映射到 config，把 `callbacks.should_cancel` 传给服务函数，并返回 `attempt_summary.json`。延迟导入保证 Task 6 的无 VLM 回归测试可独立通过，Task 7 完成后再启用 VLM 分支。

- [ ] **Step 5: 运行测试和提交**

Run: `.venv/bin/pytest tests/test_evaluation.py -v`

Expected: PASS。

```bash
git add vla_eval/profiles.py vla_eval/evaluation.py config/profiles/genie02-full.yaml tests/test_evaluation.py
git commit -m "feat: add versioned genie02 evaluation profile"
```

## Task 7: 将 VLM CLI 抽成可调用插件

**Files:**
- Modify: `Genie02_report/attempt_eval/run_episode_attempt_eval.py:21`
- Test: `tests/test_attempt_eval_service.py`

- [ ] **Step 1: 写不加载模型的 dry-run 服务测试**

```python
from pathlib import Path

from Genie02_report.attempt_eval.run_episode_attempt_eval import AttemptEvalConfig, run_attempt_evaluation


def test_run_attempt_evaluation_accepts_injected_dependencies(tmp_path: Path):
    config = AttemptEvalConfig(dataset_root=tmp_path, model_path=tmp_path / "model", output_dir=tmp_path / "out", dry_run=True)
    results = run_attempt_evaluation(config, episodes=[], progress=lambda _done, _total, _stage: None)
    assert results == []
    assert (config.output_dir / "attempt_summary.json").exists()
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/pytest tests/test_attempt_eval_service.py -v`

Expected: FAIL，缺少 `AttemptEvalConfig`。

- [ ] **Step 3: 抽出配置 dataclass 和服务函数**

在现有文件中增加与 CLI 参数一一对应的 `AttemptEvalConfig`，默认值保持当前 CLI 一致。把 `main()` 第 101-183 行循环移动到：

```python
def run_attempt_evaluation(
    config: AttemptEvalConfig,
    *,
    episodes: list[EpisodeMeta] | None = None,
    client_factory: Callable[..., LocalVLMClient] = LocalVLMClient,
    progress: Callable[[int, int, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> list[dict[str, Any]]:
```

函数返回全部结果，始终调用 `write_summary()`；`main()` 只负责 `parse_args()`、构建 config、调用函数和打印最终路径。每个 Episode 开始前调用 `should_cancel()`，为 true 时抛出 `EvaluationCancelled`；单 Episode 异常继续使用现有 fallback，不改变 JSON Schema。

- [ ] **Step 4: 运行服务测试、JSON 校验测试和 CLI dry-run**

Run: `.venv/bin/pytest tests/test_attempt_eval_service.py -v`

Expected: PASS。

Run: `.venv/bin/python Genie02_report/attempt_eval/run_episode_attempt_eval.py --help`

Expected: 所有现有参数仍存在。

- [ ] **Step 5: 提交**

```bash
git add Genie02_report/attempt_eval/run_episode_attempt_eval.py tests/test_attempt_eval_service.py
git commit -m "refactor: expose attempt evaluation service API"
```

## Task 8: 安全远程路径和 rsync 命令构造

**Files:**
- Create: `vla_eval/remote.py`
- Test: `tests/test_remote.py`

- [ ] **Step 1: 写路径穿越、参数注入和正确命令测试**

```python
import pytest

from vla_eval.config import RemoteSource
from vla_eval.remote import build_rsync_argv, normalize_remote_relative_path


@pytest.mark.parametrize("value", ["../secret", "/etc", "run\n--delete", "run\x00bad"])
def test_normalize_remote_path_rejects_unsafe_values(value):
    with pytest.raises(ValueError):
        normalize_remote_relative_path(value)


def test_build_rsync_argv_never_uses_shell(tmp_path):
    remote_source = RemoteSource(
        name="lab-a",
        host="10.0.0.8",
        port=22,
        username="eval-read",
        key_path=tmp_path / "key",
        known_hosts_path=tmp_path / "known_hosts",
        roots=("/data/rollouts",),
    )
    argv = build_rsync_argv(remote_source, "/data/rollouts", "run-1", tmp_path)
    assert argv[0] == "rsync"
    assert "--delete" not in argv
    assert argv[-1] == f"{tmp_path}/"
    assert "eval-read@10.0.0.8:/data/rollouts/run-1/" in argv
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/pytest tests/test_remote.py -v`

Expected: FAIL，缺少远程模块。

- [ ] **Step 3: 实现严格相对路径和 argv 列表**

`normalize_remote_relative_path()` 使用 `PurePosixPath`，拒绝绝对路径、`..`、空段、控制字符和以 `-` 开头的段。`build_rsync_argv()` 只接收配置文件中的 host/user/port/key/known_hosts，使用：

```python
return [
    "rsync", "-a", "--partial", "--append-verify", "--protect-args",
    "--info=progress2", "--out-format=%i|%l|%n",
    "-e", ssh_command_from_trusted_config(source),
    f"{source.username}@{source.host}:{remote_path}/",
    f"{staging.resolve()}/",
]
```

运行端必须调用 `subprocess.Popen(argv, shell=False, ...)`。远端部署文档要求使用只读 `rrsync` 包装器或 SFTP-only 账号；代码中不提供 `--delete`、`--rsync-path` 或用户自定义参数。

- [ ] **Step 4: 运行测试和提交**

Run: `.venv/bin/pytest tests/test_remote.py -v`

Expected: PASS。

```bash
git add vla_eval/remote.py tests/test_remote.py
git commit -m "feat: validate remote paths and rsync arguments"
```

## Task 9: 远程导入任务、断点续传和原子发布

**Files:**
- Create: `vla_eval/import_jobs.py`
- Test: `tests/test_import_jobs.py`

- [ ] **Step 1: 写成功发布与失败保留 staging 测试**

```python
from pathlib import Path

import pytest

from vla_eval.datasets import DatasetInspection, DatasetKind
from vla_eval.import_jobs import ImportSpec, TransferError, execute_import


def import_spec(staging: Path, inbox: Path) -> ImportSpec:
    return ImportSpec(
        job_id="job-1",
        source_name="lab-a",
        remote_root="/data/rollouts",
        remote_relative_path="run-1",
        staging_path=staging,
        target_path=inbox,
    )


def test_import_publishes_only_after_preflight(tmp_path, monkeypatch):
    staging = tmp_path / "staging" / "job-1"
    inbox = tmp_path / "inbox" / "alice" / "run-1"

    def fake_transfer(_argv, on_progress):
        staging.mkdir(parents=True)
        (staging / "received.marker").write_text("ok")
        on_progress(100.0)

    inspection = DatasetInspection(DatasetKind.LEROBOT, True, "a" * 64, 2, 0, ())
    result = execute_import(
        import_spec(staging, inbox),
        transfer=fake_transfer,
        inspector=lambda _path: inspection,
    )
    assert result.dataset_path == inbox
    assert inbox.exists()
    assert not staging.exists()


def test_import_failure_keeps_partial_files(tmp_path):
    spec = import_spec(tmp_path / "staging/job-2", tmp_path / "inbox/alice/run-2")
    with pytest.raises(TransferError):
        execute_import(spec, transfer=lambda _argv, _progress: (_ for _ in ()).throw(TransferError("network")))
    assert spec.staging_path.exists()
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/pytest tests/test_import_jobs.py -v`

Expected: FAIL，缺少导入模块。

- [ ] **Step 3: 实现可注入 transfer 的同步导入核心**

`execute_import(spec, *, transfer=run_rsync, inspector=inspect_dataset)` 状态顺序固定为 `CONNECTING -> TRANSFERRING -> VERIFYING -> PREFLIGHT -> READY`；所有更新在独立事务中写数据库。成功时先检查目标不存在，再用同一文件系统的 `Path.replace()` 原子发布。失败时记录 `FAILED` 和用户可读错误，保留 staging 以供下一次 rsync 继续。

`ImportSpec` 是冻结 dataclass，字段与测试构造函数一致；`ImportResult(dataset_path, inspection)` 返回发布路径和 Task 5 检查结果；`TransferError` 仅表示可重试的网络/rsync 失败，预检失败使用 `DatasetValidationError`。

- [ ] **Step 4: 实现 rsync 流式进度解析**

使用 `Popen(..., stdout=PIPE, stderr=STDOUT, text=True, shell=False)` 逐行读取 `--info=progress2`，用正则 `r"\s*(\d{1,3})%"` 更新 0-100 进度；进程非零退出码映射为 `TransferError`，日志保存最后 200 行，私钥路径不写入用户日志。

- [ ] **Step 5: 运行测试和提交**

Run: `.venv/bin/pytest tests/test_import_jobs.py tests/test_remote.py -v`

Expected: PASS。

```bash
git add vla_eval/import_jobs.py tests/test_import_jobs.py
git commit -m "feat: import remote datasets with resumable transfer"
```

## Task 10: RQ 队列和评测状态恢复

**Files:**
- Create: `vla_eval/queueing.py`
- Create: `vla_eval/tasks.py`
- Create: `tests/fakes.py`
- Modify: `tests/conftest.py`
- Test: `tests/test_tasks.py`

- [ ] **Step 1: 写队列名和失败状态测试**

```python
def test_queue_names_are_isolated(fake_redis):
    queues = create_queues("redis://unused", connection=fake_redis)
    assert queues.transfer.name == "transfers"
    assert queues.evaluation.name == "evaluations"


def test_evaluation_task_records_failure(db_engine, evaluation_job, monkeypatch):
    monkeypatch.setattr("vla_eval.tasks.run_evaluation", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        run_evaluation_task(evaluation_job.id)
    assert reload_job(db_engine, evaluation_job.id).state == "FAILED"
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/pytest tests/test_tasks.py -v`

Expected: FAIL，缺少队列模块。

- [ ] **Step 3: 实现两条持久队列和任务入口**

`QueueBundle(transfer, evaluation)` 由同一 Redis 连接创建。`run_import_task(import_id)` 调用 Task 9；`run_evaluation_task(job_id)` 调用 Task 6，阶段/进度 callback 每次在新数据库事务中更新。重试根据最后失败阶段传入 `resume_from`，并在跳过阶段前验证对应产物存在。异常映射为 `FAILED` 后重新抛出，让 RQ 保存 traceback；用户页面只显示 `error_code` 和清洗后的 `error_message`。

`tests/fakes.py` 提供记录 `.enqueue(function, *args)` 调用的 `FakeQueue` 和 `FakeQueueBundle`；`tests/conftest.py` 增加 `fakeredis.FakeRedis` 生成的 `fake_redis`、内存 SQLite `db_engine`、`fake_queues`、`dataset`、`ready_dataset`、`evaluation_job` fixture，并实现按主键重新查询的 `reload_job(engine, job_id)` helper。所有后续 Web 测试复用这些名称，不连接真实 Redis。

- [ ] **Step 4: 增加启动恢复规则**

`recover_interrupted_jobs()` 把数据库中 `RUNNING`、`METRICS`、`VLM`、`REPORT`、`TRANSFERRING` 状态改为 `INTERRUPTED`，不自动重复 GPU 计算或覆盖 staging。网页重试会创建新的 RQ job，沿用同一业务 job ID 和已完成阶段产物。

- [ ] **Step 5: 运行测试和提交**

Run: `.venv/bin/pytest tests/test_tasks.py -v`

Expected: PASS。

```bash
git add vla_eval/queueing.py vla_eval/tasks.py tests/test_tasks.py
git commit -m "feat: queue persistent import and evaluation tasks"
```

## Task 11: 登录、会话和 CSRF

**Files:**
- Create: `vla_eval/security.py`
- Create: `vla_eval/web/app.py`
- Create: `vla_eval/web/routes_auth.py`
- Create: `vla_eval/web/templates/base.html`
- Create: `vla_eval/web/templates/login.html`
- Create: `tests/web/conftest.py`
- Test: `tests/web/test_auth.py`

- [ ] **Step 1: 写未登录跳转、登录和 CSRF 测试**

```python
def test_protected_page_redirects_to_login(client):
    response = client.get("/datasets", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_login_rejects_missing_csrf(client, user):
    response = client.post("/login", data={"username": "alice", "password": "secret"})
    assert response.status_code == 403
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/pytest tests/web/test_auth.py -v`

Expected: FAIL，缺少 Web 应用。

- [ ] **Step 3: 实现密码和会话工具**

使用 `pwdlib.PasswordHash.recommended()` 创建/验证 Argon2 哈希。Starlette `SessionMiddleware` 配置 `https_only=True`、`same_site="lax"`、12 小时有效期。每个会话生成 `csrf_token=secrets.token_urlsafe(32)`，所有 POST 表单携带隐藏字段并用 `secrets.compare_digest()` 验证。

- [ ] **Step 4: 实现应用工厂和认证路由**

```python
def create_app(config: AppConfig, engine: Engine, queues: QueueBundle) -> FastAPI:
    app = FastAPI(title="VLA Evaluation")
    app.add_middleware(SessionMiddleware, secret_key=config.session_secret, https_only=True, same_site="lax", max_age=43200)
    app.state.config = config
    app.state.engine = engine
    app.state.queues = queues
    app.include_router(auth_router)
    return app
```

登录成功只保存 `user_id` 和 CSRF token；退出清空会话。禁止在 Cookie 或日志中保存密码哈希。

`tests/web/conftest.py` 提供 `client`、`user` 和已登录的 `auth_client`；`auth_client.csrf` 从登录前 GET `/login` 返回的隐藏字段读取，不硬编码 token。它复用根 conftest 的 `dataset`/`ready_dataset`，并提供 `successful_job` fixture，产物目录中写入确定的 `metrics_core.json` 和 CSV，供 Task 14 使用。

- [ ] **Step 5: 运行测试和提交**

Run: `.venv/bin/pytest tests/web/test_auth.py -v`

Expected: PASS。

```bash
git add vla_eval/security.py vla_eval/web tests/web/test_auth.py
git commit -m "feat: add authenticated web sessions"
```

## Task 12: 数据导入与数据集页面

**Files:**
- Create: `vla_eval/web/routes_imports.py`
- Create: `vla_eval/web/routes_datasets.py`
- Create: `vla_eval/web/templates/imports/index.html`
- Create: `vla_eval/web/templates/imports/new.html`
- Create: `vla_eval/web/templates/imports/detail.html`
- Create: `vla_eval/web/templates/datasets/index.html`
- Create: `vla_eval/web/templates/datasets/detail.html`
- Create: `vla_eval/web/static/app.css`
- Test: `tests/web/test_imports.py`
- Test: `tests/web/test_datasets.py`

- [ ] **Step 1: 写普通用户不能提交任意主机的测试**

```python
def test_import_form_accepts_only_configured_source(auth_client, fake_queues):
    response = auth_client.post(
        "/imports",
        data={"csrf_token": auth_client.csrf, "source_name": "evil-host", "root": "/data", "relative_path": "run-1", "target_name": "run-1"},
    )
    assert response.status_code == 422
    assert fake_queues.transfer.count == 0
```

- [ ] **Step 2: 写 READY 数据集才可见为可评测的测试**

```python
def test_dataset_page_disables_evaluation_until_ready(auth_client, dataset):
    dataset.status = "PREFLIGHT_FAILED"
    response = auth_client.get(f"/datasets/{dataset.id}")
    assert response.status_code == 200
    assert 'aria-disabled="true"' in response.text
```

```python
def test_remote_directory_browser_stays_under_registered_root(auth_client, fake_sftp):
    response = auth_client.get("/api/remote-sources/lab-a/directories", params={"root": "/data/rollouts", "path": "../etc"})
    assert response.status_code == 422
    assert fake_sftp.listdir_calls == []
```

- [ ] **Step 3: 运行测试并确认失败**

Run: `.venv/bin/pytest tests/web/test_imports.py tests/web/test_datasets.py -v`

Expected: FAIL，路由未注册。

- [ ] **Step 4: 实现路由和任务提交**

POST `/imports` 只从 `config.remote_sources[source_name]` 取主机和凭据；验证 root 属于源配置，relative path 使用 Task 8，target name 仅允许 Unicode 字母数字、空格、`._-`。创建 `ImportJob(QUEUED)` 后仅向 `transfers` 队列传业务 ID。

GET `/imports/{id}` 与 `/datasets/{id}` 返回完整页面；GET `/api/imports/{id}` 返回 JSON 供 HTMX 每 2 秒轮询。状态进入终态后响应头设置 `HX-Trigger: job-finished` 停止轮询。

GET `/api/remote-sources/{name}/directories` 使用 Paramiko、固定 Host Key 和配置中的只读密钥列出目录；只返回目录名，不返回文件，root 必须精确匹配配置项，path 必须通过 Task 8。把 SFTP client factory 作为依赖注入，以便测试不访问网络。

POST `/datasets/{id}/attachments` 只接受 `.json/.yaml/.yml/.csv`，单文件上限 20 MiB，总附件上限 100 MiB；文件名使用 `Path(name).name` 后再次校验，不允许覆盖原始数据文件，统一写入数据集下的 `_attachments/`。

- [ ] **Step 5: 实现安静、可扫描的操作界面**

页面使用固定宽度状态列、阶段进度条、错误摘要和明确命令按钮；不得把数据集、任务或页面 section 全部做成装饰卡片。长路径允许换行并提供复制按钮，按钮使用 Lucide 图标和 tooltip。

- [ ] **Step 6: 运行测试和提交**

Run: `.venv/bin/pytest tests/web/test_imports.py tests/web/test_datasets.py -v`

Expected: PASS。

```bash
git add vla_eval/web/routes_imports.py vla_eval/web/routes_datasets.py \
  vla_eval/web/templates vla_eval/web/static tests/web/test_imports.py tests/web/test_datasets.py
git commit -m "feat: add dataset import and discovery UI"
```

## Task 13: 新建评测、任务进度、重试与取消

**Files:**
- Create: `vla_eval/web/routes_evaluations.py`
- Create: `vla_eval/web/templates/evaluations/new.html`
- Create: `vla_eval/web/templates/evaluations/detail.html`
- Test: `tests/web/test_evaluations.py`

- [ ] **Step 1: 写提交、去重和取消测试**

```python
def test_submit_evaluation_enqueues_business_id(auth_client, ready_dataset, fake_queues):
    response = auth_client.post(
        "/evaluations",
        data={"csrf_token": auth_client.csrf, "dataset_id": ready_dataset.id, "profile": "genie02-full", "vlm_enabled": "true"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert fake_queues.evaluation.enqueued[0].args == (response.headers["location"].rsplit("/", 1)[-1],)


def test_duplicate_successful_run_requires_explicit_force(auth_client, successful_job):
    response = auth_client.post("/evaluations", data=matching_form(successful_job))
    assert response.status_code == 409
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/pytest tests/web/test_evaluations.py -v`

Expected: FAIL，路由不存在。

- [ ] **Step 3: 实现运行键和提交规则**

`run_key = sha256(dataset.fingerprint + profile.name + profile.version + canonical_params_json)`。只有 `Dataset.status == READY` 可提交；已有相同成功运行时返回 409 和复用链接，用户再次提交 `force=true` 才创建新任务。创建任务时写入 `provenance_json`，其中 Git SHA 从 `VLA_EVAL_GIT_SHA` 环境变量读取，模型与 Prompt 版本来自已加载 profile。

- [ ] **Step 4: 实现阶段展示、重试和协作式取消**

GET `/api/evaluations/{id}` 返回 `state/stage/progress/error/updated_at`。POST `/evaluations/{id}/retry` 仅允许 `FAILED` 或 `INTERRUPTED`；重新确认数据指纹未变化。POST `/evaluations/{id}/cancel` 设置 `cancel_requested=True`；Evaluation Worker 在 Episode 间和阶段间检查，CUDA `generate()` 运行中不强杀进程。

- [ ] **Step 5: 运行测试和提交**

Run: `.venv/bin/pytest tests/web/test_evaluations.py -v`

Expected: PASS。

```bash
git add vla_eval/web/routes_evaluations.py vla_eval/web/templates/evaluations tests/web/test_evaluations.py
git commit -m "feat: submit and monitor evaluation jobs"
```

## Task 14: 报告页面和安全下载

**Files:**
- Create: `vla_eval/web/routes_reports.py`
- Create: `vla_eval/web/templates/reports/detail.html`
- Test: `tests/web/test_reports.py`

- [ ] **Step 1: 写报告指标和路径越界下载测试**

```python
def test_report_page_shows_core_metrics(auth_client, successful_job):
    response = auth_client.get(f"/reports/{successful_job.id}")
    assert response.status_code == 200
    assert "GSR" in response.text
    assert "90.0%" in response.text


def test_download_rejects_path_escape(auth_client, successful_job):
    response = auth_client.get(f"/reports/{successful_job.id}/files/../../app.sqlite3")
    assert response.status_code in {404, 422}
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/pytest tests/web/test_reports.py -v`

Expected: FAIL，报告路由不存在。

- [ ] **Step 3: 实现报告聚合与下载白名单**

报告页读取 `metrics_core.json`、`episode_metrics.csv` 和可选 `attempt_summary.json`，不在请求中重新计算。下载文件名只允许任务登记的 `report_*.md`、`metrics_core.json`、`episode_metrics.csv`、`attempt_summary.json/csv`、`smoothness_curve.svg`；解析后必须位于该 job 的 `output_dir`。

- [ ] **Step 4: 实现操作型报告布局**

首屏显示 GSR、成功数、TTS、平滑度、待复核数；下方使用表格展示 Episode，支持结果、复核状态筛选，失败 Episode 展示证据帧路径和 VLM 原因。不得让 VLM 覆盖原始成功标签，两个字段并列显示。

- [ ] **Step 5: 运行测试和提交**

Run: `.venv/bin/pytest tests/web/test_reports.py -v`

Expected: PASS。

```bash
git add vla_eval/web/routes_reports.py vla_eval/web/templates/reports tests/web/test_reports.py
git commit -m "feat: render and export evaluation reports"
```

## Task 15: 管理 CLI、Docker Compose 和 Ubuntu 22.04 部署

**Files:**
- Create: `vla_eval/cli.py`
- Create: `deploy/Dockerfile.web`
- Create: `deploy/Dockerfile.evaluation`
- Create: `deploy/entrypoint.sh`
- Create: `deploy/backup.sh`
- Create: `deploy/Caddyfile`
- Create: `docker-compose.yml`
- Create: `.env.example`
- Create: `docs/deployment/ubuntu-22.04.md`
- Test: `tests/test_cli.py`

- [ ] **Step 1: 写 init-db 和 create-user CLI 测试**

```python
def test_create_user_hashes_password(cli_runner, app_config):
    result = cli_runner.invoke(["create-user", "alice", "--password", "secret"])
    assert result.exit_code == 0
    user = load_user("alice")
    assert user.password_hash != "secret"
    assert verify_password("secret", user.password_hash)
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/pytest tests/test_cli.py -v`

Expected: FAIL，CLI 不存在。

- [ ] **Step 3: 实现管理命令**

`python -m vla_eval.cli` 提供 `init-db`、`create-user`、`disable-user`、`scan-datasets`、`recover-jobs` 和 `smoke`。非交互模式通过环境变量读取初始密码，禁止把明文密码打印到日志。

测试使用 `typer.testing.CliRunner` 构造 `cli_runner`，从 Task 2 示例配置创建 `app_config`，并从 `vla_eval.security` 导入 `verify_password`；`load_user()` 是测试内通过 SQLAlchemy 按用户名查询的 helper。

Web 应用同时提供 `/health`：检查 SQLite 查询、Redis ping 和数据根目录可写；任一失败返回 503 和对应组件名，不在响应中暴露凭据或绝对密钥路径。

- [ ] **Step 4: 创建三个最小镜像**

`Dockerfile.web` 安装基础依赖；`Dockerfile.evaluation` 基于已验证的 NVIDIA CUDA runtime 镜像并安装 `.[gpu]`；Transfer Worker 复用 Web 镜像并额外安装 Ubuntu `rsync` 和 OpenSSH client。所有容器以非 root UID 运行。

- [ ] **Step 5: 编写 Compose 服务和健康检查**

`docker-compose.yml` 定义 `caddy`、`web`、`redis`、`transfer-worker`、`evaluation-worker`。仅 Caddy 发布 `443:443`；Redis 和 Web 使用内部网络。Evaluation Worker 配置 `gpus: all`，所有服务挂载 `/srv/vla-eval/data`，Redis 启用 AOF，服务使用 `restart: unless-stopped`。

- [ ] **Step 6: 编写 Ubuntu 22.04 部署文档**

文档给出 NVIDIA 驱动验证、Docker Engine、Compose Plugin、NVIDIA Container Toolkit、目录权限、Caddy 内网证书、SSH Host Key、只读 rrsync/SFTP 账号、SMB 可选配置、关闭睡眠、开机自启、备份和 Ubuntu 22.04 在 2027 年 4 月前的升级/Ubuntu Pro 计划。

`deploy/backup.sh` 使用 `sqlite3 .backup` 生成一致数据库备份，并归档 `config/`；目标目录来自 `VLA_EVAL_BACKUP_DIR`，保留最近 30 份。部署文档使用 systemd timer 每日执行，并明确原始视频由公司存储策略负责，不复制进该小型配置备份。

- [ ] **Step 7: 运行测试、Compose 配置检查和提交**

Run: `.venv/bin/pytest tests/test_cli.py -v`

Expected: PASS。

Run: `docker compose config --quiet`

Expected: 无输出，退出码 0。

```bash
git add vla_eval/cli.py deploy docker-compose.yml .env.example docs/deployment tests/test_cli.py
git commit -m "feat: deploy evaluation service on ubuntu 22.04"
```

## Task 16: 端到端验收和视觉验证

**Files:**
- Create: `tests/e2e/test_evaluation_workflow.py`
- Create: `tests/e2e/test_visual_layout.py`
- Modify: `docs/deployment/ubuntu-22.04.md`

- [ ] **Step 1: 写完整工作流测试**

测试创建用户、登记测试远端源、用 fake rsync 导入最小 LeRobot fixture、预检、提交无 VLM 评测、等待成功并下载 `metrics_core.json`。断言所有状态按设计流转，任务记录包含数据指纹、方案版本和代码版本。

- [ ] **Step 2: 写 Playwright 桌面与移动端布局测试**

```python
@pytest.mark.parametrize("viewport", [{"width": 1440, "height": 1000}, {"width": 390, "height": 844}])
def test_core_pages_have_no_horizontal_overflow(page, live_server, viewport):
    page.set_viewport_size(viewport)
    login(page, live_server)
    for path in ("/imports", "/datasets", "/evaluations/new"):
        page.goto(live_server + path)
        assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
```

- [ ] **Step 3: 运行全套非 GPU 测试**

Run: `.venv/bin/pytest --cov=vla_eval --cov=Genie02_report --cov-report=term-missing`

Expected: 全部测试 PASS；缺少本地大样例的回归测试允许 SKIP，不能出现 XFAIL。

- [ ] **Step 4: 在 Ubuntu 4090 上运行 GPU 冒烟测试**

Run: `docker compose run --rm evaluation-worker nvidia-smi`

Expected: 显示 RTX 4090 和可用驱动。

Run: `docker compose run --rm evaluation-worker python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"`

Expected: 输出包含 `NVIDIA GeForce RTX 4090`。

Run: 在网站对 2 个 Episode 启用 VLM，确认生成 `attempt_summary.json`、至少一个 episode JSON、抽帧证据，并且 GPU 任务并发为 1。

- [ ] **Step 5: 验证重启恢复**

在传输中重启 `transfer-worker`，确认 staging 保留且重试续传；在评测阶段重启 `evaluation-worker`，确认任务进入 `INTERRUPTED`，由网页手动重试而非自动重复计算。

- [ ] **Step 6: 更新部署文档中的实测版本并提交**

记录实际 NVIDIA 驱动、Docker、Compose、CUDA runtime、PyTorch、模型权重标识和完整 smoke 命令输出摘要。

```bash
git add tests/e2e docs/deployment/ubuntu-22.04.md
git commit -m "test: verify end-to-end evaluation workflow"
```

## 最终验收命令

```bash
.venv/bin/ruff check .
.venv/bin/pytest
docker compose config --quiet
docker compose up -d
docker compose ps
curl -kfsS https://localhost/health
```

Expected:

- Ruff 无错误。
- pytest 全部通过，仅允许未安装大样例时的显式 SKIP。
- Compose 五个服务均为 running/healthy。
- `/health` 返回 `{"status":"ok"}`，且检查 SQLite、Redis 和数据目录可写；Worker 运行状态由 `docker compose ps` 与 RQ dashboard/日志确认。
- 组员无需 SSH，可从已登记远端源导入数据、提交评测、查看进度、重试失败阶段并下载报告。
