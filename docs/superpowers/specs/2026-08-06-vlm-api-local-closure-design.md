# VLM API 本机闭环设计

## 目标

在不依赖真实供应商密钥、Ubuntu 服务器或 GPU 的前提下，将
`feature/vla-eval-web-vlm-api-backend` 分支推进到本机可验证状态。

## 范围

- 修复全仓 Ruff 报告的问题，不改变 Genie02 指标、报告字段或业务语义。
- 复核现有 VLM API 测试覆盖；仅在存在行为缺口时增加测试。
- 运行全量 Pytest 和 Ruff。
- 本机存在 Docker 时验证 Compose 配置，并执行不依赖密钥和 GPU 的构建检查。
- 将 `.coverage` 作为测试生成物排除在 Git 工作区之外。

## 非目标

- 不调用真实 VLM 服务，不接触或保存真实 API 密钥。
- 不宣称完成 Ubuntu、CUDA、4090、模型权重或生产重启恢复验收。
- 不推送分支、不合并分支、不修改远端仓库。

## 修改原则

静态检查修复采用最小改动：缓存装饰器和导入按 Python 3.11 规范整理；可选
Parquet 介入字段读取只捕获读取阶段的预期异常并记录调试日志；报告日期显式使用
本机时区。测试优先复用现有覆盖，避免为纯格式变更增加重复单元测试。

## 验收标准

- `ruff check .` 返回 0。
- `.venv/bin/pytest` 返回 0，任何跳过项必须说明原因。
- `docker compose config --quiet` 在 Docker Compose 可用时返回 0。
- `git status --short` 不再显示 `.coverage`，且只包含本次有意修改。
