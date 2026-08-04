# Genie02_report 代码分析计划

## 目标
梳理项目框架、主要功能、数据流、核心类/函数，并给出适合新读者的文件与类阅读顺序。

## 阶段
- [complete] 1. 盘点目录、依赖、README 与可执行入口
- [complete] 2. 追踪根目录指标计算与报告生成调用链
- [complete] 3. 追踪 attempt_eval 的 VLM 尝试评估调用链
- [complete] 4. 核对输入输出样例、运行方式与测试/风险点
- [complete] 5. 汇总架构图和分层阅读路线

## 范围
- 只读分析源码和样例数据。
- 不修改业务源码或生成物。

## Errors Encountered
| Error | Attempt | Resolution |
|---|---:|---|
| 系统 Python 3.9.6 缺 `pyarrow`，无法重跑 LeRobot 报告 | 1 | 使用已有报告产物核对；未安装依赖以免改变用户环境 |
| 系统 Python 缺 `cv2`，无法执行 attempt_eval dry-run | 1 | 使用已有抽帧/VLM 输出核对 |
| 工作区隔离 Python 同样缺 `pyarrow` | 2 | 停止运行验证，记录环境限制 |
| `ruff` 命令不存在 | 1 | 记录静态检查未运行，不安装工具 |
