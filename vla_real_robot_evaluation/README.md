# VLA 真机评测系统调研

从 [`VLA真机评测调研与系统框架.md`](VLA真机评测调研与系统框架.md) 开始阅读。

```text
vla_real_robot_evaluation/
├── VLA真机评测调研与系统框架.md   # 主报告与建议
├── framework/                    # 可执行评测框架
├── research/                     # 带一手来源的调研证据
├── tmp/                          # 论文/网页/解析临时文件
├── task_plan.md                  # 工作计划
├── findings.md                   # 关键发现
└── progress.md                   # 过程与核验记录
```

当前交付是调研与系统设计，不包含评测平台代码。推荐下一步先实现 P0 数据契约与 Genie02 兼容导入，再接在线采集和统计门禁。
