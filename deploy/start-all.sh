#!/usr/bin/env bash
# 一键启动 VLA 评测三服务（tmux 三窗口）：web / transfers / evaluations
#
# 用法（在服务器上执行）:
#   ./deploy/start-all.sh        # 评测 worker 默认用 GPU 0
#   ./deploy/start-all.sh 1      # 评测 worker 改用 GPU 1
#
# 常用操作:
#   tmux attach -t vla-eval      # 进入会话查看（Ctrl-b d 退出，服务继续跑）
#   Ctrl-b 0/1/2                 # 切换 web / transfers / evaluations 窗口
#   tmux kill-session -t vla-eval # 停止全部三个服务
#
# 路径默认按 /czj 部署，可用环境变量覆盖：
#   VLA_EVAL_TMUX_SESSION / VLA_EVAL_CONDA_ENV / VLA_EVAL_APP_DIR
#   VLA_EVAL_CONFIG_FILE / VLA_EVAL_PROFILES_ROOT / VLA_EVAL_GPU / VLA_EVAL_WEB_PORT
set -euo pipefail

SESSION="${VLA_EVAL_TMUX_SESSION:-vla-eval}"
CONDA_ENV="${VLA_EVAL_CONDA_ENV:-/czj/envs/vla-eval}"
APP_DIR="${VLA_EVAL_APP_DIR:-/czj/code/vla-evaluation/app}"
CONFIG_FILE="${VLA_EVAL_CONFIG_FILE:-/czj/code/vla-evaluation/config/app.yaml}"
PROFILES_ROOT="${VLA_EVAL_PROFILES_ROOT:-/czj/code/vla-evaluation/data/profiles}"
WEB_PORT="${VLA_EVAL_WEB_PORT:-8000}"
GPU="${1:-${VLA_EVAL_GPU:-0}}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "错误: 服务器未安装 tmux，先执行: conda install -y tmux 或 apt install tmux" >&2
  exit 1
fi

if [ ! -x "${CONDA_ENV}/bin/python" ]; then
  echo "错误: conda 环境 ${CONDA_ENV} 不可用（找不到 bin/python）。" >&2
  echo "先执行 conda env list 查看真实环境路径，然后：" >&2
  echo "  VLA_EVAL_CONDA_ENV=<真实路径> ./deploy/start-all.sh" >&2
  exit 1
fi

if tmux has-session -t "$SESSION" >/dev/null 2>&1; then
  echo "会话 ${SESSION} 已存在，未重复启动。"
  echo "  查看: tmux attach -t ${SESSION}"
  echo "  重启全部: tmux kill-session -t ${SESSION} 后重新运行本脚本"
  exit 0
fi

# 直接用 PATH 方式激活环境，不依赖 bin/activate 脚本（部分精简环境没有它）
common="export CONDA_PREFIX='${CONDA_ENV}' && export PATH='${CONDA_ENV}/bin':\$PATH \
&& cd '${APP_DIR}' \
&& export VLA_EVAL_CONFIG='${CONFIG_FILE}' \
&& export VLA_EVAL_PROFILES_ROOT='${PROFILES_ROOT}'"

# 先建空窗口，再用 send-keys 把命令敲进去：窗口常驻，命令报错会留在屏幕上，
# 不会因为启动命令瞬间失败导致会话和 tmux 服务一起消失。
tmux new-session -d -s "$SESSION" -n web
tmux set-option -t "$SESSION" exit-empty off
tmux set-option -t "${SESSION}:web" remain-on-exit on
tmux send-keys -t "${SESSION}:web" \
  "${common} && uvicorn vla_eval.server:create_app_from_env --factory --host 0.0.0.0 --port ${WEB_PORT}" Enter

tmux new-window -t "$SESSION" -n transfers
tmux set-option -t "${SESSION}:transfers" remain-on-exit on
tmux send-keys -t "${SESSION}:transfers" \
  "${common} && python -m vla_eval.cli worker --queue transfers" Enter

tmux new-window -t "$SESSION" -n evaluations
tmux set-option -t "${SESSION}:evaluations" remain-on-exit on
tmux send-keys -t "${SESSION}:evaluations" \
  "${common} && CUDA_VISIBLE_DEVICES=${GPU} python -m vla_eval.cli worker --queue evaluations" Enter

echo "三服务已在 tmux 会话 ${SESSION} 中启动（evaluations 使用 GPU ${GPU}）"
echo "  生效 profiles 目录: ${PROFILES_ROOT}（改 app/config/profiles 不生效）"
echo "  查看:     tmux attach -t ${SESSION}（Ctrl-b d 退出）"
echo "  切换窗口: Ctrl-b 0=web  1=transfers  2=evaluations"
echo "  停止全部: tmux kill-session -t ${SESSION}"
