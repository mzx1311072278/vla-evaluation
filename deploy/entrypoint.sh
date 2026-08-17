#!/usr/bin/env bash
# Shared container entrypoint: ensure the schema exists, then hand off to the
# image CMD (uvicorn for web, `cli worker` for the workers).
set -euo pipefail

echo "[entrypoint] initializing database"
python -m vla_eval.cli init-db

echo "[entrypoint] starting: $*"
exec "$@"
