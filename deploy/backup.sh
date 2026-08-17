#!/usr/bin/env bash
# Consistent online backup of the SQLite database plus a config/ archive.
#
# IMPORTANT: raw video and dataset binaries are NOT copied here. They follow the
# company storage policy (separate, large-capacity storage). This script only
# protects the small, critical configuration + database state.
set -euo pipefail

: "${VLA_EVAL_BACKUP_DIR:?VLA_EVAL_BACKUP_DIR is required (destination directory)}"
: "${VLA_EVAL_DB_PATH:?VLA_EVAL_DB_PATH is required (path to app.sqlite3)}"
CONFIG_DIR="${VLA_EVAL_CONFIG_DIR:-/srv/vla-eval/config}"
KEEP="${VLA_EVAL_KEEP:-30}"

if ! command -v sqlite3 >/dev/null 2>&1; then
	echo "error: sqlite3 CLI is required (apt-get install sqlite3)" >&2
	exit 1
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
work_dir="${VLA_EVAL_BACKUP_DIR}/${timestamp}"
mkdir -p "$work_dir"

# sqlite3 .backup performs a safe, consistent online copy (handles WAL).
sqlite3 "$VLA_EVAL_DB_PATH" ".backup '${work_dir}/app.sqlite3'"

if [ -d "$CONFIG_DIR" ]; then
	tar -C "$CONFIG_DIR" -czf "${work_dir}/config.tar.gz" .
fi

sha256sum "${work_dir}/app.sqlite3" > "${work_dir}/app.sqlite3.sha256"

archive="${VLA_EVAL_BACKUP_DIR}/${timestamp}.tar.gz"
tar -C "${VLA_EVAL_BACKUP_DIR}" -czf "$archive" "$timestamp"
rm -rf "$work_dir"

# Retain only the newest $KEEP archives.
ls -1dt "${VLA_EVAL_BACKUP_DIR}"/*.tar.gz 2>/dev/null \
	| tail -n +$((KEEP + 1)) \
	| while IFS= read -r stale; do
		rm -f "$stale"
	done

echo "[backup] wrote ${archive}; retaining the newest ${KEEP}"
