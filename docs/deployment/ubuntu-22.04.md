# Ubuntu 22.04 deployment runbook (single 4090 host)

This guide brings the VLA evaluation service up on a single Ubuntu 22.04 LTS box
with one NVIDIA RTX 4090. The service runs entirely under Docker Compose; only
Caddy publishes a port (443/TLS). Everything else lives on an internal Docker
network.

> Docker is not available on the development machine, so the runtime checks
> (`docker compose config --quiet`, the GPU smoke, the VLM smoke) are listed
> below as steps to run **on the Ubuntu host**. The compose file, Dockerfiles,
> Caddyfile, and shell scripts were validated by YAML parsing, path checks, and
> `bash -n` during development.

## 1. Operating system baseline

```bash
lsb_release -a            # Description: Ubuntu 22.04 LTS
sudo apt-get update
sudo apt-get -y upgrade
```

Ubuntu 22.04 LTS reaches **end of standard support in April 2027**. Before that
date, either:

- upgrade to the next LTS (e.g. 24.04) during a maintenance window, or
- enable **Ubuntu Pro** (`sudo pro attach <token>`) for Extended Security
  Maintenance (ESM) coverage of the base packages beyond 2027-04.

Plan the migration now; do not run an unsupported LTS in production.

## 2. NVIDIA driver

```bash
nvidia-smi                # confirm the driver + 4090 are visible
# If not installed:
sudo apt-get install -y nvidia-driver-535   # or the latest driver series
sudo reboot
nvidia-smi                # must list the RTX 4090 with a working CUDA version
```

## 3. Docker Engine + Compose plugin

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"        # log out/in afterwards
docker --version
docker compose version                  # Compose plugin v2
```

## 4. NVIDIA Container Toolkit

The toolkit makes the host GPU visible inside containers (`--gpus all`
equivalent via the compose device reservation).

```bash
distribution=$(. /etc/os-release; echo "$ID$VERSION_ID")
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## 5. Host directories and ownership

All application containers run as the non-root uid **1001** (`vlaeval`). The host
directories must be owned by 1001 so volume writes succeed.

```bash
sudo mkdir -p /srv/vla-eval/{app,config/profiles,data/db,data/credentials,data/inbox,data/runs,data/staging,logs,models,secrets}
sudo chown -R 1001:1001 /srv/vla-eval
sudo chmod 700 /srv/vla-eval/secrets /srv/vla-eval/data/credentials
```

> Note: the compose mounts use the canonical path `/srv/vla-eval/...`. If you
> used a different root, adjust both the host paths in `docker-compose.yml` and
> the values inside `/srv/vla-eval/config/app.yaml`.

## 6. Application configuration

Copy the example config and the `.env` file, then edit:

```bash
sudo tee /srv/vla-eval/config/app.yaml >/dev/null <<'YAML'
data_root: /srv/vla-eval/data
database_url: sqlite:////srv/vla-eval/data/db/app.sqlite3
redis_url: redis://redis:6379/0
model_path: /srv/vla-eval/data/models/Qwen2.5-VL-7B-Instruct
session_secret: "${VLA_EVAL_SESSION_SECRET}"
remote_sources:
  lab-a:
    host: 10.0.0.8
    port: 22
    username: eval-read
    key_path: /run/secrets/lab_a_key
    known_hosts_path: /run/secrets/known_hosts
    roots:
      - /data/rollouts
YAML

cp .env.example .env
# Edit .env: set VLA_EVAL_SESSION_SECRET (openssl rand -hex 32)
sudo install -m 600 -o 1001 -g 1001 .env /srv/vla-eval/app/.env
```

Place the app source under `/srv/vla-eval/app` (this repository). Place SSH
credentials under `/srv/vla-eval/secrets/` (uid 1001 readable):

```bash
sudo install -m 600 -o 1001 -g 1001 lab_a_key      /srv/vla-eval/secrets/lab_a_key
ssh-keyscan -H 10.0.0.8 | sudo tee /srv/vla-eval/secrets/known_hosts >/dev/null
```

## 7. Build and start the stack

Build the web image first (the transfer worker image is derived from it):

```bash
cd /srv/vla-eval/app
docker compose config --quiet          # validate the compose file
docker compose build web
docker compose build                   # builds transfer-worker (FROM vla-eval-web) + evaluation-worker
docker compose up -d
docker compose ps
```

## 8. Caddy internal TLS certificate

`deploy/Caddyfile` serves a single site on 443 with `tls internal`, which makes
Caddy mint a certificate from its **internal CA**. There is no plaintext HTTP
port.

1. Edit `deploy/Caddyfile` and replace `vla-eval.local` with this host's FQDN or
   IP address.
2. Caddy stores its internal CA under the `caddy_data` volume. To trust it on
   client browsers, export the root and install it:

   ```bash
   docker compose exec caddy caddy trust        # installs the root CA system-wide on the host
   ```

   Alternatively, retrieve `~/Library/Application Support/Caddy/pki/authorities/local/root.crt`
   (via `docker cp`) and import it into each client's trust store.

Verify the site over HTTPS (the `-k` accepts the internal cert until you install
the root CA):

```bash
curl -kfsS https://localhost/health      # => {"status":"ok"}
```

## 9. Runtime verification (run on the host)

These are the checks that cannot run on the development machine:

```bash
# Compose file is well-formed and references resolve.
docker compose config --quiet

# The web health endpoint is green (DB + Redis + data root).
docker compose ps                        # web is "healthy"
curl -kfsS https://localhost/health

# The GPU is visible to the evaluation worker.
docker compose run --rm evaluation-worker nvidia-smi

# The VLM stack loads and CUDA is available.
docker compose run --rm evaluation-worker python -c \
  "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

## 10. SSH host key and read-only rrsync / SFTP account (data ingress)

The host must also *receive* datasets pushed from lab machines, if you use the
inbound-push path. Create a locked-down account:

```bash
sudo addgroup --system sftp-pusher
sudo useradd --system --gid sftp-pusher --home /srv/vla-eval/data/inbox --shell /usr/sbin/nologin eval-push
sudo usermod -p '!' eval-push
# Install the pusher's public key in /srv/vla-eval/.ssh/authorized_keys (uid 1001 owned).
```

For **read-only rsync** (lab machines can push but not delete), wrap the key with
`rrsync` in `authorized_keys`:

```
command="/usr/bin/rrsync -wo /srv/vla-eval/data/inbox",no-pty,no-port-forwarding,no-X11-forwarding,no-agent-forwarding ssh-ed25519 AAAA... lab-a-push-key
```

Ensure the server host key is pinned on the lab side (`~/.ssh/known_hosts`) so
pushes fail loudly on a man-in-the-middle.

```bash
# On the host, keep a stable SSH host key:
sudo ssh-keygen -A
```

## 11. Optional SMB share (alternative data ingress)

If lab machines prefer SMB (see design §12):

```bash
sudo apt-get install -y samba
sudo smbpasswd -a eval-push            # set a strong password
# Add a read/write share for /srv/vla-eval/data/inbox, force user = eval-push (uid 1001).
sudo systemctl enable --now smbd
```

Restrict the share to the lab VLAN in `/etc/samba/smb.conf` (`hosts allow = …`).
SMB traffic should stay on the LAN; do not expose 445 beyond the lab network.

## 12. Disable sleep and enable on boot

A single 4090 server must stay awake and survive reboots:

```bash
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
sudo systemctl enable docker
```

## 13. Backups (daily systemd timer)

`deploy/backup.sh` performs a consistent online SQLite backup (via
`sqlite3 .backup`, safe under WAL) and archives the small `config/` tree.

> **Raw video and dataset binaries are NOT included.** They are large and follow
> the company storage policy (dedicated capacity on the NAS/object store). This
> backup only protects the small, critical database + configuration state.

Install the timer:

```bash
sudo apt-get install -y sqlite3
sudo install -m 755 deploy/backup.sh /usr/local/sbin/vla-eval-backup
sudo tee /etc/systemd/system/vla-eval-backup.service >/dev/null <<'UNIT'
[Unit]
Description=VLA evaluation config + DB backup

[Service]
Type=oneshot
EnvironmentFile=/srv/vla-eval/app/.env
ExecStart=/usr/local/sbin/vla-eval-backup
User=1001
UNIT

sudo tee /etc/systemd/system/vla-eval-backup.timer >/dev/null <<'UNIT'
[Unit]
Description=Daily VLA evaluation backup

[Timer]
OnCalendar=*-*-* 03:00:00
Persistent=true

[Install]
WantedBy=timers.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable --now vla-eval-backup.timer
```

`VLA_EVAL_BACKUP_DIR` (in `.env`) selects the destination; the script keeps the
newest 30 archives and prunes older ones.

## 14. Ubuntu 22.04 → 2027-04 EOL plan

- Standard support for Ubuntu 22.04 ends **April 2027**.
- Before that date: either upgrade to the next LTS (re-test the GPU stack end to
  end) or attach **Ubuntu Pro** for ESM.
- Re-verify the CUDA base image tag in `deploy/Dockerfile.evaluation` against the
  driver present after any OS upgrade.

## 15. Restart and crash recovery

The two workers recover from a restart differently because their workloads have
different durability guarantees. The web process is stateless and can always be
restarted without data loss.

### Transfer worker (rsync) restart

The transfer worker writes into a per-job **staging** directory under
`data/staging/<import_id>` using rsync `--partial`/`--append-verify`, so an
in-progress transfer leaves resumable partial bytes on disk. A restart of the
transfer worker therefore **keeps the staging directory and resumes** the
in-flight import where it stopped:

```bash
docker compose restart transfer-worker
docker compose logs -f transfer-worker
```

If the worker was killed mid-transfer, the in-flight import job is marked
`INTERRUPTED` on the next `recover-jobs` sweep; the staging bytes are preserved
and a retry resumes rather than re-downloading from scratch.

### Evaluation worker restart

Evaluations are **not** auto-recomputed. If the evaluation worker crashes or is
restarted while a job is in `METRICS`/`VLM`/`REPORT`, that job is recorded as
`INTERRUPTED` (a terminal-but-retryable state) by `recover_interrupted_jobs` and
must be **retried manually from the web UI** (the "重试" button on the job
page) or the CLI. It is never silently recomputed, because the metrics/VLM
artifacts may be half-written and the user must explicitly confirm a rerun:

```bash
# Mark in-flight evaluations as INTERRUPTED after a crash/reboot:
docker compose run --rm web python -m vla_eval.cli recover-jobs
```

Then, in the web UI, open each `INTERRUPTED` evaluation and click **重试**. The
rerun resumes from the last completed stage boundary (METRICS → optional VLM →
REPORT) when resumable artifacts are present, otherwise it recomputes from
METRICS.

> Web app restart never loses state: job progress is persisted to SQLite on
> every state/progress callback, so closing the browser or restarting the web
> container does not interrupt or lose a running evaluation -- reopening the
> job page shows the current persisted progress (the in-process e2e test
> `tests/e2e/test_evaluation_workflow.py` asserts this acceptance criterion).

## 16. Record measured host versions (fill in on the 4090)

The GPU smoke (§9), restart-recovery (§15), and the version table below **cannot
be validated on the development machine** (no NVIDIA GPU, no Docker, no 4090).
Run them on the Ubuntu host on first install and record the measured values
here so the next operator can detect drift on upgrade. Replace each `_TBD_`.

| Component                 | Command to capture                                  | Measured |
| ------------------------- | --------------------------------------------------- | -------- |
| NVIDIA driver / 4090      | `nvidia-smi` (CUDA version line)                    | _TBD_    |
| Docker Engine             | `docker --version`                                  | _TBD_    |
| Compose plugin            | `docker compose version`                            | _TBD_    |
| CUDA runtime (container)  | `docker compose run --rm evaluation-worker nvidia-smi` | _TBD_ |
| PyTorch (container)       | `docker compose run --rm evaluation-worker python -c "import torch; print(torch.__version__, torch.version.cuda)"` | _TBD_ |
| GPU visible to PyTorch    | `... python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"` | _TBD_ |
| Eval model weight id      | directory hash / commit of `data/models/Qwen2.5-VL-7B-Instruct` | _TBD_ |
| rsync (host)              | `rsync --version` (must be >= 3.2.7)                | _TBD_    |

Re-run the §9 GPU smoke after any driver, Docker, CUDA, or PyTorch upgrade and
update this table.

## Operational commands cheat sheet

```bash
docker compose logs -f web               # follow web logs
docker compose logs -f evaluation-worker
docker compose restart web
docker compose down                       # stop everything (data volumes persist)
docker compose pull && docker compose up -d   # apply image updates

# Create the first admin user non-interactively:
VLA_EVAL_INITIAL_PASSWORD='...' docker compose run --rm web \
  python -m vla_eval.cli create-user admin --admin

# Recover interrupted jobs after a crash/reboot:
docker compose run --rm web python -m vla_eval.cli recover-jobs

# Rescan the inbox for ready datasets:
docker compose run --rm web python -m vla_eval.cli scan-datasets
```
