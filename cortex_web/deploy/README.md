# CORTEX web — deploy

End-to-end deploy on **AWS Lightsail** (Stanford AWS, fits under ~$10/mo):
one Lightsail instance runs Caddy + Postgres + the FastAPI app + the static
SPA + the EEG bundle. CloudFront (free tier, optional) sits in front for
edge caching. The DB + per-session exports back up to **Box** every 6 hours
via the `cortex-backup.timer` systemd unit.

```
                                  ┌──────────────────────────────────────────┐
                                  │   one Lightsail instance ($5 / $10/mo)   │
  user → https://cortex.lab.com → │ Caddy :443 → uvicorn :8000 → Postgres    │
                          (CDN)   │                       └── SPA + /bundle  │
                                  └──────────────────────────────────────────┘
                                          │
                                          ▼
                                  Box  (every 6 h: DB + admin exports)
```

## Cost (pilot, ≤1k sessions/mo)

| Component | $/mo |
|---|---|
| Lightsail `nano_2_0` (1 vCPU, 1 GB) — works but tight | $5 |
| Lightsail `micro_2_0` (1 vCPU, 2 GB) — recommended | **$10** |
| CloudFront in front (within the 1 TB/mo always-free tier) | $0 |
| S3 (not used — bundle lives on the instance) | $0 |
| Box backups (existing storage) | $0 |
| **Total** | **~$10** |

## What's in this directory

```
deploy/
  README.md                    ← you are here
  Caddyfile.template           Caddy site config (TLS auto, reverse-proxy to uvicorn)
  .env.example                 template for /etc/cortex/cortex.env
  systemd/
    cortex.service             the FastAPI service
    cortex-backup.service      Box backup one-shot (called by the timer)
    cortex-backup.timer        every 6 h
  scripts/
    provision.sh               one-shot setup on a fresh Ubuntu 24.04 box
    deploy_app.sh              code update after first provision
    backup_to_box.sh           DB + admin exports → Box (uses rclone)
    rclone-box-setup.md        one-time rclone↔Box auth walkthrough
```

## Step-by-step

### 1. Lightsail instance + DNS

In the Lightsail console:

  1. **Create instance** → Linux/Unix → OS Only → Ubuntu **24.04 LTS**.
  2. Plan: **$10 / month — 2 GB RAM / 60 GB SSD** (the $5 plan works but Postgres
     plus the SPA build squeezes the 1 GB).
  3. **Instance name**: `cortex-prod-1`.
  4. **Networking** → attach a **Static IP** (free while attached).
  5. **Firewall** → open **TCP 80** and **TCP 443**. Leave 22 open for SSH.

DNS: add an `A` record at your DNS provider:
  `cortex.lab.example.org` → the Lightsail static IP.

Wait for the A record to resolve (`dig +short cortex.lab.example.org`) before
running `provision.sh`, since Caddy issues an HTTP-01 cert at startup.

### 2. SSH in + provision

```bash
ssh ubuntu@<your-static-ip>
sudo apt-get update && sudo apt-get install -y git
git clone https://github.com/bdsp-core/ilae-skill-certification-test-multi.git
cd ilae-skill-certification-test-multi/cortex_web
sudo bash deploy/scripts/provision.sh cortex.lab.example.org
```

`provision.sh` is idempotent — safe to rerun. It:

  - installs Caddy + Postgres + Python + Node + rclone
  - creates the `cortex` system user + `/opt/cortex`, `/var/lib/cortex`, `/var/log/cortex`
  - sets up the `cortex` Postgres role + database with a random password
  - clones the repo into `/opt/cortex`, builds a Python venv, runs `npm ci && npm run build`
  - drops the systemd units in place + enables `cortex.service` and `cortex-backup.timer`
  - installs the Caddyfile with your domain substituted in
  - writes `/etc/cortex/cortex.env` with freshly minted secrets

When it's done you should see uvicorn responding on `127.0.0.1:8000` (private)
and Caddy responding with a valid cert on `:443`.

### 3. Ship the EEG bundle

`provision.sh` does **not** download the EEG bundle (it's ~330 MB, gitignored,
the operator's call which version to ship). From your local workstation:

```bash
# build it locally
cortex_app/build_venv/bin/python cortex_web/scripts/prepare_web_bundle.py \
    --bank /path/to/eeg_bank.h5 --version v1.5-k7 --include-spike

# rsync it onto the deploy box
rsync -avz --delete cortex_web/public/bundle/v1.5-k7/ \
    ubuntu@cortex.lab.example.org:/tmp/v1.5-k7/
ssh ubuntu@cortex.lab.example.org \
    "sudo install -d -o cortex -g cortex /opt/cortex/cortex_web/public/bundle/v1.5-k7 \
     && sudo rsync -a --delete /tmp/v1.5-k7/ /opt/cortex/cortex_web/public/bundle/v1.5-k7/ \
     && sudo chown -R cortex:cortex /opt/cortex/cortex_web/public/bundle"
```

(Or scp the h5 bank up and run `prepare_web_bundle.py` on the deploy box.)

### 4. rclone ↔ Box

Follow [`scripts/rclone-box-setup.md`](scripts/rclone-box-setup.md). Pick the
JWT-app path if this is a long-running production deploy; personal OAuth is
fine for a pilot.

Smoke test the backup once you have the remote configured:

```bash
sudo systemctl start cortex-backup.service
sudo tail -n 30 /var/log/cortex/backup.log
# → should show "uploading → box:CORTEX/backups/<year>/<month>/<stamp>"
```

### 5. Mint participant codes

```bash
sudo -u cortex bash -c '
  source <(grep -v "^#" /etc/cortex/cortex.env | sed "s/^/export /")
  /opt/cortex/.venv/bin/python -m server.admin \
      gen --count 50 --prefix cortex --out /tmp/codes.csv
'
sudo cat /tmp/codes.csv      # distribute these — passwords shown ONCE
sudo rm /tmp/codes.csv       # they're stored hashed in the DB
```

### 6. (Optional) CloudFront in front

Cuts first-byte latency for participants outside us-east. **Free** within the
1 TB/mo always-free tier (you're at ~0.3 TB/mo at 1k sessions).

In the AWS console → **CloudFront** → Create distribution:

  - **Origin domain**: `cortex.lab.example.org` (your Caddy host)
  - **Origin protocol**: HTTPS only (Caddy already has a cert)
  - **Viewer protocol policy**: Redirect HTTP → HTTPS
  - **Allowed methods**: GET, HEAD, POST, OPTIONS, PUT, PATCH, DELETE
    (POST is needed for `/api/auth`, `/api/session`, `/api/results`)
  - **Cache policy** (default): `CachingDisabled` for `/api/*` and the
    catch-all; `CachingOptimized` for `/bundle/*` (the path-versioned blobs
    are safe to cache aggressively).
  - **Origin request policy**: `AllViewer` (forwards Authorization).
  - **Alternate domain name (CNAME)**: e.g. `cortex.lab.example.org`.
  - **SSL certificate**: ACM in `us-east-1`.

Add `https://<your-cloudfront-domain>` to `CORTEX_CORS_ORIGINS` in
`/etc/cortex/cortex.env` and restart `cortex.service`.

### 7. Done

```bash
curl -sS https://cortex.lab.example.org/api/health
# → {"ok":true,"service":"cortex-web","version":"1.0"}
```

## Day-to-day operations

| Task | Command |
|---|---|
| Deploy code update | `ssh … 'sudo bash /opt/cortex/cortex_web/deploy/scripts/deploy_app.sh main'` |
| Service status | `sudo systemctl status cortex.service` |
| Tail logs | `sudo journalctl -u cortex -f` |
| Backup now | `sudo systemctl start cortex-backup.service` |
| Backup history | `sudo journalctl -u cortex-backup -n 200` |
| List Box backups | `sudo -u cortex rclone tree box:CORTEX/backups \| head -50` |
| List participants | `sudo -u cortex /opt/cortex/.venv/bin/python -m server.admin list` |
| Restart API | `sudo systemctl restart cortex.service` |
| Reload Caddy (after Caddyfile edit) | `sudo systemctl reload caddy` |

## Restoring from a Box backup

```bash
# 1. pull the most recent snapshot
sudo -u cortex rclone copy box:CORTEX/backups/2026/06/<stamp>/ /tmp/restore/

# 2. stop the API while we restore
sudo systemctl stop cortex.service

# 3a. Postgres restore
sudo -u postgres dropdb cortex
sudo -u postgres createdb -O cortex cortex
gunzip -c /tmp/restore/db.sql.gz \
    | sudo -u cortex psql "$(grep ^CORTEX_DB /etc/cortex/cortex.env | cut -d= -f2-)"

# 3b. SQLite restore (if you're on the SQLite branch)
sudo install -m 640 -o cortex -g cortex /dev/null /var/lib/cortex/cortex.db
sudo -u cortex bash -c 'gunzip -c /tmp/restore/db.sqlite.gz > /var/lib/cortex/cortex.db'

# 4. restart
sudo systemctl start cortex.service
```

## Security checklist (before opening to participants)

  - [ ] `CORTEX_JWT_SECRET` and `CORTEX_ADMIN_TOKEN` in `/etc/cortex/cortex.env`
        are the random ones `provision.sh` generated (and you didn't paste
        the example values).
  - [ ] `/etc/cortex/cortex.env` is mode 640 owned by `cortex:cortex`.
  - [ ] Lightsail firewall: 22 (SSH), 80 (Caddy ACME challenge), 443 only.
  - [ ] SSH is key-only (`sudo passwd --lock ubuntu` once you have a personal
        key authorized).
  - [ ] Postgres is **not** reachable from the public internet (Lightsail
        firewall blocks 5432 by default; Postgres listens on 127.0.0.1 only).
  - [ ] `box:CORTEX/backups` is **not** publicly shared.
  - [ ] At least one successful backup is in Box before you start
        distributing codes.

## When you outgrow this

The single-box deploy is honest at ≤ ~1k sessions/mo and ≤ ~100 concurrent.
Above that, the migration path is:

  1. Move the EEG bundle to S3 + put CloudFront origin on S3 (cuts bundle
     egress off the Lightsail bandwidth allowance).
  2. Lightsail managed Postgres ($15/mo single-AZ, $30 HA) → swap
     `CORTEX_DB` and rerun `cortex.service`.
  3. Multiple Lightsail instances behind a Lightsail load balancer
     ($18/mo).

That migration is `~$30–50/mo` and supports ~10k sessions/mo. Beyond that,
the original AWS playbook (Fargate + RDS + CloudFront) applies.
