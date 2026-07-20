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
    deploy_app.sh              clean staged release after first provision
    release_switch.sh          atomic activation + one-step rollback
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

# rsync it onto the deploy box. Bundles live at /opt/cortex/bundle: OUTSIDE
# the app tree, so vite's public/->dist/ copy can never duplicate 11GB of
# blobs on disk (2026-07-11 disk-pressure fix). Caddy serves /bundle/* from
# root /opt/cortex; the API reads CORTEX_BUNDLE_DIR=/opt/cortex/bundle
# (set in cortex.env).
rsync -avz --delete cortex_web/public/bundle/v1.5-k7/ \
    ubuntu@cortex.lab.example.org:/tmp/v1.5-k7/
ssh cortex.lab.example.org \
    "sudo install -d -o cortex -g cortex /opt/cortex/bundle/v1.5-k7 \
     && sudo rsync -a --delete /tmp/v1.5-k7/ /opt/cortex/bundle/v1.5-k7/ \
     && sudo chown -R cortex:cortex /opt/cortex/bundle"
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

### 6. CloudFront in front of the EEG bundle (LIVE since 2026-07-10)

The bundle blobs are ~200-300 MB per first full sitting and are the box's
dominant egress; they are also perfectly cacheable (path-versioned,
`Cache-Control: immutable`). A dedicated CDN hostname fronts ONLY them —
the SPA and /api stay direct on the Caddy host, so auth, rate limiting and
the in-memory session state are untouched.

Live setup (all in the account that owns the cortexeeg.org Route53 zone):

  - **ACM cert** (us-east-1, DNS-validated): `bundle.cortexeeg.org`
    (`arn:aws:acm:us-east-1:394624473373:certificate/865c81af-d167-4a85-8f44-9ad0232996db`)
  - **CloudFront distribution** `E87M6EKAC1WUZ` (`d1c985pg598sx8.cloudfront.net`):
      * Origin: `app.cortexeeg.org`, HTTPS-only (Caddy is the origin; the
        bundle files never leave the box's disk — no S3 copy to keep in sync).
      * Behavior `/bundle/*`: GET/HEAD, `Managed-CachingOptimized`
        (honors the origin's immutable/max-age headers),
        `Managed-SimpleCORS` (the SPA on app.cortexeeg.org fetch()es
        cross-origin, so blobs need `Access-Control-Allow-Origin: *`).
      * Default behavior: `Managed-CachingDisabled` passthrough. Nothing
        should use the CDN host for the SPA or /api (authed API calls fail
        there by design: Authorization is not forwarded).
  - **Route53**: `bundle.cortexeeg.org` A/AAAA alias → the distribution.

The cutover is one env var — the SPA fetches blobs from whatever base URL
the API hands out (`bundleUrl` in the session/manifest payloads):

```bash
# /etc/cortex/cortex.env
CORTEX_BUNDLE_URL=https://bundle.cortexeeg.org/bundle/<version>
```

then `sudo systemctl restart cortex`. Rollback = set it back to the
same-origin `/bundle/<version>` and restart; Caddy still serves the path
directly. Two coupling points to keep in mind:

  - The site CSP's `connect-src` (Caddyfile) must list
    `https://bundle.cortexeeg.org` or the browser blocks the fetches.
  - Ship new bundle versions to `/opt/cortex/bundle/<version>/` as before;
    the CDN needs no per-release action (new version = new paths = cold
    cache that warms on first fetch).

Cost: $0 within the CloudFront always-free tier (1 TB/mo egress; the pilot
is at ~0.3 TB/mo at 1k sessions).

### 7. Done

```bash
curl -sS https://cortex.lab.example.org/api/health
# → {"ok":true,"service":"cortex-web","version":"1.0"}
```

## Release model

Run deployments from a clean local checkout whose `HEAD` exactly matches
`origin/main`:

```bash
bash cortex_web/deploy/scripts/deploy_app.sh
```

The deploy refuses dirty or unpushed commits, takes a fresh Box backup, uploads
to `/tmp`, and builds under `/opt/cortex/releases/<timestamp>-<sha>` before it
changes the live application. `/opt/cortex/cortex_web` is a stable symlink to
the active release. The first staged deployment preserves the former physical
directory as a legacy release.

Activation restarts the API and requires a database-backed loopback health
check. Public deep health and phone-browser checks run immediately afterward.
Any failed activation, public health check, Caddy reload, or browser check
switches the stable path back to the previous release and restarts the API.

Postgres data, `/etc/cortex/cortex.env`, `/opt/cortex/.venv`, and
`/opt/cortex/bundle` remain outside release directories. Never restore the
database to roll back application code: additive migrations are intentionally
backward-compatible, and restoring an older database would discard participant
activity recorded after the backup.

Manual one-step rollback:

```bash
ssh cortex-prod \
  'sudo /opt/cortex/cortex_web/deploy/scripts/release_switch.sh rollback'
```

## Day-to-day operations

| Task | Command |
|---|---|
| Deploy pushed `main` | From a clean local checkout: `bash cortex_web/deploy/scripts/deploy_app.sh` |
| Roll back one release | `ssh cortex-prod 'sudo /opt/cortex/cortex_web/deploy/scripts/release_switch.sh rollback'` |
| Service status | `sudo systemctl status cortex.service` |
| Tail logs | `sudo journalctl -u cortex -f` |
| Backup now | `sudo systemctl start cortex-backup.service` |
| Backup history | `sudo journalctl -u cortex-backup -n 200` |
| List Box backups | `sudo -u cortex rclone tree box:CORTEX/backups \| head -50` |
| List participants | `sudo -u cortex /opt/cortex/.venv/bin/python -m server.admin list` |
| Restart API | `sudo systemctl restart cortex.service` |
| Reload Caddy (after Caddyfile edit) | `sudo systemctl reload caddy` |

## Email delivery monitoring (SES events)

Set up 2026-07-09 after a verification email went undelivered with zero
visibility. All mail sent from the `cortexeeg.org` SES identity (us-west-2,
account 394624473373) flows through the **`cortex-transactional`**
configuration set — it is the identity's *default* configuration set, so it
applies to SMTP + API sends alike with no app config. Its `errors-to-sns`
event destination publishes BOUNCE / COMPLAINT / REJECT / RENDERING_FAILURE /
DELIVERY_DELAY events (full per-message JSON: recipient, timestamp, SMTP
diagnostic) to the SNS topic `cortex-ses-events`, which delivers to the
operator's email subscription.

  - No error email = SES accepted + delivered to the recipient's MX. A
    report of "no email received" from there on means the recipient's
    junk/quarantine (org mail gateways), not our pipeline.
  - Aggregate counts: CloudWatch `AWS/SES` namespace (Send/Delivery/Bounce).
  - To re-create or add a subscriber:
    `aws sns subscribe --topic-arn arn:aws:sns:us-west-2:394624473373:cortex-ses-events --protocol email --notification-endpoint <addr> --region us-west-2`
  - Signup-side guard: `/api/register` rejects email domains with no DNS
    MX/A records (helpers.email_domain_deliverable, fails open), and the
    signup form suggests corrections for near-miss domains
    (apps/web/src/emailSuggest.ts).

### Bounce webhook (verify-screen feedback)

The app itself is also a subscriber of `cortex-ses-events`: an HTTPS
subscription posts every event to `POST /api/ses/events?token=…`
(`routers/ses_events.py`). On a **permanent** bounce the participant row is
flagged `email_undeliverable_utc`, and the verify screen — which polls
`POST /api/verify/status` while the user waits for their code — switches to
"we couldn't deliver, fix your address". The flag clears automatically when
any verify/reset code is confirmed.

Setup (already live; repeat only on a rebuild):

```bash
# 1. mint a webhook token into /etc/cortex/cortex.env and restart:
#      CORTEX_SNS_WEBHOOK_TOKEN=$(openssl rand -hex 24)
#      CORTEX_SNS_TOPIC_ARN=arn:aws:sns:us-west-2:394624473373:cortex-ses-events
# 2. subscribe (the endpoint auto-confirms):
aws sns subscribe --region us-west-2 \
    --topic-arn arn:aws:sns:us-west-2:394624473373:cortex-ses-events \
    --protocol https \
    --notification-endpoint "https://app.cortexeeg.org/api/ses/events?token=<TOKEN>"
```

Related env: `CORTEX_PUBLIC_ORIGIN` (e.g. `https://app.cortexeeg.org`) makes
verify/reset emails include a one-click link that lands in the SPA with the
code pre-filled (`mailer._one_click_link` → `apps/web/src/deepLink.ts`);
unset, emails carry only the typed code.

The endpoint 404s when `CORTEX_SNS_WEBHOOK_TOKEN` is unset (dev/CI default).
Auth is the capability token + a TopicArn allowlist — SNS message signatures
are NOT verified; the only action is a low-stakes, self-healing UI flag
(rationale in the module docstring). Add signature verification before ever
letting this endpoint do anything more consequential.

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

### Restore drill (do this without waiting for a disaster)

An untested backup is hope, not a backup. `restore_check.sh` rehearses the
whole restore path — pulls the newest snapshot from Box, restores it into a
**scratch** database (`cortex_restore_check`), sanity-checks the participant/
session/trial tables, and drops the scratch. It never touches the live DB or
the service:

```bash
sudo /opt/cortex/cortex_web/deploy/scripts/restore_check.sh
# → ✓ RESTORE CHECK PASS participants=… sessions=… training_trials=…
```

Run it after provisioning, after any Postgres upgrade, and periodically
(quarterly is fine at this scale). It also accepts a specific snapshot
(`box:CORTEX/backups/YYYY/MM/<stamp>`) or a local `db.sql.gz`/`db.sqlite.gz`.

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
  - [ ] `restore_check.sh` has PASSed at least once against a real Box
        snapshot (a backup that has never been restored is not a backup).

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
