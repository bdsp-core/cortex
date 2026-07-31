# CORTEX web deployment and operations

This directory contains the supported single-host deployment, backup, health,
and rollback automation for the canonical `cortex_web` application. The
runbook intentionally omits cloud account identifiers, instance addresses,
credentials, and participant data.

Read [`../docs/PRODUCTION_BASELINE.md`](../docs/PRODUCTION_BASELINE.md) before
operating the live service. It records the latest verified release,
non-secret environment assumptions, API surface, latency interpretation, and
named rollback anchors. Re-check the live release stamp; the snapshot is not a
substitute for runtime verification.

## Architecture

```text
participant → HTTPS/Caddy
                  ├─ immutable SPA
                  ├─ externally managed /bundle
                  └─ /api → FastAPI → PostgreSQL

systemd: cortex.service + cortex-backup.timer
persistent state: database, environment/secrets, EEG bundle, backup remote
immutable state: versioned application releases
```

The database, `/etc/cortex/cortex.env`, the shared Python environment, and
`/opt/cortex/bundle` remain outside application release directories. An
application rollback must never restore an older database or overwrite the
bundle.

## Directory contents

```text
deploy/
  Caddyfile.template
  .env.example
  systemd/
    cortex.service
    cortex-backup.service
    cortex-backup.timer
  scripts/
    provision.sh
    deploy_app.sh
    release_switch.sh
    backup_to_box.sh
    restore_check.sh
    observe_precision_compute.sql
    smoke_domain.sh
    rclone-box-setup.md
```

## Initial provisioning

Provision an approved Ubuntu 24.04 host with inbound SSH, HTTP, and HTTPS only.
Attach stable DNS before provisioning so Caddy can obtain a certificate.

From a clean clone on the host:

```bash
cd ilae-skill-certification-test-multi/cortex_web
sudo bash deploy/scripts/provision.sh app.example.org
```

The script installs the service dependencies, creates the service account and
database, generates initial secrets, installs systemd/Caddy configuration, and
builds the application. It is intended to be idempotent, but every rerun must
still be reviewed and followed by deep health and backup checks.

Provisioning writes conservative bootstrap values (including compute `off`, a
500-item sampled draw, and the legacy default bundle path); it is not a clone
of the current live environment. Deliberately reconcile its non-secret values
with [`../docs/PRODUCTION_BASELINE.md`](../docs/PRODUCTION_BASELINE.md) and the
governed bundle before exposure.

Never copy `.env.example` values into production unchanged. Keep the completed
environment file mode 640 and owned by the service account.

## EEG bundle

Provisioning does not download or build EEG data. An authorized operator must
stage the governed production bundle separately under
`/opt/cortex/bundle/<version>/`, validate its manifest hash, and set the
corresponding bundle URL/version in `/etc/cortex/cortex.env`.

For the current Precision profile the logical version is `v1.6-k7-35k`.
Building it requires an approved external HDF5 bank:

```bash
.venv/bin/python cortex_web/apps/web/scripts/prepare_web_bundle.py \
  --bank /approved/path/eeg_bank.h5 \
  --version v1.6-k7-35k \
  --include-spike
```

Do not place HDF5 banks or generated bundle blobs in Git. A CDN can front the
versioned immutable `/bundle` paths, but authenticated API traffic must remain
outside that cache path. Keep the CDN origin, CSP `connect-src`, CORS response,
and configured bundle URL synchronized.

## Backups

Configure the protected backup remote using
[`scripts/rclone-box-setup.md`](scripts/rclone-box-setup.md), then verify both a
backup and a scratch restore:

```bash
sudo systemctl start cortex-backup.service
sudo journalctl -u cortex-backup.service -n 100 --no-pager
sudo /opt/cortex/cortex_web/deploy/scripts/restore_check.sh
```

`restore_check.sh` restores into scratch state, validates expected tables, and
removes the scratch database. It must not stop or replace the live service.
Run it after provisioning, after database upgrades, and periodically.

## Release model

Deploy only from a clean local checkout whose `HEAD` exactly matches
`origin/main`:

```bash
# From the repository root:
(cd cortex_web && \
  CORTEX_QUALITY_PYTHON=.venv/bin/python \
  CORTEX_SMOKE_SYNTHETIC=1 \
    npm run quality:browser)
bash cortex_web/deploy/scripts/deploy_app.sh
```

Run the full quality gate before deployment. `deploy_app.sh` intentionally
does not rerun Vitest or pytest on the host; it verifies release provenance,
buildability, Python compilation, readiness, and the live participant surface.
The staged runtime excludes test source patterns, caches, local databases,
secrets, build output, and the externally managed EEG bundle.

The deployment workflow:

1. rejects dirty, detached, or unpushed release input;
2. completes a fresh backup;
3. uploads a clean runtime-only staged source tree;
4. installs locked dependencies and builds a versioned immutable release;
5. switches the stable application path atomically;
6. restarts the API and checks database-backed loopback health;
7. verifies public deep health and browser smoke; and
8. automatically restores the previous application release if activation
   fails.

Manual one-release rollback:

```bash
ssh cortex-prod \
  'sudo /opt/cortex/cortex_web/deploy/scripts/release_switch.sh rollback'
```

For Git history, use normal revert commits. Never rewrite `main`, edit an old
release directory, or restore persistent state merely to roll back code.

Immediately before and after an activation, resolve identity from all three
places rather than trusting a branch name:

```bash
git fetch origin main
git rev-parse HEAD origin/main
ssh cortex-prod 'readlink -f /opt/cortex/cortex_web; \
  cat /opt/cortex/cortex_web/RELEASE'
curl --fail --silent --show-error \
  'https://app.cortexeeg.org/api/health?deep=1'
```

An otherwise healthy deep response may report `bank=not_loaded_yet` after a
restart because readiness deliberately avoids parsing the large manifest.
Session creation is the lazy bank-availability check. `bank=missing` after a
load attempt or a session-creation 503 is not a successful bank check. The
health field reports the default-bank cache, not a separate Precision-bank
cache.

## Day-to-day operations

| Task | Command |
|---|---|
| Deploy pushed `main` | `bash cortex_web/deploy/scripts/deploy_app.sh` from a clean checkout |
| Roll back one release | `ssh cortex-prod 'sudo /opt/cortex/cortex_web/deploy/scripts/release_switch.sh rollback'` |
| Service status | `sudo systemctl status cortex.service` |
| Tail service logs | `sudo journalctl -u cortex.service -f` |
| Client-error stream | `sudo journalctl -u cortex.service \| grep cortex.clienterr` |
| Sampled entry/privacy/telemetry access log | `sudo tail -f /var/log/caddy/cortex-telemetry-access.json` |
| Deep health | `curl --fail --silent --show-error https://app.cortexeeg.org/api/health?deep=1` |
| Verify the live CSP | `node cortex_web/apps/web/scripts/csp_verify_live.mjs` (loads the deployed site in Chromium and fails on a CSP violation or a missing Google sign-in button; run after any CSP or auth-provider change) |
| Verify the live phone surface | `node cortex_web/apps/web/scripts/phone_smoke.mjs https://app.cortexeeg.org` (also runs automatically at the end of every deploy) |
| Observe compute rollout | `sudo -u postgres psql -d cortex -v since_utc=<UTC_TIMESTAMP> -f /opt/cortex/cortex_web/deploy/scripts/observe_precision_compute.sql` |
| Backup now | `sudo systemctl start cortex-backup.service` |
| Backup history | `sudo journalctl -u cortex-backup.service -n 200 --no-pager` |
| Validate newest backup | `sudo /opt/cortex/cortex_web/deploy/scripts/restore_check.sh` |
| List participants | `sudo -u cortex /opt/cortex/.venv/bin/python -m api.admin list` |
| Restart API | `sudo systemctl restart cortex.service` |
| Reload Caddy | `sudo systemctl reload caddy` |

The rollout observation SQL emits aggregate, privacy-safe telemetry only. Its
interpretation and rollback thresholds are documented in
`../docs/WEB_WORKER_ROLLOUT_OPERATIONS.md`.

### Client-error alert policy

`POST /api/client-error` is intentionally public so failures before sign-in
remain visible, but public input never enters immediate email alerting.
Unauthenticated events are normalized, journaled within global,
per-fingerprint, and per-IP limits, then summarized once for the preceding UTC
day. A valid bearer token permits an ordinary application crash to use the
immediate, cooldown-limited alert channel. The known
`HTMLAnchorElement`/`__reactFiber$...` circular-JSON injected-DOM signature is
always journal-only.

Cooldown and aggregate rate state survive service restarts in Postgres. Stored
request metadata is deliberately coarse: Origin relationship, Sec-Fetch-Site,
browser-major/OS family, and a daily rotating HMAC request fingerprint. Raw
IP, raw Origin, raw user agent, credentials, and query strings are not stored
in the aggregate. Caddy separately samples only `/`, `/privacy`, and
`/api/client-error`, with subnet-masked IPs, hashed user agents, sensitive
headers removed, query strings redacted, and three-day log rotation.

The one-release rollback target is mutable operational state. Inspect
`/opt/cortex/release-state/previous` before running rollback; do not assume it
is the accepted performance baseline named in the production snapshot.

## Participant-code and administrative commands

Run administrative commands as the service account with the production
environment loaded. The current module path is `api.admin`, not the retired
`server.admin` path. Treat generated code files and exports as sensitive,
short-lived material; place them outside the release tree and remove them once
transferred through an approved channel.

## Email delivery monitoring

Transactional email delivery uses the configured provider and event
destination. Bounce/complaint events may be sent to the authenticated
`/api/ses/events` endpoint so the verification UI can report an undeliverable
address. Provider account IDs, topic ARNs, subscription tokens, recipients,
and SMTP credentials belong in protected operational configuration, not this
repository.

The webhook must fail closed when its token is absent and may only perform the
documented self-healing delivery-status update. Add provider-signature
verification before expanding its authority.

## Restore procedure

A real restore is a separately authorized destructive operation. Before
starting:

1. identify and checksum the exact backup;
2. confirm whether the target is PostgreSQL or SQLite;
3. take a fresh backup of current state;
4. stop the API; and
5. have an explicit rollback plan for the restore itself.

Use `restore_check.sh` for routine validation. Do not copy a generic drop/create
sequence from documentation into production without resolving the exact target
and authorization.

## Security and release checklist

- [ ] Release commit is clean, pushed, reviewed, and matches `origin/main`.
- [ ] Web `quality:browser` gate passed with the supported Python/Node versions.
- [ ] Secrets are generated, protected, and absent from the source archive.
- [ ] PostgreSQL is bound privately and not exposed through the public firewall.
- [ ] SSH is key-only and restricted to operators.
- [ ] EEG banks, participant exports, databases, and backup snapshots are not
      tracked by Git.
- [ ] A fresh backup completed and a scratch restore has passed.
- [ ] Deep health, CSP, worker parity, and participant UI smoke pass after
      activation.
- [ ] The previous immutable release remains available for one-step rollback.
