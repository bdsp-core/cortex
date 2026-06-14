# Setting up rclone → Box for the CORTEX backup

The `backup_to_box.sh` script uploads via an rclone "remote" named `box`
(overridable via `CORTEX_RCLONE_REMOTE`). You set this up once on the deploy
box; it persists in `/home/cortex/.config/rclone/rclone.conf` and the systemd
timer just uses it.

Pick **one** of the two auth paths below. The JWT-app path is more solid for
unattended cron because it never expires; the personal-OAuth path is fine for
a pilot.

## Option A — Personal OAuth (5 minutes, fine for a pilot)

This binds rclone to **your** Box account. The token refreshes itself.

```bash
sudo -u cortex bash
cd ~
rclone config
```

Walk-through:

  1. `n` → new remote
  2. name: `box`
  3. Storage: `box` (look for it in the list — it's a built-in backend)
  4. `client_id` and `client_secret`: leave blank (uses rclone's defaults)
  5. `box_config_file`: blank
  6. `access_token`: blank
  7. `box_sub_type`: `user`
  8. Edit advanced config: `n`
  9. Use auto config: `y` if you have a browser on the same machine,
     otherwise `n` and follow the headless-OAuth flow (rclone prints a URL,
     you open it in your laptop browser, paste the resulting token back).
  10. `y` → keep
  11. `q` → quit

Verify:

```bash
rclone ls box:
rclone mkdir box:CORTEX/backups
rclone copy /etc/hostname box:CORTEX/backups/_setup_test
rclone ls    box:CORTEX/backups/_setup_test
rclone purge box:CORTEX/backups/_setup_test
```

## Option B — Box JWT app (15 minutes, best for production)

This creates a Box **server-side application** with its own service account
("App User"). It has no human associated with it; the rclone token is a
private key the app uses to mint short-lived API tokens. Doesn't expire,
survives staff changes, can be scoped to one folder.

  1. Sign in at <https://app.box.com/developers/console> as a Box account
     that can create platform apps.
  2. **Create New App** → "Custom App" → "Server Authentication (with JWT)".
     Name: `CORTEX backups`.
  3. On the app's "Configuration" tab:
     - App Access Level: **App + Enterprise Access**.
     - Application Scopes: Read **and** Write all files.
     - Advanced Features: `Make API calls using the as-user header` (off is
       fine; we don't need it).
  4. Generate a Public/Private Keypair → download the JSON config file.
     Move it to the deploy box at `/home/cortex/.config/box-app.json`,
     `chown cortex:cortex`, `chmod 600`.
  5. Submit the app for "App Authorization" in the Box admin console
     (Enterprise Settings → Apps → Custom Apps → Authorize). Use the
     "Client ID" from the JSON file's `boxAppSettings.clientID`.
  6. Configure rclone:

    ```bash
    sudo -u cortex rclone config
    # n → new remote, name "box", storage "box"
    # box_config_file: /home/cortex/.config/box-app.json
    # box_sub_type:    enterprise
    # auto config: y
    ```

  7. The first call uploads a folder readable only by the app user. To make
     it visible in your Box web UI: create the destination folder
     (`CORTEX/backups`) manually in Box first, then **collaborate** the app
     user onto it as Editor. The app user's email is in the JSON config
     under `enterpriseID`-derived addresses (or grab it from the app's
     "General Settings" tab → "Service Account Info").

Verify the same way as Option A.

## Troubleshooting

- `Couldn't decode error response: invalid character …` — wrong scopes on
  the JWT app; re-edit and re-authorize.
- `403 access_denied_insufficient_permissions` — app not authorized at the
  enterprise level (Option B step 5), or the service account isn't a
  collaborator on the target folder.
- `429 Too Many Requests` — Box rate limit. `backup_to_box.sh` uses small
  files and runs every 6 h; this should never happen organically. If it
  does, add `--tpslimit 4` to the rclone call.
