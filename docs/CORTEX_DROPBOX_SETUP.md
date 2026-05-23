# CORTEX — Dropbox result delivery setup

Each completed test session writes two result CSVs — a per-question
detail file and a one-row session summary — and can upload them straight
to a Dropbox folder you control. This is configured once, by you, before
distributing the test.

The bundle supports two auth modes:

  * **Refresh-token mode (preferred — keeps working indefinitely).** The
    bundle holds an app key + long-lived refresh token; the Dropbox SDK
    transparently renews the short-lived access token when it expires.
    Set this up once and the bundle keeps uploading until you explicitly
    revoke the refresh token.

  * **Access-token mode (legacy — short-lived).** The bundle holds a
    short-lived `sl.u.AG…` access token that Dropbox issues from the app
    console's "Generated access token" button. These expire in **~4
    hours**, after which the bundle prints `Unable to refresh access
    token without refresh token and app key` and stops uploading. Useful
    only for one-off dev sessions; not appropriate for distributed
    field use.

This doc walks through the preferred refresh-token setup. The legacy
access-token setup is at the bottom.

## 1. Create a scoped Dropbox app

1. Go to <https://www.dropbox.com/developers/apps> and click **Create app**.
2. Choose **Scoped access**.
3. For access type choose **App folder** — the safe option: the app (and
   its credentials) can only ever touch its own folder,
   `/Apps/<your-app-name>/`, and nothing else in your Dropbox.
4. Name the app (e.g. `cortex-internal-test`) and create it.

## 2. Give it write permission

On the app's **Permissions** tab, enable **`files.content.write`** — and
nothing else (it needs no read or account scope). Click **Submit**.

## 3. Copy the app key and app secret

On the **Settings** tab, copy two values:

  * **App key** (also called client ID)
  * **App secret** (click "Show" — also called client secret)

Both ship in the distributed bundle. See the security note at the
bottom of this doc; with App-folder + `files.content.write`-only
scoping the leak surface is bounded to writes into that single folder.

> **Why app_secret too?** Dropbox apps come in two flavors:
> Confidential (the default; what "Create app" produces) and Public
> (PKCE). Confidential apps require app_secret on EVERY OAuth call —
> not just the one-time refresh-token exchange. The SDK's runtime
> refresh of the access token includes app_secret in the request,
> so the bundle must carry it.

## 4. Generate a refresh token

From the repo root, run the one-shot helper:

```
.venv/bin/python scripts/cortex_dropbox_oauth.py
```

It prompts for your app key + app secret, opens the Dropbox
authorization URL in your browser, asks you to paste the authorization
code Dropbox shows you, then exchanges that code for a long-lived
refresh token. The script prints all three values at the end so you
can paste them directly into `cortex_config.yaml`.

## 5. Fill in `cortex_config.yaml`

Copy `cortex_config.example.yaml` to `cortex_config.yaml` if you haven't
already, then set:

```yaml
dropbox:
  app_key:       "<paste the App key from step 3>"
  app_secret:    "<paste the App secret from step 3>"
  refresh_token: "<paste the refresh token from step 4>"
  access_token:  ""             # leave empty in refresh-token mode
  folder:        "/results"
```

`folder` is a path *inside* the app folder, so the CSVs land in
`/Apps/<your-app-name>/results/` on your Dropbox.

## 6. Verify

Run `python scripts/cortex_smoke.py` (or any internal session). On
finalize the console prints:

```
  Dropbox client constructed (refresh-token mode)
  uploaded 2 result CSV(s) to Dropbox /results/
```

and the two CSVs appear in `/Apps/<your-app-name>/results/` on
dropbox.com. If the bundle is left running for hours and Dropbox cycles
the access token under the hood, the upload still succeeds — the SDK
silently re-fetches a new access token from the refresh token.

## 7. Rebuild the bundle

The credentials live in `cortex_config.yaml`, which is bundled into
`dist/cortex-internal-test.zip`. After updating the config, rebuild:

```
.venv/bin/python scripts/build_internal_test_zip.py
```

---

## Legacy: access-token-only mode

For a quick one-off dev session, you can skip the refresh-token flow
and use the Dropbox app console's short-lived token directly.

1. Repeat steps 1-2 above.
2. On the **Settings** tab, under **OAuth 2 → Generated access token**,
   click **Generate** and copy the token.
3. In `cortex_config.yaml`, leave `app_key` + `refresh_token` empty and
   set:

   ```yaml
   dropbox:
     app_key:       ""
     refresh_token: ""
     access_token:  "<paste the sl.u.AG… token here>"
     folder:        "/results"
   ```

**The token expires in ~4 hours.** When it does, the bundle prints
`Unable to refresh access token without refresh token and app key` on
every session finalize. To recover: regenerate the token, paste into
config, rebuild the bundle. Or migrate to refresh-token mode (steps
3-7 above) once.

---

## Notes

- **All credentials ship inside the distributed zip** — app_key +
  app_secret + refresh_token in refresh-token mode, or access_token in
  legacy mode. Because the app is App-folder-scoped with only
  `files.content.write`, the worst a leaked credential bundle allows
  is *writing files into that one folder* — it cannot read results,
  cannot see the rest of your Dropbox, cannot delete anything. Still,
  treat the zip as a shared credential. To rotate: delete the app's
  refresh tokens on the Dropbox console, regenerate app_secret if you
  want, re-run step 4, rebuild the bundle.
- Every session uploads uniquely-named files (`<uuid>_<name>_*.csv`),
  so any number of test-takers finishing at once upload with no
  collision.
- Upload failure (no internet, expired credentials) is non-fatal — the
  session still completes and the CSVs are kept locally under
  `results/sessions/<session_id>/`.
- `pip install dropbox` must be present in the test-taker's environment;
  it is pinned in `requirements-cortex.txt`.
- To merge everyone's CSVs into one roster, download the Dropbox folder
  and run `python scripts/combine_results.py <folder>`.
