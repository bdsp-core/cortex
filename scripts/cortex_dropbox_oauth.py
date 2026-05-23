"""One-shot helper that walks an admin through Dropbox's OAuth flow to
generate the long-lived `refresh_token` that the bundle then uses to
auto-renew its short-lived access token.

Background. Dropbox's console-issued "Generated access tokens" expire
in ~4 hours; once expired, the distributed bundle prints
``Unable to refresh access token without refresh token and app key`` on
every session finalize. Refresh-token mode fixes that — the SDK is
constructed with ``app_key`` + ``refresh_token`` and silently swaps the
access token at Dropbox's /oauth2/token endpoint whenever it expires.

This helper runs the OAuth code flow *once*. The app secret is used
only for the code-for-token exchange and is **not** persisted anywhere
— the resulting refresh token is then long-lived and the bundle uses
only ``app_key`` + ``refresh_token`` at runtime.

Run:

    .venv/bin/python scripts/cortex_dropbox_oauth.py

Then paste the printed refresh token into cortex_config.yaml under the
``dropbox.refresh_token`` field. See docs/CORTEX_DROPBOX_SETUP.md for
the full walkthrough.
"""
from __future__ import annotations

import getpass
import sys
import webbrowser


def _input(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def main():
    try:
        from dropbox import DropboxOAuth2FlowNoRedirect
    except ImportError:
        print("ERROR: the 'dropbox' Python package is not installed in this "
              "environment. Run `pip install dropbox` (or install the repo's "
              "requirements) and try again.", file=sys.stderr)
        sys.exit(1)

    print("CORTEX — Dropbox refresh-token setup")
    print("=" * 50)
    print()
    print("This is a ONE-TIME step. It produces a long-lived refresh token")
    print("that the distributed bundle uses to auto-renew its access token,")
    print("so the bundle keeps uploading indefinitely without manual token")
    print("regeneration.")
    print()
    print("You will need, from your Dropbox app's Settings tab at")
    print("https://www.dropbox.com/developers/apps :")
    print("  1) the App key    (also called client ID)")
    print("  2) the App secret (also called client secret — click 'Show')")
    print()
    print("Both values will be printed at the end of this script for you")
    print("to paste into cortex_config.yaml. For Confidential Dropbox apps")
    print("(the default app type), the SDK needs BOTH at runtime to refresh")
    print("the access token — so both ship in the distributed bundle. The")
    print("app's 'App folder' + files.content.write scoping bounds the")
    print("leak surface to writes into that single folder; see")
    print("docs/CORTEX_DROPBOX_SETUP.md for the full security note.")
    print()

    app_key = _input("App key:    ")
    if not app_key:
        print("aborted — no App key provided.")
        sys.exit(2)
    app_secret = getpass.getpass("App secret: ")
    if not app_secret:
        print("aborted — no App secret provided.")
        sys.exit(2)

    # token_access_type='offline' is what causes Dropbox to mint a refresh
    # token alongside the access token in the code exchange response.
    flow = DropboxOAuth2FlowNoRedirect(
        app_key, app_secret, token_access_type="offline")
    authorize_url = flow.start()

    print()
    print("Opening the Dropbox authorization URL in your browser …")
    print(f"  {authorize_url}")
    try:
        webbrowser.open(authorize_url, new=2)
    except Exception:
        pass
    print()
    print("=" * 60)
    print(" THIS TERMINAL IS WAITING FOR YOU.")
    print("=" * 60)
    print()
    print(" Switch to the browser, log in to Dropbox, click 'Allow'.")
    print(" Dropbox will then show you a long 'Access Code'.")
    print()
    print(" Dropbox's page may say 'Enter this code into <your app name>'.")
    print(" Ignore that wording — there is no place inside Dropbox to enter")
    print(" the code. The 'place' Dropbox means is THIS TERMINAL — you")
    print(" came here from the script that opened that URL.")
    print()
    print(" Copy the access code from the browser, switch back to this")
    print(" terminal, paste at the prompt below, and press Enter.")
    print()
    print("=" * 60)
    print()
    auth_code = _input("Paste the access code here, then press Enter: ")
    if not auth_code:
        print("aborted — no authorization code provided.")
        sys.exit(2)

    try:
        result = flow.finish(auth_code)
    except Exception as e:
        print(f"OAuth exchange failed: {e}", file=sys.stderr)
        sys.exit(3)

    print()
    print("Success — refresh token issued.")
    print("=" * 50)
    print()
    print("Paste these three values into cortex_config.yaml under `dropbox:`")
    print("(leave `access_token` empty in refresh-token mode):")
    print()
    print(f"  app_key:       \"{app_key}\"")
    print(f"  app_secret:    \"{app_secret}\"")
    print(f"  refresh_token: \"{result.refresh_token}\"")
    print()
    print("Then rebuild the bundle:")
    print("  .venv/bin/python scripts/build_internal_test_zip.py")
    print()


if __name__ == "__main__":
    main()
