# render_assets — vendored copies for the web video feature

These are byte copies of the repo-root desktop renderers used by the
`POST /api/videos` endpoint (server-side MP4 rendering of a session's
particle-cloud trajectory):

- `cortex_render_videos.py`
- `render_engine_explainer.py`
- `cortex_policy.py`   (imported by cortex_render_videos)

**Source of truth:** repo-root `scripts/`. They are vendored here so they
deploy with `cortex_web` to the prod box (which does not have the repo-root
`scripts/` dir). `app.py` resolves `SCRIPTS_DIR` to the repo-root `scripts/`
when it exists (dev/desktop) and falls back to this dir otherwise (prod web
box). Keep these in sync if the repo-root renderers change.

Deps required at runtime (already present in the box venv): numpy, matplotlib,
imageio_ffmpeg. They are self-contained (no engine import — pure replay).
