# apps/marketing

Marketing / landing site for **cortexeeg.org**.

Status: **redirect-only** (Phase A decision). The apex `cortexeeg.org` (and
`www`, and the retired `cortex-44-233-29-150.nip.io`) currently 301-redirect to
`https://app.cortexeeg.org` via Caddy — there is no static marketing content yet.

This package is the future home for a real marketing site (static HTML or a
small static-site build). When it exists, the Caddy `cortexeeg.org` vhost
switches from a `redir` to `root * <this build output>` + `file_server`. Until
then this directory is intentionally a placeholder.
