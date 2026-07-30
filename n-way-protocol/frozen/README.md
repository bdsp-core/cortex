# Frozen F1 baseline

This directory preserves the n-way implementation exactly as it stood before
the post-1,200-particle/30-MH R&D iteration authorized on 2026-07-20.

Baseline archive:

- `F1_1200P30_20260720.tar.gz`
- SHA-256:
  `015068f250827bcfa42b8ec3d923b473591079ffd60f12674d0ae884267c4a52`
- Per-file digest list: `F1_1200P30_20260720_FILES.sha256`
- Files preserved: 52

Before the archive was created, the baseline passed:

- 26 TypeScript tests (one opt-in performance test skipped by the standard
  suite).
- 9 Python tests.
- The opt-in 35,000-segment selector benchmark.

The archive includes the formal-SBC and production-particle reevaluation
artifacts that justify retaining this baseline. It excludes only cache files,
Python bytecode, and this `frozen/` directory.

## Verification

From `n-way-protocol`:

```bash
sha256sum -c <<'EOF'
015068f250827bcfa42b8ec3d923b473591079ffd60f12674d0ae884267c4a52  frozen/F1_1200P30_20260720.tar.gz
EOF
```

For a full per-file recovery check, extract to a new empty directory and run:

```bash
sha256sum -c /absolute/path/to/F1_1200P30_20260720_FILES.sha256
```

Never extract the archive over a working candidate. Recover it into a new
directory, verify it, and then promote it through the normal qualification
path.
