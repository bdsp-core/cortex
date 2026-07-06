# pytest collection guard for the trainer_rd R&D subproject.
#
# This directory is NOT a pytest project. Its suite is script-style: every
# file under tests/ runs as `python3 -m tests.test_<name>` from THIS
# directory (see README.md) and executes its checks on import — several also
# parse argv with argparse, so importing them under pytest crashes collection.
#
# The repo's own pytest suite is pinned to <repo>/tests by `testpaths` in the
# repo pyproject.toml, so a normal `pytest` run from the repo root never
# descends here. This conftest is belt-and-braces for the one residual
# hazard: an explicit `pytest trainer_rd/` invocation. collect_ignore_glob
# below makes pytest ignore every path in this tree, so it collects 0 items
# instead of importing (and crashing on) the script-style tests.
collect_ignore_glob = ["*"]
