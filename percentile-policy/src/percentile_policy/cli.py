"""Command-line interface for building, scoring, and transition evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .builder import QualityPolicy, build_artifact, write_json
from .model import score_candidate, verify_artifact
from .runtime import verify_runtime_metadata, write_runtime
from .transition import TransitionPolicy, evaluate_transition


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _emit(payload: dict[str, Any], output: Path | None) -> None:
    if output:
        write_json(output, payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def _build(args: argparse.Namespace) -> int:
    artifact, audit = build_artifact(
        source_csv=args.source,
        manifest_path=args.manifest,
        bootstrap_replicates=args.bootstrap_replicates,
        quantile_points=args.quantile_points,
        seed=args.seed,
        quality_policy=QualityPolicy(
            min_trials=args.min_trials,
            max_se_ell=args.max_se_ell,
        ),
        created_utc=args.created_utc,
    )
    write_json(args.output, artifact, compact=True)
    write_json(args.audit_output, audit)
    print(
        json.dumps(
            {
                "ok": True,
                "artifact": str(args.output),
                "audit": str(args.audit_output),
                "normId": artifact["normId"],
                "artifactSha256": artifact["artifactSha256"],
                "eligibleFitRows": {
                    domain: item["eligibleFitRows"]
                    for domain, item in artifact["domains"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _verify(args: argparse.Namespace) -> int:
    errors = verify_artifact(_read_json(args.artifact))
    print(json.dumps({"ok": not errors, "errors": errors}, indent=2))
    return 0 if not errors else 1


def _score(args: argparse.Namespace) -> int:
    result = score_candidate(
        _read_json(args.artifact),
        _read_json(args.candidate),
        interval_level=args.interval_level,
    )
    _emit(result, args.output)
    return 0


def _transition(args: argparse.Namespace) -> int:
    policy = TransitionPolicy()
    if args.policy:
        policy = TransitionPolicy(**_read_json(args.policy))
    result = evaluate_transition(_read_json(args.metrics), policy)
    _emit(result, args.output)
    return 0 if result["eligibleForGovernedCutover"] else 2


def _build_runtime(args: argparse.Namespace) -> int:
    metadata = write_runtime(
        _read_json(args.artifact),
        args.metadata_output,
        args.data_output,
        metadata_url=args.metadata_url,
        data_url=args.data_url,
    )
    errors = verify_runtime_metadata(metadata)
    if errors:
        raise ValueError("invalid generated runtime: " + "; ".join(errors))
    print(json.dumps({
        "ok": True,
        "metadata": str(args.metadata_output),
        "data": str(args.data_output),
        "normId": metadata["normId"],
        "normSha256": metadata["normSha256"],
        "metadataSha256": metadata["metadataSha256"],
        "runtimeDataSha256": metadata["runtimeDataSha256"],
        "floatCount": metadata["floatCount"],
    }, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    root = _repo_root()
    policy_root = root / "percentile-policy"
    command = argparse.ArgumentParser(prog="cortex-percentile")
    sub = command.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="build the provisional historical artifact")
    build.add_argument(
        "--source",
        type=Path,
        default=root / "data/engine_inputs/sdt_fits_k7.csv",
    )
    build.add_argument(
        "--manifest",
        type=Path,
        default=root / "data/engine_inputs/MANIFEST_k7.json",
    )
    build.add_argument(
        "--output",
        type=Path,
        default=policy_root / "artifacts/historical-calibration-k7-v1.json",
    )
    build.add_argument(
        "--audit-output",
        type=Path,
        default=policy_root / "reports/historical-cohort-audit.json",
    )
    build.add_argument("--bootstrap-replicates", type=int, default=500)
    build.add_argument("--quantile-points", type=int, default=201)
    build.add_argument("--seed", type=int, default=20260722)
    build.add_argument("--min-trials", type=int, default=20)
    build.add_argument("--max-se-ell", type=float, default=1.0)
    # Part of the governed candidate identity. The default build command must
    # reproduce the checked-in artifact byte-for-byte; a future norm version
    # supplies its own explicit timestamp and ID.
    build.add_argument("--created-utc", default="2026-07-22T00:00:00Z")
    build.set_defaults(func=_build)

    verify = sub.add_parser("verify", help="validate an artifact and its self-hash")
    verify.add_argument("artifact", type=Path)
    verify.set_defaults(func=_verify)

    score = sub.add_parser("score", help="score candidate posterior samples")
    score.add_argument("artifact", type=Path)
    score.add_argument("candidate", type=Path)
    score.add_argument("--interval-level", type=float, default=0.95)
    score.add_argument("--output", type=Path)
    score.set_defaults(func=_score)

    runtime = sub.add_parser(
        "build-runtime", help="pack the governed artifact for API/browser use"
    )
    runtime.add_argument(
        "--artifact", type=Path,
        default=policy_root / "artifacts/historical-calibration-k7-v1.json",
    )
    runtime.add_argument(
        "--metadata-output", type=Path,
        default=root / "cortex_web/apps/web/public/norms/"
        "historical-calibration-k7-provisional-v1.json",
    )
    runtime.add_argument(
        "--data-output", type=Path,
        default=root / "cortex_web/apps/web/public/norms/"
        "historical-calibration-k7-provisional-v1.bin",
    )
    runtime.add_argument(
        "--metadata-url",
        default="/norms/historical-calibration-k7-provisional-v1.json",
    )
    runtime.add_argument(
        "--data-url",
        default="/norms/historical-calibration-k7-provisional-v1.bin",
    )
    runtime.set_defaults(func=_build_runtime)

    transition = sub.add_parser(
        "evaluate-transition",
        help="evaluate whether first-attempt CORTEX-user norms meet cutover gates",
    )
    transition.add_argument("metrics", type=Path)
    transition.add_argument("--policy", type=Path)
    transition.add_argument("--output", type=Path)
    transition.set_defaults(func=_transition)
    return command


def main(argv: list[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        return int(args.func(args))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
