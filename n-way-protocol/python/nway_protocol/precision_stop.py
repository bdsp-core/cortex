"""Persistent sidecar client for the UNCHANGED production Precision policy.

The stopping decision itself always runs in TypeScript
(``src/precision_cli.ts`` -> ``src/precision_bridge.ts`` ->
``cortex_web/apps/web/engine/precision_policy.ts``); this module only moves
JSON lines across a pipe.  One sidecar subprocess is kept per worker process
(lazy init) so a qualification replicate pays the node startup cost once.

Failure policy: a sidecar that cannot start or that rejects a request raises
``PrecisionSidecarError`` immediately.  There is deliberately NO fallback to
a Python re-implementation -- a promotion run must exercise the frozen
production policy or fail loudly.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

PROTOCOL_ROOT = Path(__file__).resolve().parents[2]
_TSX = PROTOCOL_ROOT.parent / "cortex_web" / "node_modules" / ".bin" / "tsx"
_CLI_SOURCE = PROTOCOL_ROOT / "src" / "precision_cli.ts"
_CLI_BUNDLE = PROTOCOL_ROOT / ".precision-cli-dist" / "precision_cli.mjs"
_BUILD_SCRIPT = PROTOCOL_ROOT / "scripts" / "build_precision_cli.sh"

# Frozen production stopping configuration.  The authoritative values live in
# cortex_web/apps/web/engine/precision_policy.ts (PRECISION_PER_DOMAIN_CAP,
# PRECISION_N_MIN) and are applied by the sidecar itself when the init request
# carries no overrides; the init response echoes them and the golden parity
# fixture pins them.  They are mirrored here only so the harness can align its
# own-cap safety default with the policy's per-domain ceiling.
PRECISION_PER_DOMAIN_CAP = 60
PRECISION_N_MIN = 20


class PrecisionSidecarError(RuntimeError):
    """The Precision sidecar failed to start, died, or rejected a request."""


def ensure_sidecar_built() -> None:
    """Build the node bundle for the sidecar (no-op when tsx can run the TS).

    Called once from the parent process before workers fan out; the build
    script publishes atomically so concurrent calls are safe.
    """
    if _TSX.exists():
        return
    completed = subprocess.run(
        ["bash", str(_BUILD_SCRIPT)], cwd=PROTOCOL_ROOT,
        capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise PrecisionSidecarError(
            "failed to build the Precision sidecar bundle via "
            f"{_BUILD_SCRIPT}:\n{completed.stderr.strip()}"
        )


class PrecisionStopClient:
    """One persistent sidecar subprocess speaking line-delimited JSON."""

    def __init__(self) -> None:
        if _TSX.exists():
            command = [str(_TSX), str(_CLI_SOURCE)]
        elif _CLI_BUNDLE.exists():
            command = ["node", str(_CLI_BUNDLE)]
        else:
            raise PrecisionSidecarError(
                f"Precision sidecar bundle missing at {_CLI_BUNDLE}; run "
                f"{_BUILD_SCRIPT} (or call ensure_sidecar_built()) first"
            )
        self._stderr = tempfile.TemporaryFile(mode="w+")
        try:
            self._process = subprocess.Popen(
                command,
                cwd=PROTOCOL_ROOT,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                text=True,
                env={"CORTEX_PRECISION_SIDECAR": "1", "PATH": _node_path()},
            )
        except OSError as error:
            raise PrecisionSidecarError(
                f"failed to launch the Precision sidecar ({command}): {error}"
            ) from error
        pong = self.request({"op": "ping"})
        if pong.get("op") != "pong":
            raise PrecisionSidecarError(f"unexpected sidecar handshake: {pong}")

    def request(self, payload: dict) -> dict:
        """Send one wire request verbatim; return the parsed ok-response."""
        assert self._process.stdin is not None and self._process.stdout is not None
        try:
            self._process.stdin.write(json.dumps(payload) + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise PrecisionSidecarError(self._death_message()) from error
        line = self._process.stdout.readline()
        if not line:
            raise PrecisionSidecarError(self._death_message())
        response = json.loads(line)
        if not response.get("ok"):
            raise PrecisionSidecarError(
                f"Precision sidecar rejected {payload.get('op')}: "
                f"{response.get('error')}"
            )
        return response

    def _death_message(self) -> str:
        self._process.poll()
        self._stderr.seek(0)
        tail = self._stderr.read()[-2000:]
        return (
            "Precision sidecar exited "
            f"(returncode={self._process.returncode}); stderr tail:\n{tail}"
        )

    def init_session(
        self,
        session_id: str,
        corr_l: list[list[float]],
        corr_t: list[list[float]],
        band_edges: list[list[float]],
    ) -> dict:
        """Construct the frozen-production policy (session.ts:277-296)."""
        return self.request({
            "op": "init",
            "sessionId": session_id,
            "corrL": corr_l,
            "corrT": corr_t,
            "precisionBandEdges": band_edges,
        })

    def evaluate(
        self,
        session_id: str,
        administered: dict | None,
        cloud_t: np.ndarray,
        cloud_l: np.ndarray,
        cloud_w: np.ndarray,
        last_rejuvenation: dict | None,
        n_per_task: list[int],
        bank: dict,
    ) -> dict:
        """Post-update policy consultation (advance.ts:498-514 ordering)."""
        n, k = cloud_t.shape
        response = self.request({
            "op": "evaluate",
            "sessionId": session_id,
            "administered": administered,
            "state": {
                "N": int(n),
                "K": int(k),
                "t": cloud_t.ravel().tolist(),
                "l": cloud_l.ravel().tolist(),
                "w": cloud_w.tolist(),
                "lastRejuvenation": last_rejuvenation,
            },
            "nPerTask": [int(x) for x in n_per_task],
            "bank": bank,
        })
        return response["result"]

    def close(self) -> None:
        if self._process.stdin is not None:
            try:
                self._process.stdin.close()
            except OSError:
                pass
        self._process.wait(timeout=10)
        self._stderr.close()


def _node_path() -> str:
    """PATH for the sidecar: the launching interpreter's node must resolve."""
    return os.environ.get("PATH", "/usr/bin:/bin")


_CLIENT: PrecisionStopClient | None = None
_CLIENT_PID: int | None = None


def get_client() -> PrecisionStopClient:
    """Lazy per-process singleton (one sidecar per pool worker).

    The pid guard prevents a forked worker from inheriting — and interleaving
    writes on — the parent process's sidecar pipe.
    """
    global _CLIENT, _CLIENT_PID
    if (
        _CLIENT is None
        or _CLIENT_PID != os.getpid()
        or _CLIENT._process.poll() is not None
    ):
        _CLIENT = PrecisionStopClient()
        _CLIENT_PID = os.getpid()
    return _CLIENT
