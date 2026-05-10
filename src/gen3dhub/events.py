"""Streaming JSON events for `gen3dhub run --json`.

When the CLI is invoked with `--json`, gen3dhub becomes silent on stdout *except*
for newline-delimited JSON event objects. Subprocess output (pip, model loading,
upstream inference logs) is redirected to stderr by `utils.process.run_streaming`
so stdout stays a clean event stream that an agent can parse line by line.

Event vocabulary used by `run`:

  start                  Initial event with model id, inputs, params, output
  setup_start / setup_complete / setup_failed
                         Wraps adapter.setup() when it actually runs
  post_setup_start / post_setup_complete / post_setup_failed
                         Wraps adapter.post_setup() (credential check)
  inference_start / inference_complete / inference_failed
                         Wraps the actual model inference
  preview_start / preview_complete / preview_failed
                         Wraps the post-run thumbnail render (best-effort)
  validate_complete      One-shot event with the MeshReport fields inlined
  done                   Final event with exit_code; always emitted

Every event has at least `event` and `ts` (ISO-8601 UTC). The `*_complete` and
`*_failed` events from `phase()` also carry `duration_s`.

The mode is process-global (a module-level flag). It's set by the CLI before
running anything; adapters and console functions check it implicitly via
`is_json_mode()`.
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from typing import Any

_json_mode: bool = False


def set_json_mode(enabled: bool) -> None:
    global _json_mode
    _json_mode = enabled


def is_json_mode() -> bool:
    return _json_mode


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def emit(event: str, **fields: Any) -> None:
    """Print a single NDJSON event to stdout. No-op outside JSON mode.

    Uses `default=str` so Path / Enum / unusual objects don't crash the
    serializer — they fall back to their `str()` representation, which is
    almost always what an agent wants anyway.
    """
    if not _json_mode:
        return
    payload = {"event": event, "ts": _now_iso(), **fields}
    sys.stdout.write(json.dumps(payload, default=str) + "\n")
    sys.stdout.flush()


@contextmanager
def phase(name: str, **fields: Any):
    """Context manager that brackets a logical phase with start/complete events.

    On clean exit emits `<name>_complete` with `duration_s`. On exception
    emits `<name>_failed` with `error` + `duration_s` and re-raises so the
    caller still sees the exception. Outside JSON mode this is a near-no-op
    (the emit() calls become no-ops), so wrapping things unconditionally
    is safe.
    """
    start = time.monotonic()
    emit(f"{name}_start", **fields)
    try:
        yield
    except Exception as exc:
        emit(
            f"{name}_failed",
            error=str(exc),
            error_type=type(exc).__name__,
            duration_s=round(time.monotonic() - start, 3),
            **fields,
        )
        raise
    else:
        emit(f"{name}_complete", duration_s=round(time.monotonic() - start, 3), **fields)
