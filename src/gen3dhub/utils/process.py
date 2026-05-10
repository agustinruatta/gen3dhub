"""Subprocess helpers that stream output via Rich and raise on non-zero exit."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from gen3dhub.console import console, err_console


class CommandError(RuntimeError):
    """Raised when a subprocess returns a non-zero exit code."""

    def __init__(self, command: Sequence[str], returncode: int) -> None:
        super().__init__(f"Command failed (exit {returncode}): {' '.join(command)}")
        self.command = list(command)
        self.returncode = returncode


def run_streaming(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    description: str | None = None,
) -> None:
    """Run a command, streaming stdout/stderr to the user. Raises CommandError on failure.

    The output is intentionally not captured — for long-running tasks (cloning, pip install,
    inference) the user wants to see progress live.
    """
    # In --json mode the CLI promises stdout is a pure NDJSON event stream,
    # so subprocess stdout has to go to stderr — otherwise pip / model loading
    # logs would interleave with our events and break agent parsers. Outside
    # JSON mode, inherit the parent's streams so the user sees live progress.
    from gen3dhub import events

    if events.is_json_mode():
        stdout_target = sys.stderr
    else:
        if description:
            console.print(f"[muted]$ {description}[/muted]")
        console.print(f"[muted]  → {' '.join(command)}[/muted]")
        stdout_target = None  # inherit from parent

    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(env) if env is not None else None,
        check=False,
        stdout=stdout_target,
    )
    if completed.returncode != 0:
        err_console.print(f"[error]✗ command failed with exit {completed.returncode}[/error]")
        raise CommandError(command, completed.returncode)


def run_capture(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Run a command and return its stdout. Raises CommandError on failure."""
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(env) if env is not None else None,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise CommandError(command, completed.returncode)
    return completed.stdout
