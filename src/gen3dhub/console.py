"""Shared Rich console used across the application for consistent output."""

from __future__ import annotations

from rich.console import Console
from rich.theme import Theme

_THEME = Theme(
    {
        "info": "cyan",
        "success": "bold green",
        "warning": "yellow",
        "error": "bold red",
        "muted": "dim white",
        "model": "bold magenta",
    }
)

console = Console(theme=_THEME)
err_console = Console(theme=_THEME, stderr=True)


def _silent() -> bool:
    """Skip human-readable console output when the CLI is in --json mode.

    Imported lazily to avoid a circular import — events.py is in the same
    package and only some entry points use it.
    """
    from gen3dhub import events

    return events.is_json_mode()


def info(message: str) -> None:
    if _silent():
        return
    console.print(f"[info]i[/info] {message}")


def success(message: str) -> None:
    if _silent():
        return
    console.print(f"[success]✓[/success] {message}")


def warn(message: str) -> None:
    if _silent():
        return
    err_console.print(f"[warning]![/warning] {message}")


def error(message: str) -> None:
    if _silent():
        return
    err_console.print(f"[error]✗[/error] {message}")
