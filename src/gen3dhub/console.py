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


def info(message: str) -> None:
    console.print(f"[info]i[/info] {message}")


def success(message: str) -> None:
    console.print(f"[success]✓[/success] {message}")


def warn(message: str) -> None:
    err_console.print(f"[warning]![/warning] {message}")


def error(message: str) -> None:
    err_console.print(f"[error]✗[/error] {message}")
