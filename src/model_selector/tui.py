"""Interactive prompts (questionary-based) used when CLI args are missing.

The intent is *not* to be a full-screen TUI — it's a thin interactive layer that
fills in arguments the user (a human, not an agent) didn't provide on the command
line. AI agents and scripts pass everything via flags and skip this entirely.
"""

from __future__ import annotations

from pathlib import Path

import questionary

from model_selector.models.base import InputKind, ModelInfo
from model_selector.registry import list_models


def select_model(prompt: str = "Which model do you want to use?") -> str:
    models = list_models()
    if not models:
        raise RuntimeError("No models registered.")
    choices = [
        questionary.Choice(title=f"{m.display_name}  —  {m.summary}", value=m.id) for m in models
    ]
    answer = questionary.select(prompt, choices=choices).ask()
    if answer is None:
        raise SystemExit("Cancelled.")
    return str(answer)


def collect_inputs(model: ModelInfo) -> dict[str, str]:
    """Prompt the user for each declared input slot of the model."""
    collected: dict[str, str] = {}
    for spec in model.inputs:
        question = f"{spec.description}"
        if spec.kind is InputKind.IMAGE:
            value = questionary.path(question, only_directories=False).ask()
        else:  # text, future kinds
            value = questionary.text(question).ask()
        if value is None:
            raise SystemExit("Cancelled.")
        if spec.required and not value:
            raise SystemExit(f"Input '{spec.name}' is required.")
        collected[spec.name] = str(value)
    return collected


def ask_output_path(default: Path) -> Path:
    answer = questionary.path(
        f"Where should the output be written? (default: {default})",
        default=str(default),
    ).ask()
    if answer is None:
        raise SystemExit("Cancelled.")
    return Path(answer).expanduser()


def confirm(message: str, *, default: bool = True) -> bool:
    answer = questionary.confirm(message, default=default).ask()
    if answer is None:
        raise SystemExit("Cancelled.")
    return bool(answer)
