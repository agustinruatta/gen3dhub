"""Typer CLI entry point — `model-selector` command.

Designed to be fully usable in two modes:

1. Non-interactive (for AI agents and scripts): every option can be passed via flags.
   Examples:
       model-selector list
       model-selector setup --model stable-fast-3d
       model-selector run --model stable-fast-3d --image cat.png --output cat.glb
       model-selector doctor --model stable-fast-3d

2. Interactive (for humans): running `model-selector` with no command opens a menu;
   running a command with missing required args triggers questionary prompts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.panel import Panel
from rich.table import Table

from model_selector import __version__, tui
from model_selector.config import Paths
from model_selector.console import console, error, info, success
from model_selector.models.base import ModelInfo, RunRequest
from model_selector.registry import get_adapter, known_model_ids, list_models

app = typer.Typer(
    name="model-selector",
    help="Download, configure, and run AI models from a single CLI/TUI.",
    no_args_is_help=False,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"model-selector {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option("--version", help="Show version and exit.", callback=_version_callback),
    ] = False,
) -> None:
    """If invoked with no subcommand, launch the persistent TUI."""
    if ctx.invoked_subcommand is None:
        _launch_tui()


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@app.command("list")
def list_command() -> None:
    """List the models supported by this tool."""
    models = list_models()
    table = Table(title="Supported models", header_style="bold cyan")
    table.add_column("ID", style="model")
    table.add_column("Name")
    table.add_column("Summary")
    table.add_column("Inputs")
    table.add_column("Output")
    for m in models:
        inputs = ", ".join(f"{i.name}({i.kind.value})" for i in m.inputs)
        table.add_row(m.id, m.display_name, m.summary, inputs, m.output_extension)
    console.print(table)


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------


@app.command("setup")
def setup_command(
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help=f"Model ID. One of: {', '.join(known_model_ids())}"),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Reinstall even if already present.")
    ] = False,
) -> None:
    """Download and install a model into an isolated, per-model environment."""
    paths = Paths.default()
    paths.ensure()
    model_id = model or tui.select_model("Which model do you want to install?")
    adapter = get_adapter(model_id, paths)
    console.print(
        Panel.fit(
            f"[model]{adapter.info.display_name}[/model]\n{adapter.info.summary}\n"
            f"[muted]Homepage:[/muted] {adapter.info.homepage}",
            title="Setup",
            border_style="cyan",
        )
    )
    adapter.setup(force=force)


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


@app.command("doctor")
def doctor_command(
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model ID to diagnose. If omitted, checks all."),
    ] = None,
) -> None:
    """Verify environment, dependencies, and authentication for one or all models."""
    from model_selector.utils.system import system_summary

    paths = Paths.default()
    paths.ensure()
    targets = [model] if model else known_model_ids()

    console.print(f"[muted]Host:[/muted] {system_summary()}")

    failed = False
    for model_id in targets:
        adapter = get_adapter(model_id, paths)
        console.print(
            f"\n[bold]Checking {adapter.info.display_name}[/bold] "
            f"([model]{model_id}[/model])"
        )
        problems = adapter.verify()
        if not problems:
            success("All checks passed.")
        else:
            failed = True
            for p in problems:
                error(p)

    if failed:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@app.command("run")
def run_command(
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help=f"Model ID. One of: {', '.join(known_model_ids())}"),
    ] = None,
    image: Annotated[
        Path | None,
        typer.Option("--image", "-i", help="Path to an input image (for image-input models)."),
    ] = None,
    text: Annotated[
        str | None,
        typer.Option("--text", "-t", help="Input text prompt (for text-input models)."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Path where the produced artifact should be written."),
    ] = None,
    auto_setup: Annotated[
        bool,
        typer.Option(
            "--auto-setup/--no-auto-setup",
            help="If the model isn't installed yet, install it automatically before running.",
        ),
    ] = True,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes", "-y", help="Skip confirmation prompts (for non-interactive use)."
        ),
    ] = False,
) -> None:
    """Run inference. Missing required arguments trigger interactive prompts."""
    paths = Paths.default()
    paths.ensure()

    model_id = model or tui.select_model()
    adapter = get_adapter(model_id, paths)
    model_info = adapter.info

    if not adapter.is_installed:
        if auto_setup and (yes or tui.confirm(f"'{model_id}' is not installed. Install now?")):
            adapter.setup()
        else:
            error(f"Model '{model_id}' is not installed. Run: model-selector setup -m {model_id}")
            raise typer.Exit(code=1)

    inputs = _resolve_inputs(model_info, image=image, text=text)
    output_path = _resolve_output(model_info, inputs=inputs, explicit=output, yes=yes)

    request = RunRequest(inputs=dict(inputs), output_path=output_path)
    info(f"Running [model]{model_id}[/model]…")
    adapter.run(request)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve_inputs(
    model_info: ModelInfo,
    *,
    image: Path | None,
    text: str | None,
) -> dict[str, str | Path]:
    flag_values: dict[str, str | Path | None] = {"image": image, "text": text}

    inputs: dict[str, str | Path] = {}
    missing = []
    for spec in model_info.inputs:
        provided = flag_values.get(spec.kind.value)
        if provided:
            inputs[spec.name] = provided
        elif spec.required:
            missing.append(spec)

    if missing:
        # Interactive fill-in. Reuse the questionary prompts.
        prompted = tui.collect_inputs(model_info)
        for spec in missing:
            if spec.name in prompted:
                inputs[spec.name] = prompted[spec.name]

    return inputs


def _resolve_output(
    model_info: ModelInfo,
    *,
    inputs: dict[str, str | Path],
    explicit: Path | None,
    yes: bool,
) -> Path:
    if explicit is not None:
        return explicit.expanduser()

    image = inputs.get("image")
    stem = Path(str(image)).stem if image is not None else "output"
    default = Path.cwd() / f"{stem}{model_info.output_extension}"

    if yes:
        return default
    return tui.ask_output_path(default)


def _launch_tui() -> None:
    """Launch the persistent Textual TUI. Imported lazily to keep startup fast."""
    from model_selector.tui_app import run as run_tui

    run_tui()


@app.command("tui")
def tui_command() -> None:
    """Launch the persistent interactive TUI. Same as running `model-selector` with no command."""
    _launch_tui()


if __name__ == "__main__":
    app()
