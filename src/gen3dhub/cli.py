"""Typer CLI entry point — `gen3dhub` command.

Designed to be fully usable in two modes:

1. Non-interactive (for AI agents and scripts): every option can be passed via flags.
   Examples:
       gen3dhub list
       gen3dhub setup --model stable-fast-3d
       gen3dhub run --model stable-fast-3d --image cat.png --output cat.glb
       gen3dhub doctor --model stable-fast-3d

2. Interactive (for humans): running `gen3dhub` with no command opens a menu;
   running a command with missing required args triggers questionary prompts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.panel import Panel

from gen3dhub import __version__, tui
from gen3dhub.config import Paths
from gen3dhub.console import console, error, info, success, warn
from gen3dhub.models.base import ModelInfo, ParamKind, ParamSpec, RunRequest
from gen3dhub.registry import get_adapter, known_model_ids, list_models

app = typer.Typer(
    name="gen3dhub",
    help=(
        "Hub for AI models that generate 3D assets — download, configure, and run.\n\n"
        "AGENT QUICKSTART (non-interactive):\n"
        "  gen3dhub doctor                                  # exit 0 if healthy\n"
        "  gen3dhub setup --model <id>                      # one-time install\n"
        "  gen3dhub run --model <id> --image <p> --output <q> --yes\n\n"
        "Run `gen3dhub agent` for the full agent-oriented usage guide.\n"
        "Run `gen3dhub <cmd> --help` for per-subcommand flags."
    ),
    no_args_is_help=False,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"gen3dhub {__version__}")
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
    """List the models supported by this tool, with strengths, weaknesses,
    and a fit assessment against the current host."""
    from gen3dhub.utils.system import assess_fit, system_summary

    console.print(f"[muted]Host:[/muted] {system_summary()}\n")
    for m in list_models():
        inputs = ", ".join(f"{i.name} ({i.kind.value})" for i in m.inputs)
        strengths = "\n".join(f"  • {s}" for s in m.strengths)
        weaknesses = "\n".join(f"  • {w}" for w in m.weaknesses)
        fit = assess_fit(m.hardware)
        fit_color, fit_icon = {
            "ok": ("green", "✓"),
            "warn": ("yellow", "⚠"),
            "error": ("red", "✗"),
        }[fit.severity]
        fit_block = (
            f"[bold {fit_color}]{fit_icon} On this machine[/bold {fit_color}]\n"
            f"  {fit.headline}"
        )
        if fit.detail:
            fit_block += f"\n  [dim]{fit.detail}[/dim]"

        if m.params:
            param_lines = []
            for p in m.params:
                if p.kind is ParamKind.SELECT and p.choices:
                    type_hint = " | ".join(p.choices)
                else:
                    type_hint = p.kind.value
                param_lines.append(
                    f"  • [bold]{p.name}[/bold] = {p.default!r}  "
                    f"[muted]({type_hint})[/muted]"
                )
            params_block = (
                "[bold magenta]⚙ Parameters[/bold magenta] "
                "[muted](pass via `--param NAME=VALUE` or set in the TUI)[/muted]\n"
                + "\n".join(param_lines)
                + "\n\n"
            )
        else:
            params_block = ""

        body = (
            f"{m.description}\n\n"
            f"[bold cyan]→ Best for[/bold cyan]\n  {m.best_for}\n\n"
            f"[bold green]✓ Strong[/bold green]\n{strengths}\n\n"
            f"[bold yellow]✗ Weak[/bold yellow]\n{weaknesses}\n\n"
            f"{fit_block}\n\n"
            f"{params_block}"
            f"[muted]Inputs:[/muted] {inputs}    "
            f"[muted]Output:[/muted] {m.output_extension}"
        )
        console.print(
            Panel(
                body,
                title=f"[model]{m.id}[/model]  ·  {m.display_name}",
                border_style="cyan",
                padding=(1, 2),
            )
        )


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
    """Download and install a model, then prompt for any required credentials.

    Idempotent: re-running with no --force is a no-op for the install side, but
    `post_setup` is always re-evaluated so credentials can be (re)configured
    without a full reinstall.
    """
    import sys

    paths = Paths.default()
    paths.ensure()
    model_id = model or tui.select_model("Which model do you want to install?")
    adapter = get_adapter(model_id, paths)
    console.print(
        Panel.fit(
            f"[model]{adapter.info.display_name}[/model]\n{adapter.info.description}\n"
            f"[muted]Homepage:[/muted] {adapter.info.homepage}",
            title="Setup",
            border_style="cyan",
        )
    )
    adapter.setup(force=force)
    adapter.post_setup(interactive=sys.stdin.isatty())


# ---------------------------------------------------------------------------
# uninstall
# ---------------------------------------------------------------------------


@app.command("uninstall")
def uninstall_command(
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help=f"Model ID. One of: {', '.join(known_model_ids())}"),
    ] = None,
    all_: Annotated[
        bool,
        typer.Option(
            "--all", help="Uninstall every model. Mutually exclusive with --model."
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes", "-y", help="Skip the per-model confirmation prompt."
        ),
    ] = False,
) -> None:
    """Remove an installed model's repo + venv to reclaim disk space.

    Does NOT touch the Hugging Face weights cache (`~/.cache/huggingface/`),
    which is shared across HF tools. To clean those, see
    `huggingface-cli scan-cache` and `huggingface-cli delete-cache`.
    """
    import shutil

    from gen3dhub.utils.system import directory_size_bytes, format_bytes

    if all_ and model:
        error("Pass either --model or --all, not both.")
        raise typer.Exit(2)

    paths = Paths.default()
    paths.ensure()

    if all_:
        targets = [m for m in known_model_ids() if paths.model_dir(m).exists()]
        if not targets:
            info("Nothing to uninstall — no model directories present.")
            return
    elif model:
        targets = [model]
    else:
        # No flag → interactive pick. Show only installed models.
        installed = [m for m in known_model_ids() if paths.model_dir(m).exists()]
        if not installed:
            info("Nothing to uninstall — no model directories present.")
            return
        targets = [tui.select_model("Which model do you want to uninstall?")]

    total_freed = 0
    for model_id in targets:
        model_dir = paths.model_dir(model_id)
        if not model_dir.exists():
            warn(f"'{model_id}' is not installed (no dir at {model_dir}).")
            continue
        size_bytes = directory_size_bytes(model_dir)
        size_human = format_bytes(size_bytes)
        if not yes and not tui.confirm(
            f"Remove {model_dir} ({size_human})?", default=False
        ):
            info(f"Skipped '{model_id}'.")
            continue
        info(f"Removing {model_dir}")
        shutil.rmtree(model_dir)
        success(f"Uninstalled '{model_id}' (freed ~{size_human})")
        total_freed += size_bytes

    if total_freed > 0:
        success(f"Total freed: {format_bytes(total_freed)}")
        info(
            "Note: model weights downloaded by Hugging Face are kept in "
            "~/.cache/huggingface/ (shared across HF tools). To inspect or "
            "clean them, run `huggingface-cli scan-cache` and "
            "`huggingface-cli delete-cache`."
        )


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
    from gen3dhub.utils.system import system_summary

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
    mesh: Annotated[
        Path | None,
        typer.Option(
            "--mesh",
            help=(
                "Path to an existing 3D mesh (for mesh-input models like paint3d). "
                "Accepts .obj / .glb / .ply / .off."
            ),
        ),
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
    cpu: Annotated[
        bool,
        typer.Option(
            "--cpu",
            help=(
                "Force CPU inference (10-60x slower; useful when the GPU runs out of "
                "VRAM or for headless servers without CUDA)."
            ),
        ),
    ] = False,
    param: Annotated[
        list[str] | None,
        typer.Option(
            "--param", "-p",
            help=(
                "Set a model-specific parameter as KEY=VALUE (repeat for multiple). "
                "Run `gen3dhub list` to see each model's parameters. "
                "Example: --param texture_resolution=2048 --param remesh_option=quad"
            ),
        ),
    ] = None,
    preview: Annotated[
        bool,
        typer.Option(
            "--preview/--no-preview",
            help=(
                "Render a 4-angle PNG thumbnail next to the output GLB after a "
                "successful run. Best-effort — preview failure never aborts the run."
            ),
        ),
    ] = True,
    json_: Annotated[
        bool,
        typer.Option(
            "--json",
            help=(
                "Stream NDJSON events to stdout (start, *_start/_complete/_failed, "
                "validate_complete, done). Implies --yes; subprocess output is "
                "redirected to stderr so stdout stays a clean event stream."
            ),
        ),
    ] = False,
) -> None:
    """Run inference. Missing required arguments trigger interactive prompts."""
    import sys
    import time

    from gen3dhub import events
    from gen3dhub import history as hist

    if json_:
        # Pure NDJSON on stdout. Imply --yes so we never block on a prompt
        # that the agent can't answer.
        events.set_json_mode(True)
        yes = True

    paths = Paths.default()
    paths.ensure()

    # Treat --yes as "no prompts of any kind" — agents typically pair it with
    # piped input, and a TTY check alone would be too lenient.
    interactive = sys.stdin.isatty() and not yes

    model_id = model or tui.select_model()
    adapter = get_adapter(model_id, paths)
    model_info = adapter.info

    inputs = _resolve_inputs(model_info, image=image, text=text, mesh=mesh)
    output_path = _resolve_output(model_info, inputs=inputs, explicit=output, yes=yes)
    params = _parse_params(model_info, param or [])

    request = RunRequest(
        inputs=dict(inputs),
        output_path=output_path,
        params=params,
        extra={"force_cpu": "1"} if cpu else {},
    )

    events.emit(
        "start",
        model=model_id,
        inputs={k: str(v) for k, v in inputs.items()},
        params={k: str(v) for k, v in params.items()},
        output=str(output_path) if output_path else None,
    )

    info(f"Running [model]{model_id}[/model]…")
    start = time.monotonic()
    produced: Path | None = None
    preview_path: Path | None = None
    exit_code = 0

    try:
        if not adapter.is_installed:
            if auto_setup and (
                yes or tui.confirm(f"'{model_id}' is not installed. Install now?")
            ):
                with events.phase("setup", model=model_id):
                    adapter.setup()
                with events.phase("post_setup", model=model_id):
                    adapter.post_setup(interactive=interactive)
            else:
                error(
                    f"Model '{model_id}' is not installed. "
                    f"Run: gen3dhub setup -m {model_id}"
                )
                raise typer.Exit(code=1)
        else:
            with events.phase("post_setup", model=model_id):
                adapter.post_setup(interactive=interactive)

        with events.phase("inference", model=model_id):
            produced = adapter.run(request)

        if preview and produced is not None:
            with events.phase("preview", output=str(produced)):
                preview_path = _try_render_preview(produced)

        if produced is not None:
            _validate_and_report(produced)
    except typer.Exit as exc:
        exit_code = exc.exit_code if isinstance(exc.exit_code, int) else 1
        raise
    except Exception:
        exit_code = 1
        raise
    finally:
        duration = time.monotonic() - start
        events.emit("done", exit_code=exit_code, output=str(produced) if produced else None)
        hist.append(
            paths,
            hist.HistoryEntry(
                id=hist.make_id(),
                timestamp=hist.now_iso(),
                model=model_id,
                inputs={k: str(v) for k, v in inputs.items()},
                params={k: str(v) for k, v in params.items()},
                output=str(produced) if produced is not None else None,
                preview=str(preview_path) if preview_path is not None else None,
                duration_s=duration,
                exit_code=exit_code,
            ),
        )


def _try_render_preview(glb_path: Path) -> Path | None:
    """Render a thumbnail next to `glb_path`. Returns the path on success,
    None on failure. Failures are warnings, never fatal — the actual inference
    output is the source of truth; the preview is purely additive UX.
    """
    preview_path = glb_path.with_suffix(".preview.png")
    try:
        from gen3dhub.utils.preview import render_thumbnail

        render_thumbnail(glb_path, preview_path)
    except Exception as exc:
        warn(f"Preview generation skipped: {exc}")
        return None
    info(f"Preview: {preview_path}")
    return preview_path


def _validate_and_report(glb_path: Path) -> dict | None:
    """Run mesh validation, emit a `validate_complete` event in JSON mode, and
    print a human-readable summary otherwise. Best-effort: failures don't
    propagate — validation is a quality signal, not a precondition.
    """
    from gen3dhub import events

    try:
        from gen3dhub.utils.validation import format_report_human, validate_mesh

        report = validate_mesh(glb_path)
    except Exception as exc:
        warn(f"Validation skipped: {exc}")
        return None

    report_dict = report.as_dict()
    events.emit("validate_complete", **report_dict)
    if not events.is_json_mode():
        console.print()
        console.print(format_report_human(report))
    return report_dict


# ---------------------------------------------------------------------------
# validate (standalone)
# ---------------------------------------------------------------------------


@app.command("validate")
def validate_command(
    path: Annotated[
        Path,
        typer.Argument(
            help="Path to a 3D mesh file (.glb / .obj / .ply / .off / .stl)."
        ),
    ],
    json_: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit a single JSON object (one line) instead of a human report.",
        ),
    ] = False,
) -> None:
    """Inspect any mesh file: vertex/triangle counts, materials, manifoldness, etc.

    Useful as a sanity check on outputs from `gen3dhub run`, but works on any
    mesh — third-party assets, hand-sculpted exports, etc. Same metrics that
    `run` auto-reports after a successful inference.
    """
    import json as _json

    from gen3dhub.utils.validation import format_report_human, validate_mesh

    try:
        report = validate_mesh(path)
    except Exception as exc:
        error(str(exc))
        raise typer.Exit(code=1) from None

    if json_:
        # Single line so this composes with `jq` and shell loops.
        print(_json.dumps(report.as_dict(), default=str))
        return

    console.print(format_report_human(report))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve_inputs(
    model_info: ModelInfo,
    *,
    image: Path | None,
    text: str | None,
    mesh: Path | None,
) -> dict[str, str | Path]:
    flag_values: dict[str, str | Path | None] = {
        "image": image,
        "text": text,
        "mesh": mesh,
    }

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

    # Pick the most representative input as the basis for the default
    # filename: mesh first (the thing being textured) for mesh-input models,
    # image otherwise. For mesh-input models we suffix `_painted` so the
    # output doesn't silently overwrite the input mesh.
    mesh = inputs.get("mesh")
    image = inputs.get("image")
    if mesh is not None:
        stem = f"{Path(str(mesh)).stem}_painted"
    elif image is not None:
        stem = Path(str(image)).stem
    else:
        stem = "output"
    default = Path.cwd() / f"{stem}{model_info.output_extension}"

    if yes:
        return default
    return tui.ask_output_path(default)


def _parse_params(model_info: ModelInfo, raw: list[str]) -> dict[str, object]:
    """Validate and coerce `--param KEY=VALUE` pairs against the model's spec.

    Unknown keys, missing `=`, and out-of-range values raise typer.BadParameter
    with a message that lists the model's allowed parameters — so the user
    learns the surface from the error rather than from a separate `--help`.
    """
    if not raw:
        return {}
    declared: dict[str, ParamSpec] = {p.name: p for p in model_info.params}

    if not declared:
        # Caller passed --param but this model declares none. Fail loudly so the
        # user notices instead of silently ignoring.
        offending = ", ".join(entry.split("=", 1)[0] for entry in raw)
        raise typer.BadParameter(
            f"Model '{model_info.id}' takes no tunable parameters; got: {offending}"
        )

    result: dict[str, object] = {}
    for entry in raw:
        if "=" not in entry:
            raise typer.BadParameter(
                f"--param expects KEY=VALUE, got {entry!r}. "
                f"Available for '{model_info.id}': {_format_param_help(model_info)}"
            )
        key, _, value = entry.partition("=")
        key = key.strip()
        value = value.strip()
        spec = declared.get(key)
        if spec is None:
            raise typer.BadParameter(
                f"Unknown parameter {key!r} for model '{model_info.id}'.\n"
                f"Available: {_format_param_help(model_info)}"
            )
        try:
            result[key] = _coerce_param(value, spec)
        except ValueError as exc:
            raise typer.BadParameter(f"--param {key}: {exc}") from None
    return result


def _coerce_param(raw: str, spec: ParamSpec) -> object:
    if spec.kind is ParamKind.INT:
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"expected an integer, got {raw!r}") from exc
    if spec.kind is ParamKind.FLOAT:
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(f"expected a number, got {raw!r}") from exc
    if spec.kind is ParamKind.BOOL:
        normalized = raw.strip().lower()
        if normalized in ("true", "1", "yes", "y", "on"):
            return True
        if normalized in ("false", "0", "no", "n", "off"):
            return False
        raise ValueError(f"expected a boolean (true/false), got {raw!r}")
    if spec.kind is ParamKind.SELECT:
        if spec.choices and raw not in spec.choices:
            raise ValueError(
                f"value {raw!r} is not one of {', '.join(spec.choices)}"
            )
        return raw
    return raw  # ParamKind.TEXT


def _format_param_help(model_info: ModelInfo) -> str:
    """Compact "name(kind, default=…)" listing used in --param error messages."""
    bits = []
    for p in model_info.params:
        if p.kind is ParamKind.SELECT and p.choices:
            bits.append(f"{p.name}={'|'.join(p.choices)}")
        else:
            bits.append(f"{p.name}=<{p.kind.value}>")
    return ", ".join(bits) if bits else "(none)"


def _launch_tui() -> None:
    """Launch the persistent Textual TUI. Imported lazily to keep startup fast."""
    from gen3dhub.tui_app import run as run_tui

    run_tui()


@app.command("tui")
def tui_command() -> None:
    """Launch the persistent interactive TUI. Same as running `gen3dhub` with no command."""
    _launch_tui()


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------


@app.command("history")
def history_command(
    limit: Annotated[
        int,
        typer.Option(
            "--limit", "-n", help="How many recent entries to show. Ignored with --all."
        ),
    ] = 20,
    all_: Annotated[
        bool, typer.Option("--all", help="Show every recorded run, ignoring --limit.")
    ] = False,
    json_: Annotated[
        bool,
        typer.Option(
            "--json",
            help=(
                "Emit one JSON object per line (newline-delimited). "
                "Designed for agents and scripts."
            ),
        ),
    ] = False,
    rerun: Annotated[
        str | None,
        typer.Option(
            "--rerun",
            help=(
                "Print the equivalent `gen3dhub run …` command for the run with "
                "this id (or id prefix). Doesn't execute it — copy and paste to "
                "re-run intentionally."
            ),
        ),
    ] = None,
) -> None:
    """Show recent runs, or print a re-run command for a past run.

    Reads from `~/.cache/gen3dhub/history.jsonl` (one JSON object per line).
    """
    import json as _json
    import shlex
    from dataclasses import asdict

    from rich.table import Table

    from gen3dhub import history as hist

    paths = Paths.default()

    if rerun:
        entry = hist.find(paths, rerun)
        if entry is None:
            error(f"No run found matching id {rerun!r}")
            raise typer.Exit(2)
        cmd = ["gen3dhub", "run", "--model", entry.model]
        for kind, value in entry.inputs.items():
            flag = {"image": "--image", "mesh": "--mesh", "text": "--text"}.get(kind)
            if flag:
                cmd.extend([flag, value])
        if entry.output:
            cmd.extend(["--output", entry.output])
        for key, value in entry.params.items():
            cmd.extend(["--param", f"{key}={value}"])
        cmd.append("--yes")
        console.print(" ".join(shlex.quote(c) for c in cmd))
        return

    entries = hist.read_all(paths)
    if not all_:
        entries = entries[-limit:]

    if json_:
        for entry in entries:
            # Print directly (not via Rich) so the output is grep/jq-friendly.
            print(_json.dumps(asdict(entry), default=str))
        return

    if not entries:
        info(
            "No runs recorded yet. History accumulates as you use "
            "`gen3dhub run`."
        )
        return

    table = Table(title="Recent runs", header_style="bold cyan")
    table.add_column("ID", style="muted")
    table.add_column("When (UTC)")
    table.add_column("Model", style="model")
    table.add_column("Output")
    table.add_column("Status")
    table.add_column("Time")
    for entry in entries:
        status = "[green]✓[/green]" if entry.exit_code == 0 else "[red]✗[/red]"
        out_name = Path(entry.output).name if entry.output else "-"
        table.add_row(
            entry.id,
            entry.timestamp,
            entry.model,
            out_name,
            status,
            f"{entry.duration_s:.1f}s",
        )
    console.print(table)
    console.print(
        "[muted]Tip: `gen3dhub history --rerun <id>` prints the command to "
        "re-run a past entry.[/muted]"
    )


@app.command("agent")
def agent_command() -> None:
    """Print a comprehensive usage guide aimed at AI agents and scripts.

    Plain text only — meant to be piped to an agent's context window or read
    by a human looking for the "how do I drive this non-interactively" answer.
    """
    from gen3dhub.agent_guide import AGENT_GUIDE

    # Print directly (not via Rich console) so there's no ANSI styling and
    # the output is identical when piped to a file or another process.
    print(AGENT_GUIDE)


if __name__ == "__main__":
    app()
