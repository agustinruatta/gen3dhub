"""Persistent Textual TUI for gen3dhub.

Layout:
    MenuScreen (top-level)
      ├─ ModelsScreen   — read-only list of supported models
      ├─ SetupScreen    — pick a model, install it
      ├─ RunScreen      — pick a model, fill inputs, run inference
      └─ DoctorScreen   — verify environment for one model or all

Navigation:
    ↑ / ↓        — move between options/widgets
    Tab / Shift+Tab — cycle focus inside a screen
    Enter        — activate / submit
    Escape       — go back to the previous screen (closes app from the menu)
    Q / Ctrl+C   — quit immediately

Long-running operations (setup, run, doctor) suspend the Textual UI so the
underlying subprocess output (Rich progress bars, pip logs, etc.) renders
directly in the terminal. When the operation finishes, the user presses Enter
to return to the TUI menu — the app *never* exits implicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    DirectoryTree,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Select,
    Static,
)

from gen3dhub import __version__
from gen3dhub.config import Paths
from gen3dhub.console import error
from gen3dhub.models.base import InputKind, ParamKind, RunRequest
from gen3dhub.registry import get_adapter, known_model_ids, list_models

_ALL_TARGETS = "__all__"


# ---------------------------------------------------------------------------
# Top-level menu
# ---------------------------------------------------------------------------


class MenuScreen(Screen):
    """The screen the app boots into. Navigates to every other screen."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("q", "app.quit", "Quit"),
        Binding("ctrl+c", "app.quit", "Quit", show=False),
        Binding("escape", "app.quit", "Quit"),
    ]

    DEFAULT_CSS = """
    MenuScreen {
        align: center middle;
    }
    MenuScreen #menu-title {
        width: 60;
        height: 3;
        content-align: center middle;
        padding: 1 0 0 0;
    }
    MenuScreen #menu {
        width: 60;
        height: 12;
        border: round $primary;
    }
    MenuScreen #menu-hint {
        width: 60;
        content-align: center middle;
        color: $text-muted;
        padding: 1 0 0 0;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static("[b]gen3dhub[/b]", id="menu-title")
        yield ListView(
            ListItem(Label("List supported models"), id="opt-list"),
            ListItem(Label("Set up a model"), id="opt-setup"),
            ListItem(Label("Run inference"), id="opt-run"),
            ListItem(Label("Run diagnostics (doctor)"), id="opt-doctor"),
            ListItem(Label("Uninstall a model (free disk)"), id="opt-uninstall"),
            ListItem(Label("View agent / scripting guide"), id="opt-agent"),
            ListItem(Label("Quit"), id="opt-quit"),
            id="menu",
        )
        yield Static("↑/↓ navigate · Enter select · Q/Esc quit", id="menu-hint")
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item_id = event.item.id or ""
        if item_id == "opt-list":
            self.app.push_screen(ModelsScreen())
        elif item_id == "opt-setup":
            self.app.push_screen(SetupScreen())
        elif item_id == "opt-run":
            self.app.push_screen(RunScreen())
        elif item_id == "opt-doctor":
            self.app.push_screen(DoctorScreen())
        elif item_id == "opt-uninstall":
            self.app.push_screen(UninstallScreen())
        elif item_id == "opt-agent":
            self.app.push_screen(AgentGuideScreen())
        elif item_id == "opt-quit":
            self.app.exit()


# ---------------------------------------------------------------------------
# Common base: every sub-screen has Esc-to-go-back
# ---------------------------------------------------------------------------


class _BackableScreen(Screen):
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("q", "app.pop_screen", "Back"),
        Binding("ctrl+c", "app.quit", "Quit", show=False),
    ]

    # Shared CSS for the top "← Back" toolbar that every sub-screen renders
    # via _back_toolbar(). Each subclass also defines its own DEFAULT_CSS;
    # Textual aggregates DEFAULT_CSS up the inheritance chain so this rule
    # applies everywhere.
    DEFAULT_CSS = """
    .screen-toolbar {
        height: 3;
        padding: 0 1;
        align: left middle;
    }
    .screen-toolbar Button {
        width: auto;
    }
    """

    def _back_toolbar(self) -> Horizontal:
        """Compose-time helper: a top toolbar with a "← Back" button.

        Yield this as the first widget inside each screen's main Container so
        users have a clickable back affordance in addition to Esc/Q.
        """
        return Horizontal(
            Button("← Back", id="back-btn"),
            classes="screen-toolbar",
        )

    async def on_key(self, event):
        """Make Down/Up navigate the form like Tab/Shift+Tab.

        Inputs and Buttons don't consume Down/Up, so we intercept here and move
        focus. We deliberately *don't* steal the keys when a Select is focused
        — the Select needs Down to open its overlay (closed) or to navigate
        options (open). Use Tab or Enter to leave a Select.
        """
        if event.key not in ("down", "up"):
            return
        focused = self.focused
        if focused is None or isinstance(focused, Select):
            return
        if event.key == "down":
            self.focus_next()
        else:
            self.focus_previous()
        event.stop()


# ---------------------------------------------------------------------------
# File picker (modal)
# ---------------------------------------------------------------------------


class FilePickerScreen(ModalScreen[Path | None]):
    """Modal file/folder picker. Returns the selected absolute Path, or None on cancel.

    Three independent ways to commit a selection:
      1. Press Enter on a *file* in the tree → dismisses with that file.
      2. Click "Accept" (or hit Enter on the path Input) → dismisses with whatever
         path is currently in the path Input. Useful for committing a path that
         doesn't exist yet (e.g. output filenames the user is about to create)
         or for picking a directory when allow_directory=True.
      3. Esc / "Cancel" → dismisses with None.

    Tree navigation:
      - Highlighting a node in the tree mirrors that node's path into the path
        Input (only when the tree itself has focus, so manual edits to the
        Input aren't clobbered by an unrelated highlight).
      - Enter on a *folder* expands it (DirectoryTree default — does NOT dismiss).
      - "Up" button or Alt+↑ re-roots the tree at the parent directory.
      - Typing a directory path + Enter on the path Input re-roots the tree there.

    The `allow_directory` flag controls whether Accept will dismiss with a
    folder path. Image inputs set it False; output paths set it True so the
    caller can append a filename.
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "cancel", "Cancel"),
        Binding("alt+up", "go_up", "Up dir"),
    ]

    DEFAULT_CSS = """
    FilePickerScreen {
        align: center middle;
    }
    FilePickerScreen #picker-box {
        width: 90;
        height: 30;
        border: thick $primary;
        background: $surface;
    }
    FilePickerScreen #picker-header {
        height: 3;
        padding: 1 1 0 1;
    }
    FilePickerScreen #picker-path-row {
        height: 3;
        padding: 0 1;
    }
    FilePickerScreen #picker-path-row Input {
        width: 1fr;
    }
    FilePickerScreen #picker-path-row Button {
        width: auto;
        margin-left: 1;
    }
    FilePickerScreen #picker-tree {
        height: 1fr;
        padding: 0 1;
    }
    FilePickerScreen #picker-footer {
        height: 3;
        padding: 1 1 0 1;
        align: right middle;
    }
    FilePickerScreen #picker-footer Button {
        margin-left: 1;
    }
    """

    def __init__(
        self,
        start_path: Path | None = None,
        *,
        allow_directory: bool = False,
        title: str = "Pick a file",
    ) -> None:
        super().__init__()
        self.start_path = (start_path or Path.cwd()).expanduser().resolve()
        self.allow_directory = allow_directory
        self.title_text = title

    def compose(self) -> ComposeResult:
        hint = (
            "↑/↓ navigate tree · Enter expands a folder / picks a file · "
            "Type or edit the path above · Accept commits · Up / Alt+↑ goes to parent · "
            "Esc cancels"
        )
        yield Container(
            Static(f"[b]{self.title_text}[/b]\n[dim]{hint}[/dim]", id="picker-header"),
            Horizontal(
                Input(value=str(self.start_path), id="picker-path"),
                Button("Up", id="picker-up"),
                id="picker-path-row",
            ),
            DirectoryTree(str(self.start_path), id="picker-tree"),
            Horizontal(
                Button("Accept", variant="primary", id="picker-accept"),
                Button("Cancel", id="picker-cancel"),
                id="picker-footer",
            ),
            id="picker-box",
        )

    def on_mount(self) -> None:
        self.query_one(DirectoryTree).focus()

    # ---- tree highlight → mirror into path Input ----

    def on_tree_node_highlighted(self, event) -> None:
        # Only sync when the user is navigating the tree, not when they're
        # mid-edit in the path Input.
        if not isinstance(self.focused, DirectoryTree):
            return
        node_data = getattr(event.node, "data", None)
        node_path = getattr(node_data, "path", None)
        if node_path is None:
            return
        self.query_one("#picker-path", Input).value = str(Path(node_path).resolve())

    # ---- tree Enter on a file → dismiss with that file ----

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        self.dismiss(Path(event.path))

    # ---- path input Enter → re-root or commit ----

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "picker-path":
            return
        candidate = Path(event.value).expanduser()
        if candidate.is_dir():
            # Treat as navigation: re-root the tree at this directory.
            self._reroot(candidate.resolve())
        elif candidate.parent.is_dir():
            # File path under an existing directory — commit even if file doesn't
            # exist yet (useful for output paths the run is about to create).
            self.dismiss(candidate.resolve())
        else:
            self.notify(f"Path not found: {candidate}", severity="error")

    # ---- buttons ----

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "picker-cancel":
            self.action_cancel()
        elif event.button.id == "picker-up":
            self.action_go_up()
        elif event.button.id == "picker-accept":
            self.action_accept()

    # ---- actions ----

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_go_up(self) -> None:
        current = self._current_root()
        parent = current.parent
        if parent != current:
            self._reroot(parent)

    def action_accept(self) -> None:
        candidate = Path(self.query_one("#picker-path", Input).value).expanduser()
        if candidate.is_dir():
            if not self.allow_directory:
                self.notify(
                    "This picker requires a file. The current selection is a folder — "
                    "highlight a file in the tree, or type a filename in the path field.",
                    severity="error",
                )
                return
            self.dismiss(candidate.resolve())
            return
        # File (existing) or non-existent path under an existing directory.
        if candidate.exists() or candidate.parent.is_dir():
            self.dismiss(candidate.resolve() if candidate.exists() else candidate)
            return
        self.notify(
            f"Parent directory does not exist: {candidate.parent}",
            severity="error",
        )

    # ---- helpers ----

    def _current_root(self) -> Path:
        return Path(self.query_one(DirectoryTree).path).resolve()

    def _reroot(self, new_root: Path) -> None:
        tree = self.query_one(DirectoryTree)
        tree.path = str(new_root)
        path_input = self.query_one("#picker-path", Input)
        path_input.value = str(new_root)
        tree.reload()
        tree.focus()


# ---------------------------------------------------------------------------
# Models list
# ---------------------------------------------------------------------------


class ModelsScreen(_BackableScreen):
    DEFAULT_CSS = """
    ModelsScreen #models-body {
        padding: 1 2;
        height: 1fr;
    }
    ModelsScreen #models-table {
        height: auto;
        max-height: 60%;
        margin-bottom: 1;
    }
    ModelsScreen #models-detail {
        height: auto;
        border-top: solid $primary;
        padding: 1 0 0 0;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Container(
            self._back_toolbar(),
            Static("[b]Supported models[/b]"),
            DataTable(id="models-table", cursor_type="row", zebra_stripes=True),
            Static("", id="models-detail"),
            id="models-body",
        )
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#models-table", DataTable)
        table.add_columns("ID", "Name", "Inputs", "Output")
        for m in list_models():
            inputs = ", ".join(f"{i.name} ({i.kind.value})" for i in m.inputs)
            table.add_row(m.id, m.display_name, inputs, m.output_extension)
        if list_models():
            self._show_detail(list_models()[0])

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.app.pop_screen()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        models = list_models()
        if event.cursor_row is None or event.cursor_row >= len(models):
            return
        self._show_detail(models[event.cursor_row])

    def _show_detail(self, model) -> None:
        from gen3dhub.utils.system import assess_fit

        detail = self.query_one("#models-detail", Static)
        installed = (
            "[green]installed[/green]"
            if get_adapter(model.id, Paths.default()).is_installed
            else "[yellow]not installed[/yellow]"
        )
        strengths = "\n".join(f"  • {s}" for s in model.strengths)
        weaknesses = "\n".join(f"  • {w}" for w in model.weaknesses)
        fit = assess_fit(model.hardware)
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
        detail.update(
            f"[b]{model.display_name}[/b]  ({installed})\n"
            f"{model.description}\n\n"
            f"[bold cyan]→ Best for[/bold cyan]\n  {model.best_for}\n\n"
            f"[bold green]✓ Strong[/bold green]\n{strengths}\n\n"
            f"[bold yellow]✗ Weak[/bold yellow]\n{weaknesses}\n\n"
            f"{fit_block}\n\n"
            f"[dim]Homepage:[/dim] {model.homepage}"
        )


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


class SetupScreen(_BackableScreen):
    DEFAULT_CSS = """
    SetupScreen #setup-body {
        padding: 1 2;
        height: 1fr;
    }
    SetupScreen Label {
        padding: 1 0 0 0;
    }
    SetupScreen #setup-buttons {
        height: 3;
        padding: 1 0 0 0;
    }
    SetupScreen Button {
        margin-right: 2;
    }
    """

    def compose(self) -> ComposeResult:
        first_id = list_models()[0].id if list_models() else Select.BLANK
        yield Header(show_clock=False)
        yield Container(
            self._back_toolbar(),
            Static("[b]Set up a model[/b]"),
            Static(
                "[dim]Clones the source repo, creates a per-model virtualenv, "
                "installs pinned dependencies.[/dim]"
            ),
            Label("Model:"),
            Select(
                [(f"{m.display_name}  —  {m.id}", m.id) for m in list_models()],
                id="setup-model",
                allow_blank=False,
                value=first_id,
            ),
            Horizontal(
                Button("Install", variant="primary", id="setup-install"),
                Button("Reinstall (force)", variant="warning", id="setup-force"),
                id="setup-buttons",
            ),
            id="setup-body",
        )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.app.pop_screen()
            return
        force = event.button.id == "setup-force"
        model_id = self.query_one("#setup-model", Select).value
        if not isinstance(model_id, str):
            self.app.bell()
            return
        self._run_setup(model_id, force=force)

    def _run_setup(self, model_id: str, *, force: bool) -> None:
        paths = Paths.default()
        paths.ensure()
        adapter = get_adapter(model_id, paths)
        with self.app.suspend():
            try:
                adapter.setup(force=force)
                # Always run post_setup — terminal is restored during suspend(),
                # so getpass and other interactive prompts work normally.
                adapter.post_setup(interactive=True)
            except Exception as exc:
                error(str(exc))
            input("\nPress Enter to return to the menu… ")


# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------


class RunScreen(_BackableScreen):
    DEFAULT_CSS = """
    RunScreen #run-body {
        padding: 1 2;
        height: 1fr;
    }
    RunScreen Label {
        padding: 1 0 0 0;
    }
    RunScreen Input {
        margin: 0;
    }
    RunScreen .path-row {
        height: 3;
    }
    RunScreen .path-row Input {
        width: 1fr;
    }
    RunScreen .path-row Button {
        width: auto;
        margin-left: 1;
    }
    RunScreen #run-buttons {
        height: 3;
        padding: 1 0 0 0;
    }
    """

    def compose(self) -> ComposeResult:
        first_id = list_models()[0].id if list_models() else Select.BLANK
        yield Header(show_clock=False)
        yield Container(
            self._back_toolbar(),
            Static("[b]Run inference[/b]"),
            Static(
                "[dim]Fill the inputs that apply to the chosen model. "
                "Empty fields are ignored.[/dim]"
            ),
            Label("Model:"),
            Select(
                [(f"{m.display_name}  —  {m.id}", m.id) for m in list_models()],
                id="run-model",
                allow_blank=False,
                value=first_id,
            ),
            Label("Image path:", id="run-image-label"),
            Horizontal(
                Input(placeholder="/path/to/input.png", id="run-image"),
                Button("Browse…", id="run-browse-image"),
                classes="path-row",
                id="run-image-row",
            ),
            Label("Mesh path:", id="run-mesh-label"),
            Horizontal(
                Input(placeholder="/path/to/mesh.glb", id="run-mesh"),
                Button("Browse…", id="run-browse-mesh"),
                classes="path-row",
                id="run-mesh-row",
            ),
            Label("Text prompt:", id="run-text-label"),
            Input(placeholder="Describe what to generate…", id="run-text"),
            Label("Output path (optional — defaults to <stem>.<ext> in CWD):"),
            Horizontal(
                Input(placeholder="/path/to/output.glb", id="run-output"),
                Button("Browse…", id="run-browse-output"),
                classes="path-row",
            ),
            Static(
                "[b]Parameters[/b] [muted](model-specific)[/muted]",
                id="run-params-label",
            ),
            Container(id="run-params"),
            Checkbox(
                "Install model automatically if not installed yet",
                value=True,
                id="run-auto-setup",
            ),
            Checkbox(
                "Force CPU inference (10-60x slower; use if GPU runs out of VRAM)",
                value=False,
                id="run-force-cpu",
            ),
            Horizontal(
                Button("Run", variant="primary", id="run-go"),
                id="run-buttons",
            ),
            id="run-body",
        )
        yield Footer()

    async def on_mount(self) -> None:
        # Hide the input rows that don't apply to the model chosen by default.
        initial = self.query_one("#run-model", Select).value
        if isinstance(initial, str):
            self._refresh_input_visibility(initial)
            await self._refresh_params(initial)

    async def on_select_changed(self, event: Select.Changed) -> None:
        # Only react to the model picker — Selects on other screens reuse the
        # same event type but we don't care about them here.
        if event.select.id != "run-model":
            return
        if isinstance(event.value, str):
            self._refresh_input_visibility(event.value)
            await self._refresh_params(event.value)

    async def _refresh_params(self, model_id: str) -> None:
        """Mount one widget per declared ParamSpec for the chosen model.

        Widgets get IDs prefixed with `param-<name>` so submit can collect
        them generically. Container is cleared on every model change so we
        don't accumulate stale widgets.
        """
        try:
            adapter = get_adapter(model_id, Paths.default())
        except KeyError:
            return
        container = self.query_one("#run-params", Container)
        await container.remove_children()

        params = adapter.info.params
        # Hide the label too when the model has no params at all.
        self.query_one("#run-params-label").display = bool(params)
        container.display = bool(params)
        if not params:
            return

        for spec in params:
            await container.mount(Label(f"{spec.label}: [dim]{spec.description}[/dim]"))
            await container.mount(self._make_param_widget(spec))

    def _make_param_widget(self, spec):
        """Build the right widget kind for a ParamSpec; default value pre-filled."""
        widget_id = f"param-{spec.name}"
        if spec.kind is ParamKind.SELECT:
            return Select(
                [(c, c) for c in (spec.choices or ())],
                id=widget_id,
                allow_blank=False,
                value=str(spec.default),
            )
        if spec.kind is ParamKind.BOOL:
            return Checkbox("", value=bool(spec.default), id=widget_id)
        # int / float / text all render as a plain Input; coercion happens at submit.
        return Input(value=str(spec.default), id=widget_id, classes="param-input")

    def _refresh_input_visibility(self, model_id: str) -> None:
        """Show/hide image/mesh/text rows based on the selected model's InputSpec.

        Stale values are intentionally left in the hidden Inputs — they're
        already filtered out at submit time because we only read inputs whose
        kind appears in the adapter's `info.inputs`. So switching back to a
        model that uses a previously-hidden input restores the value.
        """
        try:
            adapter = get_adapter(model_id, Paths.default())
        except KeyError:
            return
        needed = {spec.kind for spec in adapter.info.inputs}
        sections = [
            (InputKind.IMAGE, "#run-image-label", "#run-image-row"),
            (InputKind.MESH, "#run-mesh-label", "#run-mesh-row"),
            (InputKind.TEXT, "#run-text-label", "#run-text"),
        ]
        for kind, label_sel, widget_sel in sections:
            visible = kind in needed
            self.query_one(label_sel).display = visible
            self.query_one(widget_sel).display = visible

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.app.pop_screen()
            return
        if event.button.id == "run-browse-image":
            self._open_picker_for("#run-image")
            return
        if event.button.id == "run-browse-mesh":
            self._open_picker_for("#run-mesh")
            return
        if event.button.id == "run-browse-output":
            self._open_picker_for("#run-output")
            return
        if event.button.id != "run-go":
            return
        model_id = self.query_one("#run-model", Select).value
        if not isinstance(model_id, str):
            self.app.bell()
            return

        image = self.query_one("#run-image", Input).value.strip()
        text = self.query_one("#run-text", Input).value.strip()
        mesh = self.query_one("#run-mesh", Input).value.strip()
        output = self.query_one("#run-output", Input).value.strip()

        adapter = get_adapter(model_id, Paths.default())
        inputs: dict[str, str | Path] = {}
        for spec in adapter.info.inputs:
            if spec.kind is InputKind.IMAGE and image:
                inputs[spec.name] = image
            elif spec.kind is InputKind.TEXT and text:
                inputs[spec.name] = text
            elif spec.kind is InputKind.MESH and mesh:
                inputs[spec.name] = mesh

        missing = [s.name for s in adapter.info.inputs if s.required and s.name not in inputs]
        if missing:
            self.notify(f"Missing required inputs: {', '.join(missing)}", severity="error")
            return

        auto_setup = self.query_one("#run-auto-setup", Checkbox).value
        force_cpu = self.query_one("#run-force-cpu", Checkbox).value

        try:
            params = self._collect_params(adapter.info)
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return

        request = RunRequest(
            inputs=inputs,
            output_path=Path(output).expanduser() if output else None,
            params=params,
            extra={"force_cpu": "1"} if force_cpu else {},
        )
        self._run_inference(adapter, request, auto_setup=auto_setup)

    def _collect_params(self, model_info) -> dict[str, object]:
        """Read each param widget by its `param-<name>` id and coerce to the spec's type."""
        result: dict[str, object] = {}
        for spec in model_info.params:
            widget_id = f"#param-{spec.name}"
            if spec.kind is ParamKind.BOOL:
                result[spec.name] = self.query_one(widget_id, Checkbox).value
                continue
            if spec.kind is ParamKind.SELECT:
                value = self.query_one(widget_id, Select).value
                result[spec.name] = str(value) if isinstance(value, str) else spec.default
                continue
            raw = self.query_one(widget_id, Input).value.strip()
            if not raw:
                result[spec.name] = spec.default
                continue
            if spec.kind is ParamKind.INT:
                try:
                    result[spec.name] = int(raw)
                except ValueError as exc:
                    raise ValueError(
                        f"Parameter '{spec.label}' must be an integer (got {raw!r})."
                    ) from exc
            elif spec.kind is ParamKind.FLOAT:
                try:
                    result[spec.name] = float(raw)
                except ValueError as exc:
                    raise ValueError(
                        f"Parameter '{spec.label}' must be a number (got {raw!r})."
                    ) from exc
            else:  # TEXT
                result[spec.name] = raw
        return result

    def _run_inference(self, adapter, request: RunRequest, *, auto_setup: bool) -> None:
        with self.app.suspend():
            try:
                if not adapter.is_installed:
                    if auto_setup:
                        print(f"'{adapter.model_id}' is not installed. Running setup first…\n")
                        adapter.setup()
                    else:
                        raise RuntimeError(
                            f"Model '{adapter.model_id}' is not installed and auto-setup is "
                            f"disabled. Either tick the checkbox, run Setup from the menu, or "
                            f"call `gen3dhub setup --model {adapter.model_id}` from the shell."
                        )
                # post_setup is idempotent and silent when already configured.
                # Running it on every run means a user who skipped credentials
                # the first time gets re-prompted here instead of seeing the
                # opaque "No HF token detected" pre-run check failure.
                adapter.post_setup(interactive=True)
                adapter.run(request)
            except Exception as exc:
                error(str(exc))
            input("\nPress Enter to return to the menu… ")

    def _open_picker_for(self, target_input_selector: str) -> None:
        """Push the FilePicker modal and pipe the selected path into the given Input.

        For the output path picker, accepting a folder is allowed: we then
        compose a default filename (`<image-stem>.<ext>`) inside that folder so
        the user doesn't have to type it manually.
        """
        current = self.query_one(target_input_selector, Input).value.strip()
        # Default starting point: the user's home directory. Most user files
        # live under $HOME, and the project dir / CWD is a poor default
        # because it depends on where the launcher was invoked from.
        start = Path(current).expanduser().parent if current else Path.home()
        if not start.exists():
            start = Path.home()

        is_output = target_input_selector == "#run-output"
        title = "Pick the output path or a folder" if is_output else "Pick the input image"

        def _on_pick(path: Path | None) -> None:
            if path is None:
                return
            target = self.query_one(target_input_selector, Input)
            # Output picker: if the user accepted a folder, append a sensible
            # default filename so the value is a complete writable path.
            if is_output and path.is_dir():
                target.value = str(path / self._default_output_filename())
            else:
                target.value = str(path)

        self.app.push_screen(
            FilePickerScreen(start_path=start, allow_directory=is_output, title=title),
            _on_pick,
        )

    def _default_output_filename(self) -> str:
        """Compose `<image-stem><output_ext>` based on the currently-selected model."""
        from contextlib import suppress

        model_id = self.query_one("#run-model", Select).value
        ext = ".out"
        if isinstance(model_id, str):
            with suppress(KeyError):
                ext = get_adapter(model_id, Paths.default()).info.output_extension
        image_value = self.query_one("#run-image", Input).value.strip()
        stem = Path(image_value).stem if image_value else "output"
        return f"{stem}{ext}"


# ---------------------------------------------------------------------------
# Doctor
# ---------------------------------------------------------------------------


class DoctorScreen(_BackableScreen):
    DEFAULT_CSS = """
    DoctorScreen #doctor-body {
        padding: 1 2;
        height: 1fr;
    }
    DoctorScreen Label {
        padding: 1 0 0 0;
    }
    DoctorScreen #doctor-buttons {
        height: 3;
        padding: 1 0 0 0;
    }
    """

    def compose(self) -> ComposeResult:
        choices = [("All models", _ALL_TARGETS)] + [
            (f"{m.display_name}  —  {m.id}", m.id) for m in list_models()
        ]
        yield Header(show_clock=False)
        yield Container(
            self._back_toolbar(),
            Static("[b]Run diagnostics[/b]"),
            Static(
                "[dim]Verifies installation, virtualenv, and Hugging Face "
                "authentication.[/dim]"
            ),
            Label("Target:"),
            Select(choices, id="doctor-model", allow_blank=False, value=_ALL_TARGETS),
            Horizontal(
                Button("Run doctor", variant="primary", id="doctor-go"),
                id="doctor-buttons",
            ),
            id="doctor-body",
        )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.app.pop_screen()
            return
        if event.button.id != "doctor-go":
            return
        target = self.query_one("#doctor-model", Select).value
        targets = known_model_ids() if target == _ALL_TARGETS else [str(target)]
        self._run_doctor(targets)

    def _run_doctor(self, targets: list[str]) -> None:
        paths = Paths.default()
        paths.ensure()
        with self.app.suspend():
            for model_id in targets:
                adapter = get_adapter(model_id, paths)
                print(f"\n=== {adapter.info.display_name} ({model_id}) ===")
                problems = adapter.verify()
                if not problems:
                    print("  ✓ All checks passed.")
                else:
                    for p in problems:
                        print(f"  ✗ {p}")
            input("\nPress Enter to return to the menu… ")


# ---------------------------------------------------------------------------
# Uninstall (reclaim disk)
# ---------------------------------------------------------------------------


class UninstallScreen(_BackableScreen):
    """Pick an installed model and remove its repo + venv to reclaim disk space.

    Live disk-size readout next to each model so the user knows what they're
    freeing before they confirm.
    """

    DEFAULT_CSS = """
    UninstallScreen #uninstall-body {
        padding: 1 2;
        height: 1fr;
    }
    UninstallScreen Label {
        padding: 1 0 0 0;
    }
    UninstallScreen #uninstall-buttons {
        height: 3;
        padding: 1 0 0 0;
    }
    """

    def compose(self) -> ComposeResult:
        from gen3dhub.utils.system import directory_size_bytes, format_bytes

        paths = Paths.default()
        # Build choices showing only installed models with their disk footprint.
        choices = []
        for m in list_models():
            model_dir = paths.model_dir(m.id)
            if not model_dir.exists():
                continue
            size = format_bytes(directory_size_bytes(model_dir))
            choices.append((f"{m.display_name}  ({size})  —  {m.id}", m.id))

        yield Header(show_clock=False)
        yield Container(
            self._back_toolbar(),
            Static(
                "[b]Uninstall a model[/b]\n"
                "[dim]Removes the per-model repo + venv. Hugging Face weights "
                "in ~/.cache/huggingface/ are kept (shared across HF tools).[/dim]"
            ),
            Label("Model to uninstall:"),
            Select(
                choices,
                id="uninstall-model",
                allow_blank=False,
                prompt="(no models installed)" if not choices else "Pick one",
                value=choices[0][1] if choices else Select.BLANK,
            ),
            Horizontal(
                Button(
                    "Uninstall",
                    variant="error",
                    id="uninstall-go",
                    disabled=not choices,
                ),
                id="uninstall-buttons",
            ),
            id="uninstall-body",
        )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.app.pop_screen()
            return
        if event.button.id != "uninstall-go":
            return
        model_id = self.query_one("#uninstall-model", Select).value
        if not isinstance(model_id, str):
            self.app.bell()
            return
        self._run_uninstall(model_id)

    def _run_uninstall(self, model_id: str) -> None:
        import shutil

        from gen3dhub.utils.system import directory_size_bytes, format_bytes

        paths = Paths.default()
        model_dir = paths.model_dir(model_id)
        with self.app.suspend():
            if not model_dir.exists():
                error(f"'{model_id}' is not installed (no dir at {model_dir})")
            else:
                size = format_bytes(directory_size_bytes(model_dir))
                print(f"Removing {model_dir} ({size})…")
                shutil.rmtree(model_dir)
                print(f"✓ Uninstalled '{model_id}' (freed ~{size})")
                print()
                print(
                    "Note: model weights downloaded by Hugging Face are kept in "
                    "~/.cache/huggingface/ (shared across HF tools)."
                )
            input("\nPress Enter to return to the menu… ")


# ---------------------------------------------------------------------------
# Agent / scripting guide
# ---------------------------------------------------------------------------


class AgentGuideScreen(_BackableScreen):
    """Read-only scrollable view of the same text printed by `gen3dhub agent`.

    Inherits from _BackableScreen so it picks up the "← Back" toolbar helper
    and shared CSS, but overrides `on_key` to a no-op so Down/Up keep
    scrolling the guide content (the parent's on_key would otherwise hijack
    them for focus traversal, which doesn't apply here).
    """

    DEFAULT_CSS = """
    AgentGuideScreen #agent-body {
        padding: 1 2;
        height: 1fr;
    }
    AgentGuideScreen #agent-text {
        padding: 1 0;
    }
    """

    async def on_key(self, event):
        # Override _BackableScreen.on_key with a no-op: VerticalScroll handles
        # Down/Up natively for scrolling, which is what we want here.
        return

    def compose(self) -> ComposeResult:
        from gen3dhub.agent_guide import AGENT_GUIDE

        yield Header(show_clock=False)
        yield self._back_toolbar()
        yield VerticalScroll(
            Static(
                "[b]Agent / scripting guide[/b]  "
                "[dim](same text as `gen3dhub agent`)[/dim]"
            ),
            Static(AGENT_GUIDE, id="agent-text", markup=False),
            id="agent-body",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#agent-body", VerticalScroll).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.app.pop_screen()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class ModelSelectorApp(App[None]):
    """Persistent Textual TUI. Stays open until the user explicitly quits."""

    TITLE = "gen3dhub"
    SUB_TITLE = f"Hub for 3D-gen AI models · v{__version__}"

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
    ]

    def on_mount(self) -> None:
        self.push_screen(MenuScreen())


def run() -> None:
    """Launch the TUI. Returns when the user quits."""
    ModelSelectorApp().run()
