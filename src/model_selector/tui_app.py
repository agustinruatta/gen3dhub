"""Persistent Textual TUI for Model Selector.

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
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
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

from model_selector.config import Paths
from model_selector.console import error
from model_selector.models.base import InputKind, RunRequest
from model_selector.registry import get_adapter, known_model_ids, list_models

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
        height: 9;
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
        yield Static("[b]Model Selector[/b]", id="menu-title")
        yield ListView(
            ListItem(Label("List supported models"), id="opt-list"),
            ListItem(Label("Set up a model"), id="opt-setup"),
            ListItem(Label("Run inference"), id="opt-run"),
            ListItem(Label("Run diagnostics (doctor)"), id="opt-doctor"),
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
    """Modal with a DirectoryTree. Returns the selected file path, or None on cancel."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "cancel", "Cancel"),
        Binding("q", "cancel", "Cancel"),
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
    FilePickerScreen #picker-tree {
        height: 1fr;
        padding: 0 1;
    }
    FilePickerScreen #picker-footer {
        height: 3;
        padding: 1 1 0 1;
        align: right middle;
    }
    """

    def __init__(self, start_path: Path | None = None) -> None:
        super().__init__()
        self.start_path = (start_path or Path.cwd()).expanduser().resolve()

    def compose(self) -> ComposeResult:
        yield Container(
            Static(
                "[b]Pick a file[/b]\n"
                "[dim]↑/↓ navigate · Enter open folder / select file · Esc cancel[/dim]",
                id="picker-header",
            ),
            DirectoryTree(str(self.start_path), id="picker-tree"),
            Horizontal(
                Button("Cancel", id="picker-cancel"),
                id="picker-footer",
            ),
            id="picker-box",
        )

    def on_mount(self) -> None:
        self.query_one(DirectoryTree).focus()

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        self.dismiss(Path(event.path))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "picker-cancel":
            self.action_cancel()

    def action_cancel(self) -> None:
        self.dismiss(None)


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

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        models = list_models()
        if event.cursor_row is None or event.cursor_row >= len(models):
            return
        self._show_detail(models[event.cursor_row])

    def _show_detail(self, model) -> None:
        detail = self.query_one("#models-detail", Static)
        installed = (
            "[green]installed[/green]"
            if get_adapter(model.id, Paths.default()).is_installed
            else "[yellow]not installed[/yellow]"
        )
        detail.update(
            f"[b]{model.display_name}[/b]  ({installed})\n"
            f"{model.summary}\n"
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
            Label("Image path:"),
            Horizontal(
                Input(placeholder="/path/to/input.png", id="run-image"),
                Button("Browse…", id="run-browse-image"),
                classes="path-row",
            ),
            Label("Text prompt (only used by text-input models):"),
            Input(placeholder="(leave blank for image-input models)", id="run-text"),
            Label("Output path (optional — defaults to <stem>.<ext> in CWD):"),
            Horizontal(
                Input(placeholder="/path/to/output.glb", id="run-output"),
                Button("Browse…", id="run-browse-output"),
                classes="path-row",
            ),
            Horizontal(
                Button("Run", variant="primary", id="run-go"),
                id="run-buttons",
            ),
            id="run-body",
        )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run-browse-image":
            self._open_picker_for("#run-image")
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
        output = self.query_one("#run-output", Input).value.strip()

        adapter = get_adapter(model_id, Paths.default())
        inputs: dict[str, str | Path] = {}
        for spec in adapter.info.inputs:
            if spec.kind is InputKind.IMAGE and image:
                inputs[spec.name] = image
            elif spec.kind is InputKind.TEXT and text:
                inputs[spec.name] = text

        missing = [s.name for s in adapter.info.inputs if s.required and s.name not in inputs]
        if missing:
            self.notify(f"Missing required inputs: {', '.join(missing)}", severity="error")
            return

        request = RunRequest(
            inputs=inputs,
            output_path=Path(output).expanduser() if output else None,
        )
        self._run_inference(adapter, request)

    def _run_inference(self, adapter, request: RunRequest) -> None:
        with self.app.suspend():
            try:
                if not adapter.is_installed:
                    print(f"'{adapter.model_id}' is not installed. Running setup first…\n")
                    adapter.setup()
                adapter.run(request)
            except Exception as exc:
                error(str(exc))
            input("\nPress Enter to return to the menu… ")

    def _open_picker_for(self, target_input_selector: str) -> None:
        """Push the FilePicker modal and pipe the selected path into the given Input."""
        current = self.query_one(target_input_selector, Input).value.strip()
        start = Path(current).expanduser().parent if current else Path.cwd()
        if not start.exists():
            start = Path.cwd()

        def _on_pick(path: Path | None) -> None:
            if path is not None:
                self.query_one(target_input_selector, Input).value = str(path)

        self.app.push_screen(FilePickerScreen(start_path=start), _on_pick)


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
# App
# ---------------------------------------------------------------------------


class ModelSelectorApp(App[None]):
    """Persistent Textual TUI. Stays open until the user explicitly quits."""

    TITLE = "Model Selector"
    SUB_TITLE = "Run AI models from one place"

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
    ]

    def on_mount(self) -> None:
        self.push_screen(MenuScreen())


def run() -> None:
    """Launch the TUI. Returns when the user quits."""
    ModelSelectorApp().run()
