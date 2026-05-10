#!/usr/bin/env python3
"""Generate SVG screenshots of every TUI screen for the README and docs.

Uses Textual's headless test pilot to mount each screen and `export_screenshot`
to dump a fancy terminal-framed SVG. Re-runnable: each invocation overwrites
the existing files in docs/screenshots/.

Run with:
    uv run scripts/take_screenshots.py
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import tempfile
from pathlib import Path

# Make the script work both when called directly and via `uv run`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

SCREENSHOTS_DIR = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
SIZE = (120, 38)

# Demo paths shown inside the captured screens. Kept neutral so the published
# SVGs don't bake in the maintainer's actual filesystem layout.
DEMO_HOME = "/home/user"


async def _take(app, output: Path, *, title: str | None = None) -> None:
    svg = app.export_screenshot(title=title)
    output.write_text(svg)
    print(f"  wrote {output.relative_to(SCREENSHOTS_DIR.parent.parent)}")


async def capture_menu() -> None:
    from gen3dhub.tui_app import ModelSelectorApp

    app = ModelSelectorApp()
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause(0.3)
        await _take(app, SCREENSHOTS_DIR / "menu.svg", title="gen3dhub — main menu")


async def capture_models() -> None:
    from gen3dhub.tui_app import ModelSelectorApp, ModelsScreen

    app = ModelSelectorApp()
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        await app.push_screen(ModelsScreen())
        await pilot.pause(0.3)
        await _take(app, SCREENSHOTS_DIR / "models.svg", title="Supported models")


async def capture_setup() -> None:
    from gen3dhub.tui_app import ModelSelectorApp, SetupScreen

    app = ModelSelectorApp()
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        await app.push_screen(SetupScreen())
        await pilot.pause(0.3)
        await _take(app, SCREENSHOTS_DIR / "setup.svg", title="Set up a model")


async def capture_run() -> None:
    """Populate inputs + params for paint3d so all input rows appear."""
    from textual.widgets import Input, Select

    from gen3dhub.tui_app import ModelSelectorApp, RunScreen

    app = ModelSelectorApp()
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        await app.push_screen(RunScreen())
        await pilot.pause(0.3)
        # paint3d declares both image + mesh inputs — best for showing the
        # dynamic form.
        select = app.screen.query_one("#run-model", Select)
        select.value = "paint3d"
        await pilot.pause(0.3)
        # Pre-fill some plausible paths so the boxes aren't empty.
        app.screen.query_one("#run-image", Input).value = "~/refs/cat.png"
        app.screen.query_one("#run-mesh", Input).value = "~/assets/cat_shape.glb"
        app.screen.query_one("#run-output", Input).value = "~/assets/cat_textured.glb"
        await pilot.pause(0.2)
        await _take(app, SCREENSHOTS_DIR / "run.svg", title="Run inference")


async def capture_doctor() -> None:
    from gen3dhub.tui_app import DoctorScreen, ModelSelectorApp

    app = ModelSelectorApp()
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        await app.push_screen(DoctorScreen())
        await pilot.pause(0.3)
        await _take(app, SCREENSHOTS_DIR / "doctor.svg", title="Diagnostics")


async def capture_history(temp_cache: Path) -> None:
    """Seed the cache with sample history entries before mounting the screen."""
    from gen3dhub import history as hist
    from gen3dhub.config import Paths
    from gen3dhub.tui_app import HistoryScreen, ModelSelectorApp

    paths = Paths.default()
    paths.ensure()

    # Three plausible past runs covering all three adapters.
    entries = [
        hist.HistoryEntry(
            id="20260510-141502-a3f9c2",
            timestamp="2026-05-10T14:15:02Z",
            model="stable-fast-3d",
            inputs={"image": "~/refs/cat.png"},
            params={"remesh_option": "quad", "texture_resolution": "2048"},
            output="~/assets/cat.glb",
            preview="~/assets/cat.preview.png",
            duration_s=1.4,
            exit_code=0,
        ),
        hist.HistoryEntry(
            id="20260510-145837-b7d108",
            timestamp="2026-05-10T14:58:37Z",
            model="hunyuan3d-2",
            inputs={"image": "~/refs/dragon.png"},
            params={"octree_resolution": "512", "seed": "42"},
            output="~/assets/dragon_shape.glb",
            preview="~/assets/dragon_shape.preview.png",
            duration_s=28.7,
            exit_code=0,
        ),
        hist.HistoryEntry(
            id="20260510-152244-c9e055",
            timestamp="2026-05-10T15:22:44Z",
            model="paint3d",
            inputs={
                "mesh": "~/assets/dragon_shape.glb",
                "image": "~/refs/dragon.png",
            },
            params={"prompt": "fierce red dragon, scales detail"},
            output="~/assets/dragon_textured.glb",
            preview="~/assets/dragon_textured.preview.png",
            duration_s=412.0,
            exit_code=0,
        ),
    ]
    for entry in entries:
        hist.append(paths, entry)

    app = ModelSelectorApp()
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        await app.push_screen(HistoryScreen())
        await pilot.pause(0.3)
        await _take(app, SCREENSHOTS_DIR / "history.svg", title="Run history")


async def capture_uninstall(temp_cache: Path) -> None:
    """Create dummy model dirs so the picker has options instead of the empty state."""
    from gen3dhub.config import Paths
    from gen3dhub.tui_app import ModelSelectorApp, UninstallScreen

    paths = Paths.default()
    paths.ensure()
    # Touch the marker + a small fake file per "installed" model so the picker
    # has rows with a non-zero size readout.
    for model_id in ("stable-fast-3d", "hunyuan3d-2", "paint3d"):
        model_dir = paths.model_dir(model_id)
        model_dir.mkdir(parents=True, exist_ok=True)
        paths.model_installed_marker(model_id).touch()
        # Write a small "fake" file so the size column has content (~100 KiB).
        (model_dir / "fake.bin").write_bytes(b"X" * 100 * 1024)

    app = ModelSelectorApp()
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        await app.push_screen(UninstallScreen())
        await pilot.pause(0.3)
        await _take(app, SCREENSHOTS_DIR / "uninstall.svg", title="Uninstall a model")

    # Clean up so other screens see an empty state.
    import shutil
    for model_id in ("stable-fast-3d", "hunyuan3d-2", "paint3d"):
        shutil.rmtree(paths.model_dir(model_id), ignore_errors=True)


async def capture_path_management() -> None:
    """Render the path-management screen with a neutral, synthetic install layout.

    The screen normally shows the real `shutil.which("gen3dhub")` and the
    detected dev-checkout root — both reveal the maintainer's filesystem.
    For published screenshots we monkey-patch both to display a typical
    `uv tool install` layout under DEMO_HOME instead.
    """
    import shutil as _shutil

    from gen3dhub import tui_app
    from gen3dhub.tui_app import ModelSelectorApp, PathManagementScreen

    orig_which = _shutil.which
    orig_detect = tui_app._detect_project_source

    def fake_which(cmd: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if cmd == "gen3dhub":
            return f"{DEMO_HOME}/.local/bin/gen3dhub"
        return orig_which(cmd, *args, **kwargs)

    _shutil.which = fake_which  # type: ignore[assignment]
    tui_app._detect_project_source = lambda: Path(f"{DEMO_HOME}/Projects/gen3dhub")
    try:
        app = ModelSelectorApp()
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause()
            await app.push_screen(PathManagementScreen())
            await pilot.pause(0.3)
            await _take(app, SCREENSHOTS_DIR / "path-management.svg", title="Manage on PATH")
    finally:
        _shutil.which = orig_which  # type: ignore[assignment]
        tui_app._detect_project_source = orig_detect


async def capture_agent_guide() -> None:
    from gen3dhub.tui_app import AgentGuideScreen, ModelSelectorApp

    app = ModelSelectorApp()
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        await app.push_screen(AgentGuideScreen())
        await pilot.pause(0.3)
        await _take(app, SCREENSHOTS_DIR / "agent-guide.svg", title="Agent guide")


async def capture_file_picker() -> None:
    """Render the file picker rooted at a fake $HOME with neutral sample dirs.

    The picker opens at `Path.home()` when no path is pre-filled, which would
    otherwise leak the maintainer's real home directory listing into the SVG.
    We override $HOME for the duration of this capture so the path bar and the
    listed entries are synthetic.
    """
    from gen3dhub.tui_app import ModelSelectorApp, RunScreen

    real_home = os.environ.get("HOME")
    with tempfile.TemporaryDirectory(prefix="gen3dhub-fake-home-") as fake_home:
        fake_home_path = Path(fake_home)
        # Populate with a handful of plausible-looking entries so the listing
        # doesn't appear empty in the screenshot.
        for sub in ("Documents", "Downloads", "Pictures", "Projects", "assets", "refs"):
            (fake_home_path / sub).mkdir()
        (fake_home_path / "notes.txt").write_text("")

        os.environ["HOME"] = str(fake_home_path)
        try:
            app = ModelSelectorApp()
            async with app.run_test(size=SIZE) as pilot:
                await pilot.pause()
                await app.push_screen(RunScreen())
                await pilot.pause(0.3)
                await pilot.click("#run-browse-image")
                await pilot.pause(0.4)
                svg_path = SCREENSHOTS_DIR / "file-picker.svg"
                await _take(app, svg_path, title="Pick a file")
                # Swap the throwaway tempdir path that ended up in the path bar
                # for the neutral DEMO_HOME used elsewhere in the docs.
                svg_path.write_text(
                    svg_path.read_text().replace(str(fake_home_path), DEMO_HOME)
                )
        finally:
            if real_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = real_home


def _scrub_real_home_paths() -> None:
    """Replace any leftover real-user home prefix in generated SVGs with DEMO_HOME.

    Belt-and-suspenders against accidental personal-path leakage: every capture
    is supposed to run with neutralized paths, but if any new screen surfaces
    `Path.home()` or a CWD that we forgot to mock, this pass catches it before
    the SVGs hit a public repo.
    """
    real_home = str(Path.home())
    pattern = re.compile(re.escape(real_home))
    for svg in SCREENSHOTS_DIR.glob("*.svg"):
        text = svg.read_text()
        new_text, n = pattern.subn(DEMO_HOME, text)
        if n:
            svg.write_text(new_text)
            print(f"  scrubbed {n} occurrence(s) of $HOME in {svg.name}")


async def main() -> None:
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing screenshots to {SCREENSHOTS_DIR}…")

    with tempfile.TemporaryDirectory(prefix="gen3dhub-screenshots-") as td:
        # Isolate every screenshot from the user's real cache so the captures
        # are deterministic and reproducible.
        os.environ["GEN3DHUB_CACHE_DIR"] = td
        temp_cache = Path(td)

        await capture_menu()
        await capture_models()
        await capture_setup()
        await capture_run()
        await capture_doctor()
        await capture_history(temp_cache)
        await capture_uninstall(temp_cache)
        await capture_path_management()
        await capture_agent_guide()
        await capture_file_picker()

    _scrub_real_home_paths()

    print(f"\nDone. {len(list(SCREENSHOTS_DIR.glob('*.svg')))} SVG files written.")


if __name__ == "__main__":
    asyncio.run(main())
