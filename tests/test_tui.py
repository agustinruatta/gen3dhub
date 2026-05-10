"""TUI smoke tests via Textual's `pilot`.

These run the app headlessly (in-memory virtual terminal) and exercise the
contract: every screen mounts, Esc returns to the menu, dynamic visibility
on the Run screen reacts to model selection. They're fast (~1-2s total) and
independent of GPU / network.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Checkbox, Select

from gen3dhub.tui_app import (
    AgentGuideScreen,
    DoctorScreen,
    HistoryScreen,
    MenuScreen,
    ModelSelectorApp,
    ModelsScreen,
    PathManagementScreen,
    RunScreen,
    SetupScreen,
    UninstallScreen,
)


async def test_app_boots_to_menu():
    app = ModelSelectorApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, MenuScreen)


@pytest.mark.parametrize(
    "screen_cls",
    [
        ModelsScreen,
        SetupScreen,
        RunScreen,
        DoctorScreen,
        HistoryScreen,
        UninstallScreen,
        PathManagementScreen,
        AgentGuideScreen,
    ],
)
async def test_each_subscreen_mounts_and_back_returns(screen_cls, temp_cache_dir):
    """Push each sub-screen, verify it mounts, then Esc back to MenuScreen.

    Uses temp_cache_dir so screens that read disk (Uninstall reads model dir
    sizes; History reads the JSONL log) don't depend on the user's real cache.
    """
    app = ModelSelectorApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.push_screen(screen_cls())
        await pilot.pause(0.1)
        assert isinstance(app.screen, screen_cls)
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, MenuScreen)


async def test_back_button_returns_to_menu(temp_cache_dir):
    """Click the visible "← Back" button (not just Esc)."""
    app = ModelSelectorApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.push_screen(SetupScreen())
        await pilot.pause(0.1)
        await pilot.click("#back-btn")
        await pilot.pause()
        assert isinstance(app.screen, MenuScreen)


async def test_run_screen_input_visibility_changes_with_model():
    """Switching the model should hide/show input rows per the model's
    declared InputSpec."""
    app = ModelSelectorApp()
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        await app.push_screen(RunScreen())
        await pilot.pause(0.2)
        select = app.screen.query_one("#run-model", Select)

        # paint3d declares mesh + image inputs.
        select.value = "paint3d"
        await pilot.pause(0.2)
        assert app.screen.query_one("#run-mesh-label").display
        assert app.screen.query_one("#run-image-label").display

        # stable-fast-3d declares only image; mesh row should hide.
        select.value = "stable-fast-3d"
        await pilot.pause(0.2)
        assert not app.screen.query_one("#run-mesh-label").display
        assert app.screen.query_one("#run-image-label").display


async def test_run_screen_params_remount_per_model():
    """Each model declares its own ParamSpecs; switching models should
    re-mount the matching widgets."""
    app = ModelSelectorApp()
    async with app.run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        await app.push_screen(RunScreen())
        await pilot.pause(0.2)
        select = app.screen.query_one("#run-model", Select)

        select.value = "stable-fast-3d"
        await pilot.pause(0.2)
        sf3d_ids = {
            w.id
            for w in app.screen.query_one("#run-params").query("Input,Select,Checkbox")
        }
        assert "param-texture_resolution" in sf3d_ids
        assert "param-remesh_option" in sf3d_ids

        select.value = "paint3d"
        await pilot.pause(0.2)
        paint_ids = {
            w.id
            for w in app.screen.query_one("#run-params").query("Input,Select,Checkbox")
        }
        assert "param-prompt" in paint_ids
        # SF3D's params should be gone — different ParamSpecs declared.
        assert "param-texture_resolution" not in paint_ids


async def test_menu_has_all_top_level_options():
    """Sanity check that the main menu lists every screen entry we expect.
    Catches accidental removals in compose()."""
    app = ModelSelectorApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        menu = app.screen.query_one("#menu")
        item_ids = {item.id for item in menu.query("ListItem")}
        for expected in (
            "opt-list",
            "opt-setup",
            "opt-run",
            "opt-history",
            "opt-doctor",
            "opt-uninstall",
            "opt-path",
            "opt-agent",
            "opt-quit",
        ):
            assert expected in item_ids, f"Missing menu entry: {expected}"


async def test_run_screen_default_checkboxes():
    """Auto-setup default ON, force-cpu default OFF — agents and the docs
    rely on these defaults."""
    app = ModelSelectorApp()
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        await app.push_screen(RunScreen())
        await pilot.pause(0.2)
        assert app.screen.query_one("#run-auto-setup", Checkbox).value is True
        assert app.screen.query_one("#run-force-cpu", Checkbox).value is False


async def test_setup_screen_has_install_and_force_buttons():
    app = ModelSelectorApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.push_screen(SetupScreen())
        await pilot.pause(0.1)
        # Both buttons present and distinct.
        install = app.screen.query_one("#setup-install", Button)
        force = app.screen.query_one("#setup-force", Button)
        assert install.label.plain == "Install"
        assert "Reinstall" in force.label.plain


async def test_run_screen_browse_buttons_open_picker():
    """Clicking Browse should push the FilePickerScreen modal."""
    from gen3dhub.tui_app import FilePickerScreen

    app = ModelSelectorApp()
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        await app.push_screen(RunScreen())
        await pilot.pause(0.2)
        # Image input should be visible for the default model — click its Browse.
        await pilot.click("#run-browse-image")
        await pilot.pause(0.2)
        assert isinstance(app.screen, FilePickerScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, RunScreen)


async def test_path_management_detects_project_source():
    """When tests run in the dev checkout, _detect_project_source should find
    pyproject.toml + .venv/ and enable the install button. Conversely, when
    detection fails, install is disabled."""
    from gen3dhub.tui_app import _detect_project_source

    src = _detect_project_source()
    # Tests run from a venv-having checkout; detection should succeed.
    assert src is not None

    app = ModelSelectorApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.push_screen(PathManagementScreen())
        await pilot.pause(0.2)
        install_btn = app.screen.query_one("#path-install", Button)
        # Install button enabled because a source dir exists.
        assert install_btn.disabled is False
