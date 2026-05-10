"""Shared pytest fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_cache_dir(monkeypatch):
    """Point GEN3DHUB_CACHE_DIR at a fresh tempdir for the duration of the test.

    Tests that touch the cache (history, paths, etc.) should depend on this
    fixture so they don't pollute the user's real `~/.cache/gen3dhub`.
    """
    with tempfile.TemporaryDirectory(prefix="gen3dhub-test-") as td:
        monkeypatch.setenv("GEN3DHUB_CACHE_DIR", td)
        yield Path(td)


@pytest.fixture
def synthetic_glb(tmp_path):
    """Create a tiny valid GLB (a unit cube) for tests that need a real mesh."""
    import trimesh

    glb = tmp_path / "cube.glb"
    trimesh.creation.box(extents=(1.0, 2.0, 0.5)).export(str(glb))
    return glb


@pytest.fixture(autouse=True)
def reset_events_json_mode():
    """Module-level state in `gen3dhub.events` could leak between tests.

    Reset before AND after each test so a misbehaving test can't poison the
    rest of the run.
    """
    from gen3dhub import events

    events.set_json_mode(False)
    yield
    events.set_json_mode(False)


@pytest.fixture
def mock_vram(monkeypatch):
    """Helper to mock the GPU VRAM detection in `utils.system`.

    Usage:
        def test_x(mock_vram):
            mock_vram(8 * 1024)   # 8 GB
            ...

    Pass `None` to simulate "no GPU detected".
    """
    from gen3dhub.utils import system

    def _mock(vram_mb: int | None) -> None:
        # `assess_fit` looks up `gpu_vram_mb` in the system module's globals at
        # call time, so monkeypatching the module attribute is enough — the
        # original function's @functools.cache becomes irrelevant (we replace
        # the function entirely).
        monkeypatch.setattr(system, "gpu_vram_mb", lambda: vram_mb)

    return _mock
