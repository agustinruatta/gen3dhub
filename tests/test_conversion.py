"""Tests for `gen3dhub.utils.conversion` — mesh format conversion."""

from __future__ import annotations

from pathlib import Path

import pytest
import trimesh

from gen3dhub.utils.conversion import (
    SUPPORTED_FORMATS,
    convert_mesh,
    format_for_path,
)


def test_supported_formats_includes_canonical():
    assert {"glb", "obj", "ply", "stl"}.issubset(set(SUPPORTED_FORMATS))


def test_format_for_path_known_extensions():
    assert format_for_path(Path("/tmp/cat.obj")) == "obj"
    assert format_for_path(Path("/tmp/cat.glb")) == "glb"
    assert format_for_path(Path("/tmp/cat.ply")) == "ply"
    assert format_for_path(Path("/tmp/cat.stl")) == "stl"


def test_format_for_path_case_insensitive():
    assert format_for_path(Path("/tmp/cat.OBJ")) == "obj"


def test_format_for_path_unknown_uses_default():
    assert format_for_path(Path("/tmp/cat.fbx")) == "glb"
    assert format_for_path(None) == "glb"


def test_format_for_path_custom_default():
    assert format_for_path(None, default="ply") == "ply"
    assert format_for_path(Path("/tmp/cat.fbx"), default="stl") == "stl"


@pytest.mark.parametrize("dst_ext", ["obj", "ply", "stl"])
def test_convert_glb_to_format_produces_loadable_file(synthetic_glb, tmp_path, dst_ext):
    """Each conversion should produce a non-empty file that trimesh can load
    back. Doesn't assert content fidelity (some formats are lossy by design)
    — just that we didn't write garbage."""
    dst = tmp_path / f"converted.{dst_ext}"
    convert_mesh(synthetic_glb, dst)
    assert dst.exists()
    assert dst.stat().st_size > 0
    loaded = trimesh.load(str(dst), force="mesh", process=False)
    assert loaded is not None


def test_convert_glb_to_glb_round_trips(synthetic_glb, tmp_path):
    """GLB → GLB shouldn't lose structure. Trivial smoke test."""
    dst = tmp_path / "round.glb"
    convert_mesh(synthetic_glb, dst)
    assert dst.exists()


def test_convert_invalid_source_raises(tmp_path):
    bad = tmp_path / "nope.glb"
    bad.write_bytes(b"not a valid mesh")
    with pytest.raises(Exception):  # noqa: B017 — trimesh raises various types
        convert_mesh(bad, tmp_path / "out.obj")
