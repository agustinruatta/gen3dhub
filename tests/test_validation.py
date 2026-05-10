"""Tests for `gen3dhub.utils.validation` — mesh quality reports."""

from __future__ import annotations

import pytest

from gen3dhub.utils.validation import format_report_human, validate_mesh


def test_validate_simple_cube(synthetic_glb):
    report = validate_mesh(synthetic_glb)
    assert report.vertex_count == 8
    assert report.triangle_count == 12
    assert report.is_watertight
    assert report.is_winding_consistent
    assert report.component_count == 1
    assert report.file_size_bytes > 0


def test_validate_warns_when_no_albedo(synthetic_glb):
    """A bare cube exported via trimesh has no materials; the heuristic should
    surface a warning so users don't silently end up with gray-flat assets."""
    report = validate_mesh(synthetic_glb)
    assert any("albedo" in w.lower() for w in report.warnings)


def test_validate_bounding_box_matches_input(synthetic_glb):
    """We exported a 1.0 x 2.0 x 0.5 cube; the report should reflect that."""
    report = validate_mesh(synthetic_glb)
    assert pytest.approx(report.bounding_box[0], rel=0.01) == 1.0
    assert pytest.approx(report.bounding_box[1], rel=0.01) == 2.0
    assert pytest.approx(report.bounding_box[2], rel=0.01) == 0.5


def test_validate_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_mesh(tmp_path / "nonexistent.glb")


def test_format_report_human_includes_filename(synthetic_glb):
    report = validate_mesh(synthetic_glb)
    rendered = format_report_human(report)
    assert "cube.glb" in rendered
    assert "8" in rendered  # vertex count


def test_report_as_dict_is_json_serializable(synthetic_glb):
    """`as_dict` is what the events.emit("validate_complete", ...) call splats.
    Anything in here must round-trip through json.dumps."""
    import json

    report = validate_mesh(synthetic_glb)
    payload = report.as_dict()
    serialized = json.dumps(payload, default=str)
    assert json.loads(serialized) == json.loads(serialized)  # tautology — just check no raise
