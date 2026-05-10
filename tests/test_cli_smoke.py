"""End-to-end-ish smoke tests for the CLI surface.

Runs the Typer app in-process via `typer.testing.CliRunner`. We don't actually
invoke any model (that requires the full per-model venv + GPU), but we do
exercise every read-only command and the validation paths that don't need a
running adapter.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from gen3dhub.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def test_help_runs():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "gen3dhub" in result.stdout.lower()


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "gen3dhub" in result.stdout
    # Should print a version-looking string.
    assert any(ch.isdigit() for ch in result.stdout)


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def test_list_shows_all_registered_models():
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    for model_id in ("stable-fast-3d", "hunyuan3d-2", "paint3d"):
        assert model_id in result.stdout


def test_list_shows_strengths_and_weaknesses():
    result = runner.invoke(app, ["list"])
    assert "Strong" in result.stdout
    assert "Weak" in result.stdout
    assert "Best for" in result.stdout


# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------


def test_describe_emits_valid_json():
    result = runner.invoke(app, ["describe"])
    assert result.exit_code == 0
    # CliRunner mixes stdout/stderr by default; describe writes only to stdout.
    parsed = json.loads(result.stdout)
    assert parsed["tool"] == "gen3dhub"
    assert "version" in parsed
    assert isinstance(parsed["models"], list)
    assert len(parsed["models"]) >= 3
    assert "events" in parsed
    assert "exit_codes" in parsed
    assert "supported_output_formats" in parsed


def test_describe_each_model_has_complete_schema():
    """Every adapter declares the full schema fields agents rely on."""
    result = runner.invoke(app, ["describe"])
    parsed = json.loads(result.stdout)
    for model in parsed["models"]:
        assert "id" in model
        assert "description" in model
        assert "best_for" in model
        assert "strengths" in model
        assert "weaknesses" in model
        assert "hardware" in model
        assert "inputs" in model
        assert "params" in model
        # Hardware fields the assess_fit() function reads.
        for key in (
            "min_gpu_vram_gb", "recommended_gpu_vram_gb",
            "cpu_fallback", "cpu_speed_hint",
        ):
            assert key in model["hardware"]


def test_describe_pretty_indents():
    result = runner.invoke(app, ["describe", "--pretty"])
    assert result.exit_code == 0
    # Pretty output should have many newlines vs the single-line default.
    assert result.stdout.count("\n") > 50
    # Still valid JSON.
    json.loads(result.stdout)


def test_describe_lists_supported_formats():
    result = runner.invoke(app, ["describe"])
    parsed = json.loads(result.stdout)
    formats = parsed["supported_output_formats"]
    assert "glb" in formats
    assert "obj" in formats


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def test_validate_runs_on_synthetic_glb(synthetic_glb):
    result = runner.invoke(app, ["validate", str(synthetic_glb)])
    assert result.exit_code == 0
    assert "verts" in result.stdout
    assert "tris" in result.stdout


def test_validate_json_emits_one_line_json(synthetic_glb):
    result = runner.invoke(app, ["validate", str(synthetic_glb), "--json"])
    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert parsed["vertex_count"] == 8
    assert parsed["triangle_count"] == 12


def test_validate_missing_file_exits_nonzero(tmp_path):
    result = runner.invoke(app, ["validate", str(tmp_path / "nonexistent.glb")])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------


def test_history_empty_message(temp_cache_dir):
    result = runner.invoke(app, ["history"])
    assert result.exit_code == 0
    assert "No runs recorded" in result.stdout


def test_history_json_empty_emits_nothing(temp_cache_dir):
    result = runner.invoke(app, ["history", "--json"])
    assert result.exit_code == 0
    assert result.stdout.strip() == ""


def test_history_with_entries_renders_table(temp_cache_dir):
    """Append a fake entry, then verify history reads it back via the CLI."""
    from gen3dhub import history as hist
    from gen3dhub.config import Paths

    paths = Paths.default()
    paths.ensure()
    hist.append(
        paths,
        hist.HistoryEntry(
            id="20260510-150234-abc123",
            timestamp="2026-05-10T15:02:34Z",
            model="stable-fast-3d",
            inputs={"image": "/tmp/cat.png"},
            params={"texture_resolution": "2048"},
            output="/tmp/cat.glb",
            preview="/tmp/cat.preview.png",
            duration_s=1.4,
            exit_code=0,
        ),
    )

    result = runner.invoke(app, ["history"])
    assert result.exit_code == 0
    assert "stable-fast-3d" in result.stdout
    assert "cat.glb" in result.stdout


def test_history_json_emits_ndjson(temp_cache_dir):
    from gen3dhub import history as hist
    from gen3dhub.config import Paths

    paths = Paths.default()
    paths.ensure()
    hist.append(
        paths,
        hist.HistoryEntry(
            id="abc", timestamp="t", model="sf3d",
            inputs={}, params={},
            output=None, preview=None, duration_s=0, exit_code=0,
        ),
    )
    result = runner.invoke(app, ["history", "--json"])
    assert result.exit_code == 0
    lines = [json.loads(line) for line in result.stdout.strip().splitlines()]
    assert len(lines) == 1
    assert lines[0]["model"] == "sf3d"


def test_history_rerun_prints_command(temp_cache_dir):
    from gen3dhub import history as hist
    from gen3dhub.config import Paths

    paths = Paths.default()
    paths.ensure()
    hist.append(
        paths,
        hist.HistoryEntry(
            id="20260510-150234-abc123",
            timestamp="t", model="stable-fast-3d",
            inputs={"image": "/tmp/cat.png"},
            params={"remesh_option": "quad"},
            output="/tmp/cat.glb",
            preview=None, duration_s=1.0, exit_code=0,
        ),
    )

    result = runner.invoke(app, ["history", "--rerun", "20260510"])
    assert result.exit_code == 0
    assert "gen3dhub run" in result.stdout
    assert "--model stable-fast-3d" in result.stdout
    assert "--image /tmp/cat.png" in result.stdout
    assert "--param remesh_option=quad" in result.stdout
    assert "--yes" in result.stdout


def test_history_rerun_unknown_id_exits_nonzero(temp_cache_dir):
    result = runner.invoke(app, ["history", "--rerun", "doesnotexist"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# doctor (without per-model installs)
# ---------------------------------------------------------------------------


def test_doctor_runs_without_crashing(temp_cache_dir):
    result = runner.invoke(app, ["doctor"])
    # Doctor exits non-zero when models aren't installed, which is the
    # expected state in a fresh tempdir. Just verify it runs.
    assert "Host:" in result.stdout
