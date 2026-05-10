"""Tests for `gen3dhub.events` — NDJSON event emitter for `run --json`."""

from __future__ import annotations

import json

import pytest

from gen3dhub import events


def test_emit_no_op_when_disabled(capsys):
    events.set_json_mode(False)
    events.emit("test", x=1)
    out = capsys.readouterr().out
    assert out == ""


def test_emit_writes_one_ndjson_line(capsys):
    events.set_json_mode(True)
    events.emit("test_event", x=1, y="hello")
    out = capsys.readouterr().out
    assert out.endswith("\n")
    parsed = json.loads(out)
    assert parsed["event"] == "test_event"
    assert parsed["x"] == 1
    assert parsed["y"] == "hello"
    assert "ts" in parsed  # ISO timestamp


def test_emit_handles_path_via_default_str(capsys, tmp_path):
    """Path objects aren't JSON-serializable by default — events.emit uses
    default=str so they fall through to their str() representation."""
    events.set_json_mode(True)
    events.emit("test", path=tmp_path)
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["path"] == str(tmp_path)


def test_phase_emits_start_and_complete(capsys):
    events.set_json_mode(True)
    with events.phase("inference", model="sf3d"):
        pass
    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert len(lines) == 2
    assert lines[0]["event"] == "inference_start"
    assert lines[0]["model"] == "sf3d"
    assert lines[1]["event"] == "inference_complete"
    assert lines[1]["model"] == "sf3d"
    assert "duration_s" in lines[1]


def test_phase_emits_failed_on_exception(capsys):
    events.set_json_mode(True)
    with pytest.raises(ValueError), events.phase("inference", model="sf3d"):
        raise ValueError("boom")
    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert len(lines) == 2
    assert lines[1]["event"] == "inference_failed"
    assert lines[1]["error"] == "boom"
    assert lines[1]["error_type"] == "ValueError"
    assert "duration_s" in lines[1]


def test_phase_no_op_outside_json_mode(capsys):
    events.set_json_mode(False)
    with events.phase("inference"):
        pass
    assert capsys.readouterr().out == ""
