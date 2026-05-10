"""Tests for `gen3dhub.history` — JSONL log of past runs."""

from __future__ import annotations

from gen3dhub import history as hist
from gen3dhub.config import Paths


def _make_entry(model: str = "sf3d", **overrides) -> hist.HistoryEntry:
    base = {
        "id": hist.make_id(),
        "timestamp": hist.now_iso(),
        "model": model,
        "inputs": {"image": "/tmp/cat.png"},
        "params": {},
        "output": "/tmp/cat.glb",
        "preview": None,
        "duration_s": 1.5,
        "exit_code": 0,
    }
    base.update(overrides)
    return hist.HistoryEntry(**base)


def test_append_then_read_round_trips(temp_cache_dir):
    paths = Paths.default()
    paths.ensure()

    entry = _make_entry()
    hist.append(paths, entry)

    entries = hist.read_all(paths)
    assert len(entries) == 1
    assert entries[0].id == entry.id
    assert entries[0].duration_s == 1.5


def test_append_preserves_chronological_order(temp_cache_dir):
    paths = Paths.default()
    paths.ensure()

    e1 = _make_entry(model="sf3d", id="001")
    e2 = _make_entry(model="hunyuan3d-2", id="002")
    hist.append(paths, e1)
    hist.append(paths, e2)

    entries = hist.read_all(paths)
    assert [e.id for e in entries] == ["001", "002"]


def test_find_by_full_id(temp_cache_dir):
    paths = Paths.default()
    paths.ensure()
    hist.append(paths, _make_entry(id="20260510-150234-abc123"))

    found = hist.find(paths, "20260510-150234-abc123")
    assert found is not None
    assert found.id == "20260510-150234-abc123"


def test_find_by_prefix(temp_cache_dir):
    paths = Paths.default()
    paths.ensure()
    hist.append(paths, _make_entry(id="20260510-150234-abc123"))

    found = hist.find(paths, "20260510-150234")
    assert found is not None


def test_find_returns_none_when_no_match(temp_cache_dir):
    paths = Paths.default()
    paths.ensure()
    hist.append(paths, _make_entry(id="20260510-150234-abc123"))

    assert hist.find(paths, "doesnotexist") is None


def test_find_returns_most_recent_on_prefix_collision(temp_cache_dir):
    paths = Paths.default()
    paths.ensure()
    hist.append(paths, _make_entry(id="20260510-150234-aaa", model="sf3d"))
    hist.append(paths, _make_entry(id="20260510-150235-bbb", model="hunyuan3d-2"))

    # Both entries share the "20260510" prefix; find returns the last-written
    # match by design — agents that need a specific entry should pass a
    # longer prefix.
    found = hist.find(paths, "20260510")
    assert found is not None
    assert found.model == "hunyuan3d-2"


def test_read_all_empty_when_no_log(temp_cache_dir):
    paths = Paths.default()
    paths.ensure()
    assert hist.read_all(paths) == []


def test_malformed_lines_skipped_silently(temp_cache_dir):
    paths = Paths.default()
    paths.ensure()
    log = paths.cache_root / "history.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    # Mix of: invalid JSON, missing-field JSON, valid entry. Reader should
    # return only the valid one and not raise.
    log.write_text(
        'not json\n'
        '{"only": "junk"}\n'
        '{"id":"x","timestamp":"t","model":"sf3d","inputs":{},"params":{},'
        '"output":null,"preview":null,"duration_s":1.0,"exit_code":0}\n'
    )
    entries = hist.read_all(paths)
    assert len(entries) == 1
    assert entries[0].id == "x"
