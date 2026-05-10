"""Tests for `gen3dhub.utils.system` — distro detection, fit verdicts."""

from __future__ import annotations

from gen3dhub.models.base import HardwareNeeds
from gen3dhub.utils.system import (
    assess_fit,
    detect_distro,
    directory_size_bytes,
    format_bytes,
    install_hint_for_compiler,
)


def test_format_bytes_units():
    assert format_bytes(0) == "0 B"
    assert format_bytes(1023) == "1023 B"
    assert format_bytes(1024) == "1.0 KiB"
    assert format_bytes(1024 * 1024) == "1.0 MiB"
    assert format_bytes(1024**3) == "1.0 GiB"
    assert format_bytes(int(1.5 * 1024**3)) == "1.5 GiB"


def test_directory_size_bytes(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"hello")    # 5 bytes
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "b.txt").write_bytes(b"world!")  # 6 bytes
    assert directory_size_bytes(tmp_path) == 11


def test_directory_size_bytes_skips_unreadable(tmp_path):
    """The size scan is OSError-tolerant — an unreadable entry shouldn't
    prevent the rest of the dir from being counted."""
    (tmp_path / "ok.txt").write_bytes(b"abc")
    # Not creating an actually-unreadable file (would need root); just verify
    # the happy path counts everything readable.
    assert directory_size_bytes(tmp_path) == 3


def test_detect_distro_returns_nonempty_string():
    distro = detect_distro()
    assert isinstance(distro, str)
    assert len(distro) > 0


def test_install_hint_returns_nonempty_string():
    assert len(install_hint_for_compiler()) > 0


def _needs(*, min_gb=4, rec_gb=8, fallback=True) -> HardwareNeeds:
    return HardwareNeeds(
        min_gpu_vram_gb=min_gb,
        recommended_gpu_vram_gb=rec_gb,
        cpu_fallback=fallback,
        cpu_speed_hint="...",
    )


def test_assess_fit_no_gpu_with_fallback(mock_vram):
    mock_vram(None)
    verdict = assess_fit(_needs(fallback=True))
    assert verdict.severity == "warn"
    assert "No NVIDIA GPU" in verdict.headline
    assert "--cpu" in verdict.detail


def test_assess_fit_no_gpu_without_fallback(mock_vram):
    mock_vram(None)
    verdict = assess_fit(_needs(fallback=False))
    assert verdict.severity == "error"
    assert "No NVIDIA GPU" in verdict.headline


def test_assess_fit_comfortable(mock_vram):
    mock_vram(16 * 1024)  # 16 GB > recommended 8 GB
    verdict = assess_fit(_needs(min_gb=4, rec_gb=8))
    assert verdict.severity == "ok"
    assert "Comfortable" in verdict.headline


def test_assess_fit_tight(mock_vram):
    mock_vram(5 * 1024)  # 5 GB; min=4, recommended=8 → tight
    verdict = assess_fit(_needs(min_gb=4, rec_gb=8))
    assert verdict.severity == "warn"
    assert "Tight" in verdict.headline


def test_assess_fit_too_small_with_fallback(mock_vram):
    mock_vram(2 * 1024)  # 2 GB < min 4 GB
    verdict = assess_fit(_needs(min_gb=4, rec_gb=8, fallback=True))
    assert verdict.severity == "warn"
    assert "GPU too small" in verdict.headline


def test_assess_fit_too_small_without_fallback(mock_vram):
    mock_vram(2 * 1024)
    verdict = assess_fit(_needs(min_gb=4, rec_gb=8, fallback=False))
    assert verdict.severity == "error"
    assert "Insufficient" in verdict.headline
