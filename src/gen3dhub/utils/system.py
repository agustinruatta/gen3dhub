"""Cross-distro system checks (compiler, headers, etc.) and install hints.

The Python side of gen3dhub is portable thanks to uv-managed venvs, but
some adapters (e.g. Stable Fast 3D) compile C/C++ extensions from source and
need a working host toolchain. The helpers here detect what's missing and
print a distro-aware command the user can copy-paste to install it.
"""

from __future__ import annotations

import functools
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gen3dhub.models.base import HardwareNeeds


def has_command(name: str) -> bool:
    """Return True if `name` is found on PATH."""
    return shutil.which(name) is not None


def has_c_compiler() -> bool:
    return has_command("gcc") or has_command("cc") or has_command("clang")


def has_cpp_compiler() -> bool:
    return has_command("g++") or has_command("c++") or has_command("clang++")


def has_nvidia_gpu() -> bool:
    """Heuristic: is `nvidia-smi` available and does it report a device?"""
    if not has_command("nvidia-smi"):
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return result.returncode == 0 and "GPU" in result.stdout
    except Exception:
        return False


@functools.cache
def gpu_vram_mb() -> int | None:
    """Total VRAM (MiB) of the largest visible NVIDIA GPU, or None.

    Cached for the life of the process — VRAM doesn't change at runtime, and
    `gen3dhub list` calls this once per model.
    """
    if not has_nvidia_gpu():
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        values = [
            int(line.strip())
            for line in result.stdout.splitlines()
            if line.strip().isdigit()
        ]
        return max(values) if values else None
    except Exception:
        return None


def detect_distro() -> str:
    """Return a normalized distro id ('ubuntu', 'arch', 'fedora', 'macos', ...)."""
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("win"):
        return "windows"
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("ID="):
                return line.split("=", 1)[1].strip().strip('"').lower()
            if line.startswith("ID_LIKE=") and "ID=" not in line:
                # Fallback if ID isn't present
                return line.split("=", 1)[1].strip().strip('"').lower().split()[0]
    except FileNotFoundError:
        pass
    return "unknown"


# Keys are ID values from /etc/os-release (or special-cased platforms).
_COMPILER_INSTALL_HINTS: dict[str, str] = {
    "ubuntu": "sudo apt install build-essential python3-dev",
    "debian": "sudo apt install build-essential python3-dev",
    "linuxmint": "sudo apt install build-essential python3-dev",
    "pop": "sudo apt install build-essential python3-dev",
    "elementary": "sudo apt install build-essential python3-dev",
    "arch": "sudo pacman -S --needed base-devel",
    "manjaro": "sudo pacman -S --needed base-devel",
    "endeavouros": "sudo pacman -S --needed base-devel",
    "cachyos": "sudo pacman -S --needed base-devel",
    "fedora": "sudo dnf groupinstall 'Development Tools' && sudo dnf install python3-devel",
    "rhel": "sudo dnf groupinstall 'Development Tools' && sudo dnf install python3-devel",
    "centos": "sudo dnf groupinstall 'Development Tools' && sudo dnf install python3-devel",
    "rocky": "sudo dnf groupinstall 'Development Tools' && sudo dnf install python3-devel",
    "alma": "sudo dnf groupinstall 'Development Tools' && sudo dnf install python3-devel",
    "opensuse-leap": "sudo zypper install -t pattern devel_C_C++",
    "opensuse-tumbleweed": "sudo zypper install -t pattern devel_C_C++",
    "macos": "xcode-select --install",
}


def install_hint_for_compiler() -> str:
    """Return a copy-pasteable command for the current distro, or generic guidance."""
    distro = detect_distro()
    hint = _COMPILER_INSTALL_HINTS.get(distro)
    if hint:
        return f"On {distro}, install with: {hint}"
    return (
        f"Detected distro '{distro}'. Install a C/C++ toolchain (gcc + g++) "
        "and Python development headers using your distro's package manager."
    )


def check_build_toolchain() -> list[str]:
    """Return a list of human-readable problems with the host build toolchain.

    Empty list means the toolchain looks fine for compiling Python C/C++
    extensions. Used by adapters that build native code at install time.
    """
    problems: list[str] = []
    if not has_c_compiler():
        problems.append(
            f"No C compiler found on PATH (looked for gcc/cc/clang). "
            f"{install_hint_for_compiler()}"
        )
    if not has_cpp_compiler():
        problems.append(
            f"No C++ compiler found on PATH (looked for g++/c++/clang++). "
            f"{install_hint_for_compiler()}"
        )
    return problems


def directory_size_bytes(path: Path) -> int:
    """Total size of all regular files under `path`. Symlinks are not followed.

    Used by `gen3dhub uninstall` to tell the user how much disk it's about to
    free. Best-effort: unreadable files are silently skipped rather than raising,
    because partial-install dirs can have transient permission issues.
    """
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file() and not entry.is_symlink():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def format_bytes(num_bytes: int) -> str:
    """Pretty-print a byte count using KiB / MiB / GiB units."""
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{num_bytes} B"  # unreachable


def system_summary() -> str:
    """One-line description of the host environment, useful for doctor output."""
    vram = gpu_vram_mb()
    gpu_descr = f"NVIDIA {vram / 1024:.1f} GB" if vram else "no NVIDIA GPU"
    return (
        f"{detect_distro()} · {platform.machine()} · "
        f"python {sys.version_info.major}.{sys.version_info.minor} · "
        f"GPU: {gpu_descr}"
    )


@dataclass(frozen=True)
class FitVerdict:
    """Result of comparing a model's HardwareNeeds against the current host."""

    severity: str  # "ok" | "warn" | "error"
    headline: str  # one-line verdict
    detail: str    # supplementary line; may be empty


def assess_fit(needs: HardwareNeeds) -> FitVerdict:
    """Compare a model's declared hardware needs against the local host.

    Returns a FitVerdict the CLI/TUI can render. Best-effort and never raises:
    no GPU info still produces a verdict (CPU-mode warning or hard error
    depending on whether the model supports CPU fallback).
    """
    vram_mb = gpu_vram_mb()

    if vram_mb is None:
        if needs.cpu_fallback:
            return FitVerdict(
                severity="warn",
                headline=f"No NVIDIA GPU detected — CPU only ({needs.cpu_speed_hint})",
                detail="Pass --cpu when running.",
            )
        return FitVerdict(
            severity="error",
            headline="No NVIDIA GPU detected and this model has no CPU fallback",
            detail="",
        )

    vram_gb = vram_mb / 1024

    if vram_gb >= needs.recommended_gpu_vram_gb:
        return FitVerdict(
            severity="ok",
            headline=(
                f"Comfortable on GPU "
                f"(have {vram_gb:.1f} GB · needs {needs.recommended_gpu_vram_gb:.0f} GB)"
            ),
            detail="",
        )

    if vram_gb >= needs.min_gpu_vram_gb:
        return FitVerdict(
            severity="warn",
            headline=(
                f"Tight fit on GPU "
                f"(have {vram_gb:.1f} GB · needs {needs.min_gpu_vram_gb:.0f} GB min, "
                f"{needs.recommended_gpu_vram_gb:.0f} GB comfortable)"
            ),
            detail="Close other GPU-using apps before running. If OOM, fall back to --cpu.",
        )

    if needs.cpu_fallback:
        return FitVerdict(
            severity="warn",
            headline=(
                f"GPU too small (have {vram_gb:.1f} GB · "
                f"needs {needs.min_gpu_vram_gb:.0f} GB)"
            ),
            detail=f"Use --cpu ({needs.cpu_speed_hint}).",
        )

    return FitVerdict(
        severity="error",
        headline=(
            f"Insufficient VRAM (have {vram_gb:.1f} GB · "
            f"needs {needs.min_gpu_vram_gb:.0f} GB) and no CPU fallback"
        ),
        detail="",
    )
