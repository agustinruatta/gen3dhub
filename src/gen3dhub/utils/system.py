"""Cross-distro system checks (compiler, headers, etc.) and install hints.

The Python side of gen3dhub is portable thanks to uv-managed venvs, but
some adapters (e.g. Stable Fast 3D) compile C/C++ extensions from source and
need a working host toolchain. The helpers here detect what's missing and
print a distro-aware command the user can copy-paste to install it.
"""

from __future__ import annotations

import platform
import shutil
import sys
from pathlib import Path


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
        import subprocess

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


def system_summary() -> str:
    """One-line description of the host environment, useful for doctor output."""
    return (
        f"{detect_distro()} · {platform.machine()} · "
        f"python {sys.version_info.major}.{sys.version_info.minor} · "
        f"GPU: {'yes (NVIDIA)' if has_nvidia_gpu() else 'no / non-NVIDIA'}"
    )
