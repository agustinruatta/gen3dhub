"""Filesystem layout and global configuration for Model Selector."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _xdg_cache_home() -> Path:
    raw = os.environ.get("XDG_CACHE_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".cache"


@dataclass(frozen=True)
class Paths:
    """Resolved filesystem paths used by the application.

    Layout:
        <cache_root>/
            models/
                <model_id>/
                    repo/        # cloned source repo (when applicable)
                    .venv/       # uv-managed virtualenv with pinned deps
                    installed    # marker file written after successful setup
                    meta.json    # adapter-managed metadata
    """

    cache_root: Path

    @classmethod
    def default(cls) -> Paths:
        override = os.environ.get("MODEL_SELECTOR_CACHE_DIR")
        root = Path(override).expanduser() if override else _xdg_cache_home() / "model-selector"
        return cls(cache_root=root)

    @property
    def models_root(self) -> Path:
        return self.cache_root / "models"

    def model_dir(self, model_id: str) -> Path:
        return self.models_root / model_id

    def model_repo_dir(self, model_id: str) -> Path:
        return self.model_dir(model_id) / "repo"

    def model_venv_dir(self, model_id: str) -> Path:
        return self.model_dir(model_id) / ".venv"

    def model_installed_marker(self, model_id: str) -> Path:
        return self.model_dir(model_id) / "installed"

    def ensure(self) -> None:
        self.models_root.mkdir(parents=True, exist_ok=True)
