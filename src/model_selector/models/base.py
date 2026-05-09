"""Abstract base class every model adapter implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from model_selector.config import Paths


class InputKind(StrEnum):
    """The kinds of inputs a model can accept."""

    IMAGE = "image"
    TEXT = "text"


@dataclass(frozen=True)
class InputSpec:
    """Declares an input slot accepted by a model adapter."""

    kind: InputKind
    name: str
    description: str
    required: bool = True


@dataclass(frozen=True)
class ModelInfo:
    """Static metadata describing a model."""

    id: str
    display_name: str
    summary: str
    homepage: str
    license_url: str | None
    requires_hf_auth: bool
    inputs: tuple[InputSpec, ...]
    output_extension: str  # e.g. ".glb"


@dataclass
class RunRequest:
    """Resolved inputs and options handed to an adapter at inference time."""

    inputs: dict[str, str | Path] = field(default_factory=dict)
    output_path: Path | None = None
    extra: dict[str, str] = field(default_factory=dict)


class ModelAdapter(ABC):
    """Lifecycle: setup() once, then verify() then run() any number of times."""

    info: ModelInfo  # subclasses must define a class attribute

    def __init__(self, paths: Paths) -> None:
        self.paths = paths

    @property
    def model_id(self) -> str:
        return self.info.id

    @property
    def is_installed(self) -> bool:
        return self.paths.model_installed_marker(self.model_id).exists()

    @abstractmethod
    def setup(self, *, force: bool = False) -> None:
        """Download weights, clone source, install dependencies into an isolated venv."""

    @abstractmethod
    def verify(self) -> list[str]:
        """Return a list of human-readable problems. Empty list means OK."""

    @abstractmethod
    def run(self, request: RunRequest) -> Path:
        """Execute inference. Returns the path to the produced artifact."""

    def mark_installed(self) -> None:
        marker = self.paths.model_installed_marker(self.model_id)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
