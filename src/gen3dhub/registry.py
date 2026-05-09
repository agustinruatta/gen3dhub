"""Central registry of supported models. Add new adapters here."""

from __future__ import annotations

from gen3dhub.config import Paths
from gen3dhub.models.base import ModelAdapter, ModelInfo
from gen3dhub.models.stable_fast_3d import StableFast3DAdapter

_ADAPTERS: dict[str, type[ModelAdapter]] = {
    StableFast3DAdapter.info.id: StableFast3DAdapter,
}


def list_models() -> list[ModelInfo]:
    return [adapter.info for adapter in _ADAPTERS.values()]


def get_adapter(model_id: str, paths: Paths) -> ModelAdapter:
    try:
        adapter_cls = _ADAPTERS[model_id]
    except KeyError as exc:
        known = ", ".join(sorted(_ADAPTERS)) or "(none)"
        raise KeyError(f"Unknown model '{model_id}'. Known models: {known}") from exc
    return adapter_cls(paths)


def known_model_ids() -> list[str]:
    return sorted(_ADAPTERS)
