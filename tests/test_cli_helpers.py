"""Tests for cli.py's pure helpers (param parsing, format coercion)."""

from __future__ import annotations

import pytest
import typer

from gen3dhub.cli import _coerce_param, _format_param_help, _parse_params
from gen3dhub.models.base import (
    HardwareNeeds,
    ModelInfo,
    ParamKind,
    ParamSpec,
)


def _model_with_params(*params: ParamSpec) -> ModelInfo:
    """Minimal ModelInfo with only the bits param parsing inspects."""
    return ModelInfo(
        id="test",
        display_name="Test",
        description="",
        best_for="",
        strengths=(),
        weaknesses=(),
        hardware=HardwareNeeds(
            min_gpu_vram_gb=4, recommended_gpu_vram_gb=8,
            cpu_fallback=True, cpu_speed_hint="",
        ),
        homepage="",
        license_url=None,
        requires_hf_auth=False,
        inputs=(),
        output_extension=".glb",
        params=params,
    )


@pytest.fixture
def model_all_kinds():
    return _model_with_params(
        ParamSpec(name="size", label="", description="", kind=ParamKind.INT, default=10),
        ParamSpec(
            name="mode", label="", description="", kind=ParamKind.SELECT,
            default="a", choices=("a", "b", "c"),
        ),
        ParamSpec(name="rate", label="", description="", kind=ParamKind.FLOAT, default=0.5),
        ParamSpec(name="enabled", label="", description="", kind=ParamKind.BOOL, default=True),
        ParamSpec(name="prompt", label="", description="", kind=ParamKind.TEXT, default=""),
    )


def test_parse_empty_returns_empty(model_all_kinds):
    assert _parse_params(model_all_kinds, []) == {}


def test_parse_int(model_all_kinds):
    assert _parse_params(model_all_kinds, ["size=42"]) == {"size": 42}


def test_parse_int_negative(model_all_kinds):
    assert _parse_params(model_all_kinds, ["size=-1"]) == {"size": -1}


def test_parse_float(model_all_kinds):
    assert _parse_params(model_all_kinds, ["rate=0.75"]) == {"rate": 0.75}


@pytest.mark.parametrize("truthy", ["true", "True", "1", "yes", "y", "on"])
def test_parse_bool_truthy(model_all_kinds, truthy):
    assert _parse_params(model_all_kinds, [f"enabled={truthy}"]) == {"enabled": True}


@pytest.mark.parametrize("falsy", ["false", "False", "0", "no", "n", "off"])
def test_parse_bool_falsy(model_all_kinds, falsy):
    assert _parse_params(model_all_kinds, [f"enabled={falsy}"]) == {"enabled": False}


def test_parse_select_valid(model_all_kinds):
    assert _parse_params(model_all_kinds, ["mode=b"]) == {"mode": "b"}


def test_parse_text_with_spaces(model_all_kinds):
    assert _parse_params(model_all_kinds, ["prompt=hello world"]) == {"prompt": "hello world"}


def test_parse_multiple(model_all_kinds):
    result = _parse_params(model_all_kinds, ["size=42", "mode=c", "enabled=false"])
    assert result == {"size": 42, "mode": "c", "enabled": False}


def test_parse_unknown_key_raises(model_all_kinds):
    with pytest.raises(typer.BadParameter, match="Unknown parameter"):
        _parse_params(model_all_kinds, ["bogus=1"])


def test_parse_missing_equals_raises(model_all_kinds):
    with pytest.raises(typer.BadParameter, match="KEY=VALUE"):
        _parse_params(model_all_kinds, ["bogus"])


def test_parse_bad_int_raises(model_all_kinds):
    with pytest.raises(typer.BadParameter, match="integer"):
        _parse_params(model_all_kinds, ["size=abc"])


def test_parse_bad_float_raises(model_all_kinds):
    with pytest.raises(typer.BadParameter, match="number"):
        _parse_params(model_all_kinds, ["rate=abc"])


def test_parse_bad_bool_raises(model_all_kinds):
    with pytest.raises(typer.BadParameter, match="boolean"):
        _parse_params(model_all_kinds, ["enabled=maybe"])


def test_parse_bad_select_raises(model_all_kinds):
    with pytest.raises(typer.BadParameter, match="not one of"):
        _parse_params(model_all_kinds, ["mode=z"])


def test_parse_param_on_paramless_model_raises():
    """If the model declares no params and the user passes one, fail loudly
    instead of silently ignoring."""
    model = _model_with_params()  # no params
    with pytest.raises(typer.BadParameter, match="takes no tunable parameters"):
        _parse_params(model, ["size=1"])


def test_format_param_help_lists_choices(model_all_kinds):
    help_str = _format_param_help(model_all_kinds)
    assert "size=<int>" in help_str
    assert "mode=a|b|c" in help_str


def test_format_param_help_empty_for_paramless():
    model = _model_with_params()
    assert _format_param_help(model) == "(none)"


def test_coerce_select_invalid_raises():
    spec = ParamSpec(
        name="mode", label="", description="", kind=ParamKind.SELECT,
        default="a", choices=("a", "b"),
    )
    with pytest.raises(ValueError, match="not one of"):
        _coerce_param("z", spec)


def test_coerce_select_valid_returns_value():
    spec = ParamSpec(
        name="mode", label="", description="", kind=ParamKind.SELECT,
        default="a", choices=("a", "b"),
    )
    assert _coerce_param("b", spec) == "b"
