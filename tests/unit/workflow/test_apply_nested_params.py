"""Unit tests for apply_nested_params function."""

from dataclasses import dataclass

import pytest

from motools.workflow.sweep import apply_nested_params


@dataclass
class InnerConfig:
    """Test inner config."""

    value: int = 1
    name: str = "inner"


@dataclass
class MiddleConfig:
    """Test middle config."""

    inner: InnerConfig
    multiplier: int = 2


@dataclass
class OuterConfig:
    """Test outer config."""

    middle: MiddleConfig
    prefix: str = "test"


def test_apply_nested_params_single_level():
    """Test updating a single-level nested parameter."""
    config = OuterConfig(
        middle=MiddleConfig(inner=InnerConfig(value=10, name="test")),
        prefix="original",
    )

    # Update single nested field
    updated = apply_nested_params(config, {"prefix": "updated"})

    assert updated.prefix == "updated"
    assert updated.middle.inner.value == 10  # Unchanged
    assert config.prefix == "original"  # Original unchanged


def test_apply_nested_params_two_levels():
    """Test updating two-level nested parameters."""
    config = OuterConfig(
        middle=MiddleConfig(inner=InnerConfig(value=10, name="test"), multiplier=2),
        prefix="original",
    )

    # Update two-level nested field
    updated = apply_nested_params(config, {"middle.multiplier": 5})

    assert updated.middle.multiplier == 5
    assert updated.middle.inner.value == 10  # Unchanged
    assert config.middle.multiplier == 2  # Original unchanged


def test_apply_nested_params_three_levels():
    """Test updating three-level nested parameters."""
    config = OuterConfig(
        middle=MiddleConfig(inner=InnerConfig(value=10, name="test")),
        prefix="original",
    )

    # Update three-level nested field
    updated = apply_nested_params(config, {"middle.inner.value": 42})

    assert updated.middle.inner.value == 42
    assert updated.middle.inner.name == "test"  # Unchanged
    assert config.middle.inner.value == 10  # Original unchanged


def test_apply_nested_params_multiple_updates():
    """Test updating multiple nested parameters at once."""
    config = OuterConfig(
        middle=MiddleConfig(inner=InnerConfig(value=10, name="test"), multiplier=2),
        prefix="original",
    )

    # Update multiple fields
    updated = apply_nested_params(
        config,
        {
            "prefix": "new_prefix",
            "middle.multiplier": 7,
            "middle.inner.value": 100,
            "middle.inner.name": "updated",
        },
    )

    assert updated.prefix == "new_prefix"
    assert updated.middle.multiplier == 7
    assert updated.middle.inner.value == 100
    assert updated.middle.inner.name == "updated"
    # Original unchanged
    assert config.middle.inner.value == 10


def test_apply_nested_params_dict_in_dataclass():
    """Test updating dictionary fields within dataclasses."""

    @dataclass
    class ConfigWithDict:
        params: dict
        name: str = "test"

    config = ConfigWithDict(params={"learning_rate": 0.01, "epochs": 10}, name="original")

    # Update dict value through nested path
    updated = apply_nested_params(config, {"params.learning_rate": 0.001})

    assert updated.params["learning_rate"] == 0.001
    assert updated.params["epochs"] == 10  # Unchanged
    assert config.params["learning_rate"] == 0.01  # Original unchanged


def test_apply_nested_params_invalid_path():
    """Test that invalid paths raise appropriate errors."""
    config = OuterConfig(
        middle=MiddleConfig(inner=InnerConfig(value=10, name="test")),
        prefix="original",
    )

    # Non-existent field
    with pytest.raises(ValueError, match="Field 'nonexistent' not found"):
        apply_nested_params(config, {"middle.nonexistent": 5})

    # Non-existent nested path
    with pytest.raises(ValueError, match="Field 'invalid' not found"):
        apply_nested_params(config, {"middle.inner.invalid": 5})


def test_apply_nested_params_empty_updates():
    """Test that empty updates return unchanged copy."""
    config = OuterConfig(
        middle=MiddleConfig(inner=InnerConfig(value=10, name="test")),
        prefix="original",
    )

    updated = apply_nested_params(config, {})

    assert updated.prefix == "original"
    assert updated.middle.inner.value == 10
    assert updated is not config  # Should be a copy


def test_apply_nested_params_preserves_deep_copy():
    """Test that deep copy prevents mutations of nested objects."""
    config = OuterConfig(
        middle=MiddleConfig(inner=InnerConfig(value=10, name="test")),
        prefix="original",
    )

    updated = apply_nested_params(config, {"middle.inner.value": 42})

    # Modify the updated config's nested object
    updated.middle.inner.name = "modified"

    # Original should remain unchanged
    assert config.middle.inner.name == "test"
    assert config.middle.inner.value == 10
