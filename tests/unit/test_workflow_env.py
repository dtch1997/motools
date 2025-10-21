"""Unit tests for environment variable validation."""

import os

import pytest

from motools.workflow.env import EnvConfig, EnvValidationError, validate_env


def test_validate_env_success(monkeypatch):
    """Test validation passes when all required vars are set."""
    monkeypatch.setenv("TEST_VAR_1", "value1")
    monkeypatch.setenv("TEST_VAR_2", "value2")

    config = EnvConfig(required=["TEST_VAR_1", "TEST_VAR_2"])

    # Should not raise
    validate_env(config)


def test_validate_env_missing_vars(monkeypatch):
    """Test validation fails when required vars are missing."""
    # Don't set TEST_VAR_1
    monkeypatch.setenv("TEST_VAR_2", "value2")

    config = EnvConfig(required=["TEST_VAR_1", "TEST_VAR_2"])

    with pytest.raises(EnvValidationError, match="Missing required environment variables"):
        validate_env(config)


def test_validate_env_all_missing():
    """Test validation with all vars missing."""
    # Make sure vars don't exist
    for var in ["MISSING_VAR_1", "MISSING_VAR_2"]:
        if var in os.environ:
            del os.environ[var]

    config = EnvConfig(required=["MISSING_VAR_1", "MISSING_VAR_2"])

    with pytest.raises(EnvValidationError) as exc_info:
        validate_env(config)

    error_msg = str(exc_info.value)
    assert "MISSING_VAR_1" in error_msg
    assert "MISSING_VAR_2" in error_msg


def test_validate_env_empty_config():
    """Test validation with no required vars."""
    config = EnvConfig(required=[])

    # Should not raise
    validate_env(config)


def test_env_config_with_optional():
    """Test EnvConfig with optional vars."""
    config = EnvConfig(
        required=["REQUIRED_VAR"],
        optional={"OPTIONAL_VAR": "Description of optional var"},
    )

    assert config.required == ["REQUIRED_VAR"]
    assert config.optional == {"OPTIONAL_VAR": "Description of optional var"}


def test_env_config_default_optional():
    """Test EnvConfig with default optional dict."""
    config = EnvConfig(required=["VAR1"])

    assert config.optional == {}
