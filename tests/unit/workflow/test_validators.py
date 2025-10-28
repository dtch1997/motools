"""Unit tests for workflow config validators."""

import pytest

from motools.workflow.validators import (
    validate_enum,
    validate_import_path,
    validate_model_name,
    validate_non_empty_string,
    validate_non_negative_int,
    validate_positive_float,
    validate_positive_int,
    validate_probability,
    validate_range,
)


class TestValidatePositiveInt:
    """Test validate_positive_int function."""

    def test_valid_positive_int(self):
        """Test valid positive integers."""
        assert validate_positive_int(1) == 1
        assert validate_positive_int(100) == 100
        assert validate_positive_int(1000000) == 1000000

    def test_zero_fails(self):
        """Test that zero fails validation."""
        with pytest.raises(ValueError, match="value must be positive, got 0"):
            validate_positive_int(0)

    def test_negative_fails(self):
        """Test that negative values fail validation."""
        with pytest.raises(ValueError, match="value must be positive, got -1"):
            validate_positive_int(-1)
        with pytest.raises(ValueError, match="value must be positive, got -100"):
            validate_positive_int(-100)

    def test_custom_field_name(self):
        """Test custom field name in error message."""
        with pytest.raises(ValueError, match="n_epochs must be positive, got -5"):
            validate_positive_int(-5, "n_epochs")


class TestValidateNonNegativeInt:
    """Test validate_non_negative_int function."""

    def test_valid_non_negative_int(self):
        """Test valid non-negative integers."""
        assert validate_non_negative_int(0) == 0
        assert validate_non_negative_int(1) == 1
        assert validate_non_negative_int(100) == 100

    def test_negative_fails(self):
        """Test that negative values fail validation."""
        with pytest.raises(ValueError, match="value must be non-negative, got -1"):
            validate_non_negative_int(-1)


class TestValidateRange:
    """Test validate_range function."""

    def test_valid_range(self):
        """Test values within range."""
        assert validate_range(5, 1, 10) == 5
        assert validate_range(1.5, 1.0, 2.0) == 1.5
        assert validate_range(0, 0, 100) == 0

    def test_min_only(self):
        """Test with only minimum value."""
        assert validate_range(50, min_val=10) == 50
        with pytest.raises(ValueError, match="value must be at least 10, got 5"):
            validate_range(5, min_val=10)

    def test_max_only(self):
        """Test with only maximum value."""
        assert validate_range(5, max_val=10) == 5
        with pytest.raises(ValueError, match="value must be at most 10, got 15"):
            validate_range(15, max_val=10)

    def test_below_min_fails(self):
        """Test that values below minimum fail."""
        with pytest.raises(ValueError, match="value must be at least 1, got 0"):
            validate_range(0, 1, 10)

    def test_above_max_fails(self):
        """Test that values above maximum fail."""
        with pytest.raises(ValueError, match="value must be at most 10, got 11"):
            validate_range(11, 1, 10)


class TestValidateImportPath:
    """Test validate_import_path function."""

    def test_valid_import_paths(self):
        """Test valid import paths."""
        assert validate_import_path("module:function") == "module:function"
        assert validate_import_path("package.module:function") == "package.module:function"
        assert (
            validate_import_path("deep.package.module:function_name")
            == "deep.package.module:function_name"
        )

    def test_empty_string_fails(self):
        """Test that empty string fails."""
        with pytest.raises(ValueError, match="import_path cannot be empty"):
            validate_import_path("")

    def test_no_colon_fails(self):
        """Test that paths without colon fail."""
        with pytest.raises(ValueError, match="Invalid import_path format"):
            validate_import_path("module.function")

    def test_empty_module_fails(self):
        """Test that empty module path fails."""
        with pytest.raises(ValueError, match="Module path in import_path cannot be empty"):
            validate_import_path(":function")

    def test_empty_function_fails(self):
        """Test that empty function name fails."""
        with pytest.raises(ValueError, match="Function name in import_path cannot be empty"):
            validate_import_path("module:")

    def test_invalid_module_path_fails(self):
        """Test that invalid module paths fail."""
        with pytest.raises(ValueError, match="Invalid module path"):
            validate_import_path("123invalid:function")
        with pytest.raises(ValueError, match="Invalid module path"):
            validate_import_path("module-name:function")

    def test_invalid_function_name_fails(self):
        """Test that invalid function names fail."""
        with pytest.raises(ValueError, match="Invalid function name"):
            validate_import_path("module:123invalid")
        with pytest.raises(ValueError, match="Invalid function name"):
            validate_import_path("module:function-name")

    def test_custom_field_name(self):
        """Test custom field name in error message."""
        with pytest.raises(ValueError, match="Invalid dataset_loader format"):
            validate_import_path("invalid", "dataset_loader")


class TestValidateModelName:
    """Test validate_model_name function."""

    def test_valid_known_models(self):
        """Test valid known model names."""
        assert validate_model_name("gpt-4o-mini-2024-07-18") == "gpt-4o-mini-2024-07-18"
        assert validate_model_name("gpt-4o-2024-08-06") == "gpt-4o-2024-08-06"
        assert validate_model_name("claude-3-opus-20240229") == "claude-3-opus-20240229"

    def test_valid_custom_models(self):
        """Test valid custom model names."""
        assert validate_model_name("custom-model-v1") == "custom-model-v1"
        assert validate_model_name("huggingface/model-name") == "huggingface/model-name"
        assert validate_model_name("organization/repo") == "organization/repo"

    def test_empty_string_fails(self):
        """Test that empty string fails."""
        with pytest.raises(ValueError, match="model name cannot be empty"):
            validate_model_name("")

    def test_invalid_model_name_fails(self):
        """Test that invalid model names fail."""
        with pytest.raises(ValueError, match="Invalid model"):
            validate_model_name("invalid-model-name")

    def test_custom_field_name(self):
        """Test custom field name in error message."""
        with pytest.raises(ValueError, match="Invalid base_model"):
            validate_model_name("invalid", "base_model")


class TestValidatePositiveFloat:
    """Test validate_positive_float function."""

    def test_valid_positive_float(self):
        """Test valid positive floats."""
        assert validate_positive_float(0.1) == 0.1
        assert validate_positive_float(1.5) == 1.5
        assert validate_positive_float(100.0) == 100.0

    def test_zero_fails(self):
        """Test that zero fails validation."""
        with pytest.raises(ValueError, match="value must be positive, got 0"):
            validate_positive_float(0.0)

    def test_negative_fails(self):
        """Test that negative values fail validation."""
        with pytest.raises(ValueError, match="value must be positive, got -0.1"):
            validate_positive_float(-0.1)


class TestValidateProbability:
    """Test validate_probability function."""

    def test_valid_probabilities(self):
        """Test valid probability values."""
        assert validate_probability(0.0) == 0.0
        assert validate_probability(0.5) == 0.5
        assert validate_probability(1.0) == 1.0

    def test_below_zero_fails(self):
        """Test that values below 0 fail."""
        with pytest.raises(ValueError, match="probability must be between 0 and 1, got -0.1"):
            validate_probability(-0.1)

    def test_above_one_fails(self):
        """Test that values above 1 fail."""
        with pytest.raises(ValueError, match="probability must be between 0 and 1, got 1.1"):
            validate_probability(1.1)


class TestValidateNonEmptyString:
    """Test validate_non_empty_string function."""

    def test_valid_strings(self):
        """Test valid non-empty strings."""
        assert validate_non_empty_string("hello") == "hello"
        assert validate_non_empty_string("  text  ") == "  text  "

    def test_empty_string_fails(self):
        """Test that empty string fails."""
        with pytest.raises(ValueError, match="value cannot be empty"):
            validate_non_empty_string("")

    def test_whitespace_only_fails(self):
        """Test that whitespace-only string fails."""
        with pytest.raises(ValueError, match="value cannot be empty"):
            validate_non_empty_string("   ")


class TestValidateEnum:
    """Test validate_enum function."""

    def test_valid_enum_values(self):
        """Test valid enum values."""
        assert validate_enum("a", {"a", "b", "c"}) == "a"
        assert validate_enum("b", ["a", "b", "c"]) == "b"
        assert validate_enum(1, {1, 2, 3}) == 1

    def test_invalid_enum_value_fails(self):
        """Test that invalid enum values fail."""
        with pytest.raises(ValueError, match="Invalid value: d. Must be one of"):
            validate_enum("d", {"a", "b", "c"})

    def test_custom_field_name(self):
        """Test custom field name in error message."""
        with pytest.raises(ValueError, match="Invalid backend: invalid"):
            validate_enum("invalid", {"openai", "tinker"}, "backend")
