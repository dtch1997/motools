"""Unit tests for motools.imports module."""

import pytest

from motools.imports import import_function


def test_import_function_valid_path():
    """Test importing a function with valid import path."""
    # Import a real function from the standard library
    func = import_function("os.path:join")
    assert callable(func)
    # Test that it works
    result = func("a", "b")
    assert result == "a/b" or result == "a\\b"  # Works on both Unix and Windows


def test_import_function_invalid_format_no_colon():
    """Test that import_function raises ValueError for paths missing colon."""
    with pytest.raises(ValueError, match="Invalid import path.*Expected format"):
        import_function("os.path.join")


def test_import_function_module_not_found():
    """Test that import_function raises ValueError when module doesn't exist."""
    with pytest.raises(ValueError, match="Invalid import path"):
        import_function("nonexistent.module:function")


def test_import_function_attribute_not_found():
    """Test that import_function raises ValueError when function doesn't exist."""
    with pytest.raises(ValueError, match="Invalid import path"):
        import_function("os.path:nonexistent_function")


def test_import_function_not_callable():
    """Test that import_function raises ValueError when target is not callable."""
    # os.path.sep is a string constant, not callable
    with pytest.raises(ValueError, match="does not point to a callable"):
        import_function("os.path:sep")


def test_import_function_class():
    """Test that import_function works with classes (which are callable)."""
    # Classes are callable, so this should work
    cls = import_function("pathlib:Path")
    assert callable(cls)
    # Test that we can instantiate it
    obj = cls("/tmp")
    assert str(obj) == "/tmp"


def test_import_function_builtin():
    """Test importing built-in functions."""
    # Import a builtin function
    func = import_function("builtins:len")
    assert callable(func)
    assert func([1, 2, 3]) == 3
