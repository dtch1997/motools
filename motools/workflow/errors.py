"""Workflow execution error types for improved error handling."""


class WorkflowError(Exception):
    """Base class for workflow-related errors."""

    pass


class ValidationError(WorkflowError):
    """Raised when configuration or input validation fails."""

    pass


class ConfigError(WorkflowError):
    """Raised when workflow configuration is invalid."""

    pass


class NetworkError(WorkflowError):
    """Raised when a network operation fails."""

    pass


class WorkflowTimeoutError(WorkflowError):
    """Raised when a workflow operation times out."""

    pass


class RetriableError(WorkflowError):
    """Base class for errors that can be retried."""

    pass


class PermanentError(WorkflowError):
    """Base class for errors that should not be retried."""

    pass
