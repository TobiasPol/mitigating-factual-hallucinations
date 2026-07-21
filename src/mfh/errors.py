"""Project-specific exceptions with actionable failure messages."""


class MFHError(Exception):
    """Base exception for expected research-pipeline failures."""


class ConfigurationError(MFHError, ValueError):
    """Raised when a configuration violates a reproducibility invariant."""


class DataValidationError(MFHError, ValueError):
    """Raised when benchmark or generation data violates its contract."""


class FrozenArtifactError(MFHError, RuntimeError):
    """Raised when an immutable artifact is changed or cannot be verified."""


class OptionalDependencyError(MFHError, ImportError):
    """Raised when a requested feature needs an uninstalled optional package."""
