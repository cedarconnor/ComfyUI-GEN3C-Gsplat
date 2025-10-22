"""Custom exceptions for GEN3C nodes."""

from __future__ import annotations


class Gen3CError(Exception):
    """Base exception for all GEN3C-related errors."""
    pass


class Gen3CDatasetError(Gen3CError):
    """Raised when dataset operations fail."""
    pass


class Gen3CTrajectoryError(Gen3CError):
    """Raised when trajectory is invalid or missing required data."""
    pass


class Gen3CPoseRecoveryError(Gen3CError):
    """Raised when pose recovery fails."""
    pass


class Gen3CValidationError(Gen3CError):
    """Raised when validation checks fail."""
    pass


class Gen3CTrainingError(Gen3CError):
    """Raised when training operations fail."""
    pass


class Gen3CInvalidInputError(Gen3CError):
    """Raised when node inputs are invalid."""
    pass
