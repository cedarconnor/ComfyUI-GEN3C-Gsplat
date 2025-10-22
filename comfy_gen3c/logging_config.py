"""Logging configuration for GEN3C nodes."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Configure logging for GEN3C nodes.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional file path for log output
        format_string: Custom format string for log messages

    Returns:
        Configured root logger for GEN3C
    """
    if format_string is None:
        format_string = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"

    # Create formatter
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Configure root logger for comfy_gen3c
    logger = logging.getLogger("comfy_gen3c")
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Don't propagate to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    Args:
        name: Module name (e.g., __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Auto-configure logging on import
try:
    # Check if logging is already configured
    root_logger = logging.getLogger("comfy_gen3c")
    if not root_logger.handlers:
        # Auto-configure with sensible defaults
        setup_logging(level=logging.INFO)
except Exception:
    # Silently fail if logging setup fails
    pass
