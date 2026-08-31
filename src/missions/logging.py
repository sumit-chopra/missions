"""Shared structlog configuration for all missions."""

import sys

import structlog

from missions.pii_mask import PIIMasker


def setup_logging() -> None:
    """Configure structlog with a console renderer writing to stderr."""
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            # Mask PII before rendering
            PIIMasker(),
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ],
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )
