import sys

import structlog


def setup_logging():
    """Ultra-lean native structlog configuration."""

    # Core metadata added to every single log statement
    processors = [
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    processors.append(structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()))

    structlog.configure(processors=processors)
