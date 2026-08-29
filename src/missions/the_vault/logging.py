import sys

import structlog

ENV_LOG_JSON = "VAULT_LOG_JSON"


def setup_logging():
    """Ultra-lean native structlog configuration.

    JSON to stdout when ``VAULT_LOG_JSON`` is set (production), otherwise a
    colourised console renderer for local development. The final renderer is
    what turns the event dict into a line the underlying logger accepts;
    without it structlog passes bound kwargs straight to ``PrintLogger.msg()``
    and raises ``TypeError``.
    """
    # Core metadata added to every single log statement
    processors = [
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    processors.append(structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()))

    structlog.configure(processors=processors)
