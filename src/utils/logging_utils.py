#!/usr/bin/env python3
"""Centralized logging configuration."""

import sys
from typing import Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_logging(
    log_file: Optional[str] = None,
    level: str = "INFO",
) -> None:
    """
    Configure the loguru root logger for the project.

    Parameters
    ----------
    log_file:
        If provided, logs are also written to this path with rotation at 500 MB
        and a 7-day retention policy.  Parent directories must exist.
    level:
        Minimum log level for the console handler (e.g. ``"DEBUG"``, ``"INFO"``).

    Notes
    -----
    The function removes loguru's default handler before adding the project
    handlers, so it is safe to call multiple times (e.g. in distributed
    training where each rank configures logging independently).
    """
    # Remove all existing handlers (including loguru's default stderr handler)
    logger.remove()

    _fmt = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | "
        "{name}:{function}:{line} - {message}"
    )

    # ------------------------------------------------------------------
    # Console handler
    # ------------------------------------------------------------------
    logger.add(
        sys.stderr,
        format=_fmt,
        level=level.upper(),
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # ------------------------------------------------------------------
    # Optional file handler
    # ------------------------------------------------------------------
    if log_file:
        logger.add(
            log_file,
            format=_fmt,
            level=level.upper(),
            rotation="500 MB",
            retention="7 days",
            compression="gz",
            backtrace=True,
            diagnose=False,  # avoid leaking internal vars into potentially shared logs
            enqueue=True,    # thread-safe async writes
        )
        logger.info(f"Logging to file: {log_file}")


def get_logger(name: str):
    """
    Return a loguru logger with *name* bound as a contextual variable.

    Usage::

        log = get_logger(__name__)
        log.info("hello from my_module")

    The returned object supports the full loguru interface
    (``debug``, ``info``, ``warning``, ``error``, ``exception``, etc.).
    """
    return logger.bind(name=name)
