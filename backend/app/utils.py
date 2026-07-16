"""
Shared utilities: structured logging setup.
"""

import logging
import sys

from app.config import settings


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger configured with a consistent format.

    Args:
        name: The logger name, typically ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
        logger.propagate = False
    return logger
