"""
Shared logger factory for Home Brain.

What it is: a thin wrapper around Python's stdlib logging.
What it knows: log format, output stream.
What it doesn't know: domains, adapters, config — nothing application-specific.

Usage: logger = get_logger("surveillance.StreamIngestionService")
"""
import logging
import sys


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger
