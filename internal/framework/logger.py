"""Structured logging helpers shared by framework and service code."""

import json
import logging
from typing import Any


def build_logger(name: str, level: str) -> logging.Logger:
    """Create a standard logger used by framework and service code."""

    logger = logging.getLogger(name)
    logger.setLevel(level.upper())
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        logger.addHandler(handler)
    return logger


def log_json(logger: logging.Logger, event: str, payload: dict[str, Any]) -> None:
    """Emit one structured JSON log line for tracing pipeline decisions."""

    logger.info(json.dumps({"event": event, **payload}, default=str))
