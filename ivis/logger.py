"""
Logging utility for the IViS pipeline.

This module defines a custom colorized logger for terminal output, used
throughout the IViS codebase to provide informative, timestamped log messages.
"""

import logging
from datetime import datetime

__all__ = ["logger"]

# Define ANSI color codes for log levels and timestamp
COLORS = {
    "info": "\033[92m",
    "warning": "\033[93m",
    "error": "\033[91m",
    "reset": "\033[0m",
    "cyan": "\033[96m",
    "orange": "\033[38;5;214m",
}

class CustomFormatter(logging.Formatter):
    def format(self, record):
        timestamp = datetime.utcnow().replace(microsecond=0).strftime('%Y-%m-%d %H:%M:%S')
        timestamp_colored = f"{COLORS['orange']}{timestamp} UTC{COLORS['reset']}"
        levelname = record.levelname.lower()
        color = COLORS.get(levelname, COLORS["reset"])
        colored_level = f"{color}[{levelname}]{COLORS['reset']}"
        return f"[{timestamp_colored}] {COLORS['cyan']}[{record.name}]{COLORS['reset']} {colored_level} {record.getMessage()}"


logger = logging.getLogger("IViS")
logger.setLevel(logging.DEBUG)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(CustomFormatter())
    logger.addHandler(handler)
