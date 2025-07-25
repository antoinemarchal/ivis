from . import core
from . import imager
from . import io
from . import pipeline
from . import utils  # if utils is a subpackage

import logging
from datetime import datetime

__version__ = "0.1.0"

# Define ANSI color codes for log levels and timestamp
COLORS = {
    "info": "\033[92m",  # Green
    "warning": "\033[93m",  # Orange/Yellow
    "error": "\033[91m",  # Red
    "reset": "\033[0m",  # Reset color
    "cyan": "\033[96m",  # Cyan for [DECONV]
    "orange": "\033[38;5;214m",  # Orange for timestamp
}

# Custom Formatter with Colored Log Levels
class CustomFormatter(logging.Formatter):
    def format(self, record):
        # Round the timestamp to the nearest second and format it in orange
        timestamp = datetime.utcnow().replace(microsecond=0).strftime('%Y-%m-%d %H:%M:%S')
        timestamp_colored = f"{COLORS['orange']}{timestamp} UTC{COLORS['reset']}"

        # Format the log level color
        levelname = record.levelname.lower()
        color = COLORS.get(levelname, COLORS["reset"])  # Get color for level
        colored_level = f"{color}[{levelname}]{COLORS['reset']}"  # Colorize [level]

        # Return the formatted log message with colored timestamp and [DECONV]
        return f"[{timestamp_colored}] {COLORS['cyan']}[DECONV]{COLORS['reset']} {colored_level} {record.getMessage()}"

# Logger Setup
logger = logging.getLogger("DECONV")
logger.setLevel(logging.DEBUG)  # Adjust as needed

# Prevent duplicate handlers
if not logger.hasHandlers():
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())
    logger.addHandler(console_handler)

# Clean up logging namespace
del logging
