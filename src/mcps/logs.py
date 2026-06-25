import logging
import logging.handlers
import os
import sys


def _resolve_log_path() -> str:
    """Resolve platform-appropriate log file path."""
    if sys.platform == "darwin":
        log_dir = os.path.expanduser("~/Library/Logs/Mcps")
        os.makedirs(log_dir, exist_ok=True)
        return os.path.join(log_dir, "mcps.log")

    # Linux: use /var/log if writable, else fall back to ~/.local/state/mcps
    system_log = "/var/log/mcps.log"
    if os.access("/var/log", os.W_OK):
        return system_log

    log_dir = os.path.expanduser("~/.local/state/mcps")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, "mcps.log")


def setup_logging():
    """
    Set up logging to write to a platform-appropriate log file.
    macOS: ~/Library/Logs/Mcps/mcps.log
    Linux: /var/log/mcps.log (fallback: ~/.local/state/mcps/mcps.log)
    """
    log_file = _resolve_log_path()


    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    # configure output to file and remove console handlers
    # Disable console output by removing default handlers
    try:
        from rich.logging import RichHandler
        # Mcp tries to use rich for logging, if available
        for handler in logging.root.handlers[:]:
            if isinstance(handler, RichHandler):
                logging.root.removeHandler(handler)# Configure logging to write to file
    except ImportError:
        pass
    for handler in logging.root.handlers[:]:
        if isinstance(handler, logging.StreamHandler) :
            logging.root.removeHandler(handler)
    # Configure logging to write to file
    logging.basicConfig(
        handlers=[file_handler],
        level=logging.INFO,  # Capture all log levels
        force=True  # Override any existing logging configuration
    )
    # Reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)