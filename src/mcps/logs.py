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
    os.environ["FASTMCP_ENABLE_RICH_LOGGING"] = "False"
    if os.path.exists('/.dockerenv') or os.environ.get('CONTAINER') == 'true':
        log_handler = logging.StreamHandler(sys.stdout)
    else:
        log_file = _resolve_log_path()
        log_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
    log_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    log_handler.setFormatter(formatter)
    # Configure logging to write to file
    logging.basicConfig(
        handlers=[log_handler],
        level=logging.INFO,  # Capture all log levels
        force=True  # Override any existing logging configuration
    )
    # Reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)