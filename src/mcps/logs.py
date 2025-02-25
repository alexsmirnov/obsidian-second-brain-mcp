import os
import logging
import logging.handlers

def setup_logging():
    """
    Set up logging to write to a file in the user's Library/Logs/Mcps directory.
    """
    log_dir = os.path.expanduser("~/Library/Logs/Mcps")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "mcps.log")


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