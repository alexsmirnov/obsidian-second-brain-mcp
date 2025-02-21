import os
import logging
import logging.handlers

import mcps.server
import mcps.config

# --- Package-level logger setup ---

# --- End of package-level logger setup ---


def main() -> None:
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
    logger = logging.getLogger("mcps")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.propagate = False
       # Current working directory
    logger.info(f"Current working directory: {os.getcwd()}")

    # File location
    logger.info(f"File location: {__file__}")

    # Current package name
    logger.info(f"Current package name: {__package__}")


    config = mcps.config.create_config()  # Use the factory method
    server = mcps.server.create_server(config)
    # mcp server configures logging in constructor
    # configure output to file and remove console handlers
    # Disable console output by removing default handlers
    try:
        from rich.console import Console
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
        # filename=log_file,
        # filemode='a',  # Append mode
        # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[file_handler],
        level=logging.INFO,  # Capture all log levels
        force=True  # Override any existing logging configuration
    )
    server.start()