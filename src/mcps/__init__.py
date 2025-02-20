import os
import sys
import pkgutil
import logging
import logging.handlers
import mcps.server
import mcps.config

# --- Package-level logger setup ---
log_dir = os.path.expanduser("~/Library/Logs/Mcps")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "mcps.log")

logger = logging.getLogger("mcps")
logger.setLevel(logging.DEBUG)

file_handler = logging.handlers.RotatingFileHandler(
    log_file, maxBytes=10 * 1024 * 1024, backupCount=5
)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.propagate = False
# --- End of package-level logger setup ---


def main() -> None:
       # Current working directory
    logger.info(f"Current working directory: {os.getcwd()}")

    # File location
    logger.info(f"File location: {__file__}")

    # Current package name
    logger.info(f"Current package name: {__package__}")


    config = mcps.config.create_config()  # Use the factory method
    server = mcps.server.create_server(config)
    server.start()