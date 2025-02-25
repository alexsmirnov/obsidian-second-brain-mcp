import os
import logging
import logging.handlers

import mcps.server
import mcps.config
from mcps.logs import setup_logging

# --- Package-level logger setup ---

# --- End of package-level logger setup ---


def main() -> None:
    config = mcps.config.create_config()  # Use the factory method
    server = mcps.server.create_server(config)
    # mcp server configures logging in constructor
    # configure output to file and remove console handlers
    # Disable console output by removing default handlers
    setup_logging()
    logger = logging.getLogger("mcps")
    # Current working directory
    logger.info(f"Current working directory: {os.getcwd()}")
    # File location
    logger.info(f"File location: {__file__}")
    # Current package name
    logger.info(f"Current package name: {__package__}")

    server.start()