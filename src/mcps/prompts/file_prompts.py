import logging
from pathlib import Path
from typing import Dict

from fastmcp import FastMCP

from mcps.config import ServerConfig


def setup_prompts(mcp: FastMCP, config: ServerConfig):
    """
    Dynamically sets up prompts from the prompts directory.

    Args:
        mcp: The FastMCP instance.
        config: The server configuration.
    """
    @mcp.prompt("echo")
    def echo_prompt(text: str, workspaceDir: str) -> str:
        logging.info(f"Echo prompt called with text: {text}")
        logging.info(f"Workspace directory: {workspaceDir}")
        return "provide short and concise answer: "+text