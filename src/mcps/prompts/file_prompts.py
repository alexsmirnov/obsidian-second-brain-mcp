from pathlib import Path
from typing import Dict

from mcps.config import ServerConfig


def setup_prompts(mcp, config: ServerConfig):
    """
    Dynamically sets up prompts from the prompts directory.

    Args:
        mcp: The FastMCP instance.
        config: The server configuration.
    """
    @mcp.prompt("echo")
    def echo_prompt(text: str) -> str:
        return "provide short and concise answer: "+text