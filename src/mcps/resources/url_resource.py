
from mcps.config import ServerConfig


async def get_resource(encoded_url: str, config: ServerConfig) -> str:
    return f"URL resource: {encoded_url}"