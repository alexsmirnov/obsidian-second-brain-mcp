
from mcps.config import ServerConfig


def get_resource(library_name: str, config: ServerConfig) -> str:
    return f"docs resource: {library_name}"