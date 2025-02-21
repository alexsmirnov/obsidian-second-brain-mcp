
from mcps.config import ServerConfig


async def get_resource(project_name: str, config: ServerConfig) -> str:
    return f"project resource: {project_name}"