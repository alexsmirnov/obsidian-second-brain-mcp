
from fastmcp import FastMCP

from mcps.config import ServerConfig



class Tools:
    """
    base class for MCP tool collections.
    
    Provides common initialization patterns and structure for tool classes
    that register methods with FastMCP instances.
    """
    
    
    def register(self) -> None:
        """
        Register all tools, prompts, and resources with the MCP instance.
        
        This method must be implemented by subclasses to register
        their specific tools with the FastMCP instance.
        """
        pass
    
    async def __aenter__(self) -> 'Tools':
        """
        Async context manager entry point.
        
        Default implementation returns self. Subclasses can override
        to perform initialization tasks.
        """
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Async context manager exit point.
        
        Default implementation does nothing. Subclasses can override
        to perform cleanup tasks.
        """
        pass