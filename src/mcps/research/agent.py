"""Research agent factory using LangGraph."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict

from mcps.research.deep_research import ResearchAgent, create_deep_research_graph

if TYPE_CHECKING:
    from mcps.research.config import ResearchConfig

__all__ = [
    "ProgressReporter",
    "Researcher",
    "ResearchResponse",
    "create_deep_research_graph",
    "create_researcher",
]


class ResearchResponse(TypedDict):
    """Public response contract returned by researcher callables."""

    answer: str
    explanation: str
    sources: list[str]


# Generic progress reporter: (message, current_progress, total|None) → None
# total=None signals indeterminate progress.
ProgressReporter = Callable[[str, float, "float | None"], Awaitable[None]]


class Researcher(Protocol):
    """Callable protocol for research agent implementations."""

    async def __call__(
        self,
        input: str,
        progress: ProgressReporter | None = None,
    ) -> ResearchResponse: ...


_RESEARCH_SYSTEM_PROMPT = (
    "You are a web research assistant. "
    "Use the search tool to find information and the fetch tool to "
    "retrieve web page content. "
    "Always cite your sources."
)


def create_researcher(
    config: ResearchConfig,
    implementation: Literal["react_agent", "deep_research"] = "react_agent",
    **kwargs: Any,
) -> Researcher:
    """Create a LangGraph research agent.

    Args:
        config: Research configuration with model and tool references.
        implementation: Agent implementation variant to use.
        **kwargs: Additional keyword arguments (reserved for future use).

    Returns:
        A compiled LangGraph agent ready to invoke.

    Raises:
        ValueError: If an unsupported implementation is requested.
    """
    match implementation:
        case "deep_research":
            return ResearchAgent(config)
        case _:
            raise ValueError(
                f"Unknown implementation: {implementation!r}. "
                "Supported: 'react_agent', 'deep_research'."
            )
