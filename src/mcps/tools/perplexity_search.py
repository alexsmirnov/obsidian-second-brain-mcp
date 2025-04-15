from mcps.config import ServerConfig
import httpx

async def do_search(query: str, config: ServerConfig) -> str:
    """
    Performs a search and returns the results. 
    Args:
        query: The search query.

    Returns:
        The search query string back.
    """
    
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.perplexity_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "Be precise and concise."},
            {"role": "user", "content": query}
        ],
        "max_tokens": 1000,
        "temperature": 0.01,
        "top_p": 0.9,
        "return_related_questions": False,
        "web_search_options": {
           "search_context_size": "medium"
      }
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return format_response_with_citations(response.json())

def format_response_with_citations(response: dict) -> str:
    """
    Formats the response from Perplexity.ai to include citations as a markdown list.

    Args:
        response: The JSON response from Perplexity.ai.

    Returns:
        A formatted string with the content and citations.
    """
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "No content available")
    citations = response.get("citations", [])

    if citations:
        citations_md = "\n".join([f"- {url}" for url in citations])
        return f"{content}\n\n### Citations\n{citations_md}"
    return content