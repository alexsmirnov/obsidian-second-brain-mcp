"""Async research callables for web search and content fetching.

This module exposes callable factories used by ``research.config``.
Each callable accepts a single string argument and performs I/O with an
optionally injected ``httpx.AsyncClient`` shared from FastMCP lifespan.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlparse

import httpx
from lxml import html
from pydantic import BaseModel, Field

__all__ = [
    "SearchResult",
    "create_duckduckgo_search",
    "create_fetch",
    "create_google_search",
]


class SearchResult(BaseModel):
    """Single search result item returned by search callables."""

    url: str = Field(description="Result URL")
    title: str = Field(description="Result title")
    snippet: str = Field(description="Result summary text")


def _google_search_params(
    api_key: str,
    cse_id: str,
    query: str,
) -> dict[str, Any]:
    return {
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "num": 10,
        "safe": "off",
    }


def _parse_google_results(data: dict[str, Any]) -> list[SearchResult]:
    items = data.get("items", [])
    parsed_results: list[SearchResult] = []
    for item in items:
        url = str(item.get("link", "")).strip()
        if not url:
            continue
        parsed_results.append(
            SearchResult(
                url=url,
                title=str(item.get("title", "N/A")),
                snippet=str(item.get("snippet", "N/A")),
            )
        )
    return parsed_results


def create_google_search(
    api_key: str,
    cse_id: str,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> Callable[[str], Awaitable[list[SearchResult]]]:
    """Create an async Google CSE search callable."""

    async def search(query: str) -> list[SearchResult]:
        if not api_key or not cse_id:
            return []
        try:
            if http_client is not None:
                response = await http_client.get(
                    "https://www.googleapis.com/customsearch/v1",
                    params=_google_search_params(api_key, cse_id, query),
                )
                response.raise_for_status()
                data = response.json()
            else:
                async with httpx.AsyncClient(timeout=30.0) as owned:
                    response = await owned.get(
                        "https://www.googleapis.com/customsearch/v1",
                        params=_google_search_params(api_key, cse_id, query),
                    )
                    response.raise_for_status()
                    data = response.json()
        except httpx.HTTPError:
            return []
        return _parse_google_results(data)

    return search


# Collected from actual Chrome request headers
CHROME_HEADERS = {
    "accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,image/apng,*/*;"
        "q=0.8,application/signed-exchange;v=b3;q=0.7"
    ),
    "accept-encoding": "gzip, deflate",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "no-cache",
    "dnt": "1",
    "pragma": "no-cache",
    "priority": "u=0, i",
    "sec-ch-ua": (
        '"Google Chrome";v="141", "Not?A_Brand";v="8", "Chromium";v="141"'
    ),
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "same-origin",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
    "user-agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/141.0.0.0 Safari/537.36"
    ),
}

# DDG POST headers — omit br/zstd so httpx can always decompress the
# response using its built-in gzip/deflate support.  origin + referer
# signal a same-origin form submission to DuckDuckGo.
DDG_HEADERS = {
    "accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "accept-encoding": "gzip, deflate",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "no-cache",
    "origin": "https://html.duckduckgo.com",
    "pragma": "no-cache",
    "referer": "https://html.duckduckgo.com/html/",
    "sec-ch-ua": (
        '"Google Chrome";v="141", "Not?A_Brand";v="8",'
        ' "Chromium";v="141"'
    ),
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "same-origin",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
    "user-agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/141.0.0.0 Safari/537.36"
    ),
}

ERROR_HTTP_4XX = "ERROR: http code 4xx"
ERROR_REQUEST_TIMEOUT = "ERROR: request timeout"
ERROR_UNSUPPORTED_CONTENT = "ERROR: unsupported content"
ERROR_EMPTY_RESPONSE = "ERROR: empty response"


def _duckduckgo_payload(
    query: str,
    *,
    region: str,
    timelimit: str | None,
) -> dict[str, str]:
    payload: dict[str, str] = {"q": query, "b": "", "l": region}
    if timelimit:
        payload["df"] = timelimit
    return payload


def _parse_duckduckgo_results(html_text: str) -> list[SearchResult]:

    tree = html.fromstring(html_text)
    items = tree.xpath("//div[contains(@class, 'body')]")
    parsed_results: list[SearchResult] = []
    for item in items:
        title = " ".join("".join(item.xpath(".//h2//text()")).split())
        hrefs: list[str] = item.xpath("./a/@href")
        href = hrefs[0] if hrefs else ""
        snippet = " ".join("".join(item.xpath("./a//text()")).split())
        if not href or href.startswith("https://duckduckgo.com/y.js?"):
            continue
        parsed_results.append(
            SearchResult(url=href, title=title, snippet=snippet)
        )
    return parsed_results


def create_duckduckgo_search(
    *,
    http_client: httpx.AsyncClient | None = None,
    region: str = "us-en",
    timelimit: str | None = None,
) -> Callable[[str], Awaitable[list[SearchResult]]]:
    """Create an async DuckDuckGo HTML search callable."""

    async def search(query: str) -> list[SearchResult]:
        payload = _duckduckgo_payload(
            query,
            region=region,
            timelimit=timelimit,
        )
        try:
            if http_client is not None:
                response = await http_client.post(
                    "https://html.duckduckgo.com/html/",
                    data=payload,
                    headers=DDG_HEADERS,
                )
            else:
                async with httpx.AsyncClient(timeout=30.0) as owned:
                    response = await owned.post(
                        "https://html.duckduckgo.com/html/",
                        data=payload,
                        headers=DDG_HEADERS,
                    )
            response.raise_for_status()
        except httpx.HTTPError:
            return []
        return _parse_duckduckgo_results(response.text)

    return search


def _truncate_content(content: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + "\n\n[Content truncated]"


def _format_source_output(url: str, content: str, max_chars: int) -> str:
    return _truncate_content(content, max_chars)


def _convert_html_to_markdown(html: str) -> str:
    import html2text

    converter = html2text.HTML2Text()
    converter.ignore_images = True
    converter.body_width = 0
    return converter.handle(html).strip()


def _extract_content(_url: str, html: str) -> str | None:
    content = _convert_html_to_markdown(html)
    if not content:
        return None
    return content


def _extract_hostname(url: str) -> str:
    return urlparse(url).netloc.lower()


def _is_arxiv_url(url: str) -> bool:
    return _extract_hostname(url).endswith("arxiv.org")


def _is_wikipedia_url(url: str) -> bool:
    return ".wikipedia.org" in _extract_hostname(url)


def _is_reddit_url(url: str) -> bool:
    return _extract_hostname(url).endswith("reddit.com")


def _is_github_blob_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc.lower() == "github.com" and "/blob/" in parsed.path


def _is_github_repo_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc.lower() != "github.com":
        return False
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return False
    return parts[2:] == []


def _mediawiki_to_markdown(raw_wikitext: str) -> str:
    content = raw_wikitext
    content = re.sub(r"<ref[^>]*>.*?</ref>", "", content, flags=re.DOTALL)
    content = re.sub(r"<ref[^>]*/>", "", content)
    content = re.sub(r"\{\{[^{}]*\}\}", "", content)

    def convert_heading(match: re.Match[str]) -> str:
        markers = match.group(1)
        level = max(1, min(6, len(markers) - 1))
        return f"{'#' * level} {match.group(2).strip()}"

    content = re.sub(
        r"^(={2,6})\s*(.*?)\s*\1\s*$",
        convert_heading,
        content,
        flags=re.MULTILINE,
    )
    content = re.sub(r"'''(.*?)'''", r"**\1**", content)
    content = re.sub(r"''(.*?)''", r"*\1*", content)
    content = re.sub(
        r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]",
        r"\1",
        content,
    )
    content = re.sub(r"^\*(?!\*)\s?", "- ", content, flags=re.MULTILINE)
    return content.strip()


def _extract_arxiv_id(url: str) -> str | None:
    parsed = urlparse(url)
    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) < 2:
        return None
    if path_parts[0] not in {"abs", "pdf"}:
        return None
    paper_id = path_parts[1]
    if paper_id.endswith(".pdf"):
        paper_id = paper_id[:-4]
    return paper_id or None


def _latex_to_markdown(latex_source: str) -> str:
    content = latex_source
    replacements = {
        r"\\section\{([^{}]+)\}": r"## \1",
        r"\\subsection\{([^{}]+)\}": r"### \1",
        r"\\subsubsection\{([^{}]+)\}": r"#### \1",
        r"\\paragraph\{([^{}]+)\}": r"##### \1",
        r"\\textbf\{([^{}]+)\}": r"**\1**",
        r"\\emph\{([^{}]+)\}": r"*\1*",
        r"\\item": "-",
    }
    for pattern, replacement in replacements.items():
        content = re.sub(pattern, replacement, content)
    content = re.sub(r"\\begin\{itemize\}|\\end\{itemize\}", "", content)
    content = re.sub(r"\\begin\{enumerate\}|\\end\{enumerate\}", "", content)
    command_pattern = (
        r"\\[a-zA-Z*]+(?:\[[^\]]*\])?(?:\{[^{}]*\})?"
    )
    content = re.sub(command_pattern, "", content)
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


async def _request_get(
    url: str,
    *,
    http_client: httpx.AsyncClient | None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    if http_client is not None:
        response = await http_client.get(url, headers=headers)
        response.raise_for_status()
        return response
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as owned:
        response = await owned.get(url, headers=headers)
        response.raise_for_status()
        return response


def _to_error_message(error: Exception) -> str:
    if isinstance(error, httpx.TimeoutException):
        return ERROR_REQUEST_TIMEOUT
    if isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code
        if 400 <= status_code < 500:
            return ERROR_HTTP_4XX
    return ERROR_UNSUPPORTED_CONTENT


def _load_arxiv_source(arxiv_id: str) -> str:
    from arxiv_to_prompt import process_latex_source

    latex_source = process_latex_source(
        arxiv_id,
        keep_comments=False,
        remove_appendix_section=True,
    )
    if not isinstance(latex_source, str):
        raise TypeError("arxiv latex source is not a string")
    return latex_source


def _convert_pdf_to_markdown(pdf_bytes: bytes) -> str:
    import pymupdf

    page_text: list[str] = []
    # Open, clean, and reload the PDF to fix annotation errors
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        # clean=True fixes appearance streams; deflate recompresses objects
        cleaned_bytes = document.tobytes(clean=True, deflate=True)
        
    with pymupdf.open(stream=cleaned_bytes, filetype="pdf") as document:
        for page in document:
            # Strip out Screen annotations entirely if errors persist
            for annot in page.annots():
                if annot.type[0] == 21: # pymupdf.PDF_ANNOT_SCREEN:
                    page.delete_annot(annot)
                    
            raw_text = page.get_text("text")
            if not isinstance(raw_text, str):
                continue
            extracted = raw_text.strip()
            if extracted:
                page_text.append(extracted)
    return "\n\n".join(page_text).strip()


def _normalize_content_type(header_value: str) -> str:
    return header_value.split(";", maxsplit=1)[0].strip().lower()


def _extract_html_response(response: httpx.Response) -> str | None:
    return _extract_content("", response.text)


def _extract_pdf_response(response: httpx.Response) -> str | None:
    if not response.content:
        return None
    markdown = _convert_pdf_to_markdown(response.content)
    return markdown or None


def _extract_plain_text_response(response: httpx.Response) -> str | None:
    content = response.text
    if not content or not content.strip():
        return None
    return content


def _looks_like_pdf(content: bytes) -> bool:
    return content.startswith(b"%PDF-")


def _looks_like_html(content: str) -> bool:
    lowered = content.lstrip().lower()
    html_prefixes = ("<!doctype html", "<html", "<head", "<body")
    return lowered.startswith(html_prefixes)


def _extract_without_content_type(response: httpx.Response) -> str | None:
    if response.content and _looks_like_pdf(response.content):
        return _extract_pdf_response(response)
    if response.text and _looks_like_html(response.text):
        return _extract_html_response(response)
    return _extract_plain_text_response(response)


CONTENT_TYPE_EXTRACTORS: dict[str, Callable[[httpx.Response], str | None]] = {
    "text/html": _extract_html_response,
    "application/xhtml+xml": _extract_html_response,
    "application/pdf": _extract_pdf_response,
    "text/plain": _extract_plain_text_response,
    "text/markdown": _extract_plain_text_response,
}


def _github_blob_to_raw_url(url: str) -> str | None:
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 5 or parts[2] != "blob":
        return None
    owner, repo, _, branch, *file_parts = parts
    if not file_parts:
        return None
    file_path = "/".join(file_parts)
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"


def _github_repo_readme_urls(url: str) -> list[str]:
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return []
    owner, repo = parts[0], parts[1]
    branches = ["main", "master"]
    files = ["README.md", "README.rst", "README.txt", "README"]
    return [
        f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_name}"
        for branch in branches
        for file_name in files
    ]


def _wikipedia_raw_url(url: str) -> str:
    parsed = urlparse(url)
    base_path = parsed.path
    return f"{parsed.scheme}://{parsed.netloc}{base_path}?action=raw"


async def _fetch_wikipedia(
    url: str,
    *,
    http_client: httpx.AsyncClient | None,
    max_chars: int,
) -> str:
    response = await _request_get(
        _wikipedia_raw_url(url),
        http_client=http_client,
        headers=CHROME_HEADERS,
    )
    raw_text = response.text.strip()
    if not raw_text:
        return ERROR_EMPTY_RESPONSE
    markdown = _mediawiki_to_markdown(raw_text)
    if not markdown:
        return ERROR_EMPTY_RESPONSE
    return _format_source_output(url, markdown, max_chars)


async def _fetch_arxiv(
    url: str,
    *,
    max_chars: int,
) -> str:
    arxiv_id = _extract_arxiv_id(url)
    if arxiv_id is None:
        return ERROR_UNSUPPORTED_CONTENT
    try:
        latex_source = _load_arxiv_source(arxiv_id)
    except Exception:
        return ERROR_UNSUPPORTED_CONTENT
    if not latex_source.strip():
        return ERROR_EMPTY_RESPONSE
    markdown = _latex_to_markdown(latex_source)
    if not markdown:
        return ERROR_EMPTY_RESPONSE
    return _format_source_output(url, markdown, max_chars)

async def _fetch_reddit(
    url: str,
    *,
    http_client: httpx.AsyncClient | None,
    max_chars: int,
) -> str:
    json_url = f"{url.rstrip('/')}.json"
    response = await _request_get(
        json_url,
        http_client=http_client,
        headers=CHROME_HEADERS,
    )
    payload = response.json()
    children = payload[0]["data"]["children"]
    if not children:
        return ERROR_EMPTY_RESPONSE
    post_data = children[0]["data"]
    title = str(post_data.get("title", "")).strip()
    selftext = str(post_data.get("selftext", "")).strip()
    if not title and not selftext:
        return ERROR_EMPTY_RESPONSE
    markdown = f"# {title}\n\n{selftext}".strip()
    return _format_source_output(url, markdown, max_chars)


async def _fetch_github_blob(
    url: str,
    *,
    http_client: httpx.AsyncClient | None,
    max_chars: int,
) -> str:
    raw_url = _github_blob_to_raw_url(url)
    if raw_url is None:
        return ERROR_UNSUPPORTED_CONTENT
    response = await _request_get(
        raw_url,
        http_client=http_client,
        headers=CHROME_HEADERS,
    )
    content = response.text.strip()
    if not content:
        return ERROR_EMPTY_RESPONSE
    return _format_source_output(url, content, max_chars)


async def _fetch_github_repo(
    url: str,
    *,
    http_client: httpx.AsyncClient | None,
    max_chars: int,
) -> str:
    readme_urls = _github_repo_readme_urls(url)
    if not readme_urls:
        return ERROR_UNSUPPORTED_CONTENT
    for readme_url in readme_urls:
        try:
            response = await _request_get(
                readme_url,
                http_client=http_client,
                headers=CHROME_HEADERS,
            )
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 404:
                continue
            raise
        content = response.text.strip()
        if content:
            return _format_source_output(url, content, max_chars)
    return ERROR_EMPTY_RESPONSE


async def _fetch_default(
    url: str,
    *,
    http_client: httpx.AsyncClient | None,
    max_chars: int,
) -> str:
    response = await _request_get(
        url,
        http_client=http_client,
        headers=CHROME_HEADERS,
    )
    content_type = _normalize_content_type(
        response.headers.get("content-type", "")
    )
    extractor = CONTENT_TYPE_EXTRACTORS.get(content_type)
    if extractor is None:
        if not content_type:
            try:
                extracted = _extract_without_content_type( response)
            except Exception:
                return ERROR_UNSUPPORTED_CONTENT
            if extracted is None:
                return ERROR_EMPTY_RESPONSE
            return _format_source_output(url, extracted, max_chars)
        return ERROR_UNSUPPORTED_CONTENT
    try:
        extracted = extractor(response)
    except Exception:
        return ERROR_UNSUPPORTED_CONTENT
    if extracted is None:
        return ERROR_EMPTY_RESPONSE
    return _format_source_output(url, extracted, max_chars)


def create_fetch(
    *,
    http_client: httpx.AsyncClient | None = None,
    max_chars: int = 8000,
) -> Callable[[str], Awaitable[str]]:
    """Create an async webpage fetch callable with markdown extraction."""

    async def fetch(url: str) -> str:
        try:
            if _is_arxiv_url(url):
                return await _fetch_arxiv(url, max_chars=max_chars)
            if _is_wikipedia_url(url):
                return await _fetch_wikipedia(
                    url,
                    http_client=http_client,
                    max_chars=max_chars,
                )
            if _is_reddit_url(url):
                return await _fetch_reddit(
                    url,
                    http_client=http_client,
                    max_chars=max_chars,
                )
            if _is_github_blob_url(url):
                return await _fetch_github_blob(
                    url,
                    http_client=http_client,
                    max_chars=max_chars,
                )
            if _is_github_repo_url(url):
                return await _fetch_github_repo(
                    url,
                    http_client=http_client,
                    max_chars=max_chars,
                )
            return await _fetch_default(
                url,
                http_client=http_client,
                max_chars=max_chars,
            )
        except (
            httpx.TimeoutException,
            httpx.HTTPStatusError,
            httpx.HTTPError,
        ) as error:
            return _to_error_message(error)
        except Exception:
            return ERROR_UNSUPPORTED_CONTENT

    return fetch
