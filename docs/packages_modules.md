# Packages and Modules

Internal package structure and dependencies for the MCPS Model Context Protocol server. #packages #modules #python

## Main Package: mcps

Root package containing server core, configuration, and logging.

### [src/mcps/server.py](../src/mcps/server.py) #module
MCP server implementation using FastMCP framework. Composes research and Obsidian lifespans and registers tools conditionally.
**Uses**: config, logs, tools.obsidian_vault, research.agent, research.lifespan
**Used by**: (entry point)

### [src/mcps/config.py](../src/mcps/config.py) #module
Configuration management and environment variable loading.
**Uses**: dotenv, rag.document_processing (default skip patterns)
**Used by**: server, rag.vault, research.config, tools, tests

### [src/mcps/logs.py](../src/mcps/logs.py) #module
Logging configuration and setup.
**Uses**: (standard library only)
**Used by**: server

## Sub-package: mcps.rag

Retrieval-Augmented Generation functionality including document processing, vector storage, search, reranking, and summarization.

### [src/mcps/rag/interfaces.py](../src/mcps/rag/interfaces.py) #module
Abstract interface definitions for all RAG components using ABC and Pydantic models.
**Uses**: pydantic
**Used by**: all rag modules, tests

### [src/mcps/rag/vault.py](../src/mcps/rag/vault.py) #module
High-level vault management orchestrating document processing, storage, search, reranking, and embeddings.
**Uses**: interfaces, document_processing, database, search, embeddings, reranking, llm_reranker, proxy_reranker
**Used by**: tools.obsidian_vault

### [src/mcps/rag/index_scheduler.py](../src/mcps/rag/index_scheduler.py) #module
Runs vault index updates at startup, on a fixed interval, and on request. Concurrent requests share one update.
**Uses**: interfaces
**Used by**: tools.obsidian_vault

### [src/mcps/rag/document_processing.py](../src/mcps/rag/document_processing.py) #module
Document file discovery, markdown processing, and text chunking strategies.
**Uses**: interfaces, frontmatter
**Used by**: vault

### [src/mcps/rag/database.py](../src/mcps/rag/database.py) #module
LanceDB vector store implementation with hybrid search and index management.
**Uses**: interfaces, lancedb, pyarrow
**Used by**: vault 

### [src/mcps/rag/embeddings.py](../src/mcps/rag/embeddings.py) #module
Provider-neutral LangChain embedding adapter.
**Uses**: interfaces, langchain_core
**Used by**: vault

### [src/mcps/rag/search.py](../src/mcps/rag/search.py) #module
Search engine, hypothetical document generation, and result formatting implementations.
**Uses**: interfaces, langchain_core
**Used by**: vault

### [src/mcps/rag/search_agent.py](../src/mcps/rag/search_agent.py) #module
Agentic search with query rewriting and search parameter estimation.
**Uses**: interfaces, langchain_core
**Used by**: (currently unused)

### [src/mcps/rag/reranking.py](../src/mcps/rag/reranking.py) #module
Provider-neutral async LangChain reranking using structured output.
**Uses**: interfaces, pydantic, langchain_core
**Used by**: search

### [src/mcps/rag/llm_reranker.py](../src/mcps/rag/llm_reranker.py) #module
LanceDB reranker that fuses LLM relevance ratings with embedding cosine similarity.
**Uses**: interfaces, lancedb, langchain_core, pyarrow
**Used by**: vault

### [src/mcps/rag/proxy_reranker.py](../src/mcps/rag/proxy_reranker.py) #module
HTTP-based proxy reranker for OpenAI-compatible `/v1/rerank` endpoints with RRF fallback.
**Uses**: lancedb, httpx, pyarrow
**Used by**: vault

### [src/mcps/rag/summarization.py](../src/mcps/rag/summarization.py) #module
Whole-document summary generator using a LangChain chat model.
**Uses**: interfaces, langchain_core
**Used by**: vault

## Sub-package: mcps.tools

MCP tool implementations exposing functionality to AI assistants.

### [src/mcps/tools/obsidian_vault.py](../src/mcps/tools/obsidian_vault.py) #module
Obsidian vault operations: file listing, content retrieval, rename/move, and semantic search. Builds the Obsidian lifespan and registers vault tools conditionally. Search, read, and traversal tools wait for a fresh index before they answer.
**Uses**: config, rag.interfaces, rag.vault, rag.index_scheduler, fastmcp, httpx
**Used by**: server

### [src/mcps/tools/internet_search.py](../src/mcps/tools/internet_search.py) #module
Internet search placeholder stub.
**Uses**: config
**Used by**: (currently unused)

## Sub-package: mcps.research

LangGraph-based web deep research agent.

### [src/mcps/research/agent.py](../src/mcps/research/agent.py) #module
Research agent public interface and factory.
**Uses**: research.deep_research
**Used by**: server, research.lifespan

### [src/mcps/research/deep_research.py](../src/mcps/research/deep_research.py) #module
Iterative deep research StateGraph: query generation, fetching, extraction, reflection, and final answer synthesis.
**Uses**: langgraph, langchain_core, pydantic, research.tools
**Used by**: research.agent

### [src/mcps/research/config.py](../src/mcps/research/config.py) #module
Research configuration factory. `build_research_config` is an async context manager that builds the LangChain chat models and search tool and enters the fetch tool (which owns the browser), yielding `None` when no browser is available.
**Uses**: config, httpx, langchain_core, langchain_openai, langchain_google_genai, pydantic, research.tools, research.tools.fetch
**Used by**: research.lifespan

### [src/mcps/research/lifespan.py](../src/mcps/research/lifespan.py) #module
FastMCP lifespan handler that creates the shared HTTP client, enters `build_research_config` (which owns the browser stack) and builds the researcher.
**Uses**: config, fastmcp, httpx, research.agent, research.config
**Used by**: server

### [src/mcps/research/tools/](../src/mcps/research/tools/__init__.py) #package
Async web search and content fetching. The fetch tool is a chain of small classes behind the `Fetch` and `Filter` protocols, assembled in `fetch.py`. `__init__` re-exports `Fetch`, `FetchResult`, `FetchStatus`, `Filter`, `Search`, `SearchResult`, `create_google_search`, `create_duckduckgo_search`, `create_fetch`.
**Uses**: httpx, lxml, pydantic, markdown, pymupdf, crawl4ai
**Used by**: research.deep_research, research.config

#### Contracts ([models.py](../src/mcps/research/tools/models.py)) #architecture
- `Fetch`: `async (url, query=None) -> FetchResult`. Expected failures are returned, never raised.
- `Filter`: `async (FetchResult, query=None) -> FetchResult`.
- `Search`: `async (query) -> list[SearchResult]`.
- `FetchResult`: frozen dataclass with `url` (always the requested URL), `status`, `mime`, `content`, `http_status`. `ok` is true for `FetchStatus.OK`; `is_retryable()` is true for `TIMEOUT`, `EMPTY`, `UNAVAILABLE`, and `HTTP_ERROR` with 401/403/429.
- `FetchStatus` (`StrEnum`): `OK`, `RESTRICTED`, `HTTP_ERROR`, `TIMEOUT`, `EMPTY`, `UNSUPPORTED`, `UNAVAILABLE` (a fetcher could not run; says nothing about the page), `FILTER_FAILED`.

#### Combinators ([combinators.py](../src/mcps/research/tools/combinators.py)) #architecture
| Class | Result |
|---|---|
| `Filtered(fetch, filter)` | `Fetch`; filters successful results only |
| `Fallback(primary, secondary, when=FetchResult.is_retryable)` | `Fetch`; calls `secondary` once when `when(primary)`; keeps the primary result if the secondary is `UNAVAILABLE` |
| `UrlSelector(routes, default)` | `Fetch`; first route whose URL predicate matches |
| `FilterSelector(routes, default=None)` | `Filter`; first route whose result predicate matches (for example by mime); unchanged result without a match |
| `FilterChain(*filters)` | `Filter`; in order, stops at the first failure |
| `Throttled(fetch, limit)` | `Fetch`; at most `limit` concurrent calls |
| `Blocked()` | `Fetch`; `RESTRICTED` without I/O |
| `Truncate(max_chars)` | `Filter`; MIME-aware safe prefix plus `[Content truncated]` marker outside the cap |

#### Modules
| Module | Responsibility |
|---|---|
| `result.py`, `combinators.py` | Contracts and combinators above |
| `models.py` | `SearchResult` |
| `common.py` | `MIME_HTML`/`MIME_MARKDOWN`/`MIME_PLAIN` constants, `normalize_mime`/`textual_mime` classifiers, the `failure` result builder, `extract_hostname`, and `valid_router_url` |
| `google.py`, `duckduckgo.py` | `Search` factories |
| `default.py` | `HttpFetch`: the single owner of page GETs (shared or owned client, browser-like Chrome headers) and of content-type extraction; HTML stays `text/html`, Markdown stays `text/markdown`, every other supported textual type and raw PDF text is `text/plain` |
| `arxiv.py`, `github.py`, `wikipedia.py` | `ArxivFetch` (HTML, then PDF, then abstract), `GitHubBlobFetch`, `GitHubRepoFetch` (README), `WikipediaFetch` (raw wikitext via `action=raw` for `wikipedia.org` and its subdomains): thin wrappers over an injected `Fetch`; results are retargeted at the requested URL |
| `browser.py` | `BrowserFetch` renders pages on an open crawl4ai crawler; `create_browser_fetch` opens one crawler on `BROWSER_CDP_URL` (yields `None` when unset or unreachable) and closes it on exit, so fetches never reconnect |
| `scrape_do.py`, `bright_data.py` | `ScrapeDoFetch`, `BrightDataFetch`: commercial unblocking fallbacks |
| `html_markdown.py` | `create_markdown_generator` (the crawl4ai generator shared with `browser.py`: pruning filter, links kept), `is_html`, and `HtmlToMarkdown`, the filter converting an HTML result to Markdown |
| `filtering.py` | The `Chunker` (`split` text to fragments, `merge` selected fragments back to one text) and `Ranker` (fragment texts to relevant indices) protocols, the `Fragment(text, start, end)` type (one contiguous source slice addressed by half-open offsets that may overlap other fragments), the `FilterModelError` ranker failure, and `RelevanceFilter`, which splits a textual result, ranks the fragment texts, and merges the selected fragments in document order |
| `markdown_chunker.py` | `MarkdownChunker`: windows that target `max_chars` characters. Each cut moves up to a tenth of that size to the nearest H1-H6 heading or the line after a blank line, and neighbors overlap by `overlap_chars`. A `Fragment` holds the exact source slice, so `merge` rebuilds the selected text from offsets alone |
| `bm25_ranker.py` | `Bm25Ranker` ranks fragments with BM25L and exposes `lexical_scores()`. Construction loads NLTK stopwords without downloads. Imports do not load NLTK data. |
| `fetch.py` | `create_fetch`: the composition root; `build_fetch_tool`: lifespan-friendly context manager that owns the browser via `create_browser_fetch` and assembles the fetch |

#### Composition ([fetch.py](../src/mcps/research/tools/fetch.py)) #architecture
```
generic = UrlSelector([.pdf -> HttpFetch], default = Throttled(browser) or HttpFetch)
generic = Fallback(generic, provider)                     # when a provider is configured
routed  = UrlSelector([restricted -> Blocked, arXiv, GitHub blob, GitHub repo, Wikipedia], default=generic)
fetch   = Filtered(routed, FilterChain(FilterSelector([html -> HtmlToMarkdown]), create_page_filter(config), Truncate))
```
`create_page_filter` walks an ordered registry in `fetch.py` (`_RANKERS`) once at startup and returns a `RelevanceFilter` combining a `MarkdownChunker` with the first ranker whose config predicate holds (`DecisionRanker` when `DECISION_MODEL` and a valid router URL are set, else `Bm25Ranker`). A new ranker is one class plus one registry entry. HTML is converted to Markdown once by `HtmlToMarkdown` before the filter, other textual types pass through, and output size is bounded only by `Truncate`. The arXiv, GitHub and Wikipedia routes sit outside the `Fallback`, so they never escalate.

#### Adding a provider
- **Fetch provider**: write a class with `async __call__(url, query=None, /) -> FetchResult` that returns `failure(url, FetchStatus.UNAVAILABLE)` when the service itself fails and declares its `mime`. Add it as a route in `create_fetch` (site-specific) or build it in `research.tools.fetch._create_provider_fallback` (unblocking fallback).
- **Search provider**: write a factory returning a `Search` and add a branch in `research.config.create_search_tool`.

## Sub-package: mcps.common

Building blocks shared by the research and RAG packages.

### [src/mcps/common/decisions/](../src/mcps/common/decisions/__init__.py) #package
Typed decision calls to the router's TypeSafe System One endpoint, `POST /v1/systemone`. `DecisionClient(http_client, router_url=..., router_key=..., model=...)` borrows an `httpx.AsyncClient` and exposes `async decide(state, questions) -> DecisionResponse`. It makes one request, has no retries and no concurrency control; callers such as `DecisionRanker` decide how to fan out. Every failure (transport, non-2xx status, malformed or unsupported response) raises `DecisionError`, and logs never include the response body or the key. `decision_endpoint` accepts a router URL with or without the `/v1` suffix.
Question and answer models live in `models.py`. Only `Noul` (yes/no probability) is modelled today. To add `choice` or `score`, add the models, widen the `Question` and `Answer` aliases to discriminated unions on `type`, and add a typed accessor on `DecisionResponse`; the client does not change.

## Sub-package: mcps.resources

MCP resource handler placeholders. The handlers are imported but commented out in `server.register()` and are not currently active.

### [src/mcps/resources/project_resource.py](../src/mcps/resources/project_resource.py) #module
Project content resource handler placeholder.
**Uses**: (standard library)
**Used by**: (currently unused)

### [src/mcps/resources/doc_resource.py](../src/mcps/resources/doc_resource.py) #module
Documentation resource handler placeholder.
**Uses**: (standard library)
**Used by**: (currently unused)

### [src/mcps/resources/url_resource.py](../src/mcps/resources/url_resource.py) #module
URL content resource handler placeholder.
**Uses**: (standard library)
**Used by**: (currently unused)

## Sub-package: mcps.prompts

Prompt template management utilities. Prompt registration is currently commented out in `server.register()`.

### [src/mcps/prompts/file_prompts.py](../src/mcps/prompts/file_prompts.py) #module
File-based prompt template loading with variable substitution.
**Uses**: (standard library)
**Used by**: (currently unused)

## Package Dependency Flow

```
Entry Point (server.main)
├── config
│   └── rag.document_processing (default skip patterns)
├── server
│   ├── config
│   ├── logs
│   ├── research.lifespan
│   │   ├── research.agent
│   │   │   └── research.deep_research
│   │   │       └── research.tools
│   │   └── research.config
│   │       └── research.tools
│   └── tools.obsidian_vault (conditional)
│       └── rag.vault
│           ├── interfaces
│           ├── document_processing
│           ├── database
│           ├── search
│           ├── embeddings
│           ├── reranking
│           ├── llm_reranker
│           ├── proxy_reranker
│           └── summarization
└── prompts (currently unused)
```

## Cross-Package Dependencies

Only internal project packages are listed. External library dependencies are documented in [Dependencies and Libraries](dependencies_libraries.md).

### Most Depended Upon
- **rag.interfaces** - Used by all RAG components and tests for interface contracts
- **config** - Used by server, vault, research, and tools for configuration

### Leaf Modules (No Internal Dependencies)
- logs
- rag.interfaces
- resources modules
- prompts.file_prompts
- research.tools
