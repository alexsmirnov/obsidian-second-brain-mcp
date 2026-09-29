# Goal
Use crawl4ai and obscura browser to fetch web pages.
Reduce number of failures recorded by `tests/web_research_evaluation.py` logged in "tmp/evaluation-*.log"
Reduce size of successful fetch by filtering content related to query
## What
While mcp server initialized, when web_research lifecycle invoked, connect crawl4ai to running browser if "BROWSER_CDP_URL" env defined, with Bearer authorization if "OBSCURA_CDP_TOKEN" present. If connection to "BROWSER_CDP_URL" failed, try to start `obscura serve --stealth --allow-private-network`. If none succesfull, disable web_research mcp tool.
While fetch URL content, when content loaded, the Fetch tool shall return only content related to query.
While fetch URL content, when content loaded, the Fetch tool shall preserve links from web page.
While fetch URL content, when known domain requested ( github, archiv), the Fetch tool shall use specialized fetch. No fallback for known domains
While fetch URL content, when known restricted domain requested ( configurable list ), the Fetch tool shall return empty result.
While fetch URL content, when known restricted domain requested ( configurable list ), the Fetch tool shall return empty result.
While fetch URL content, when any other domain requested, the Fetch tool shall return crawl4ai result filtered by query. It shall use LLM filete if "FETCH_MODEL" defined, BM25 otherwise
While fetch URL content, when crawl4ai failed to fetch, the Fetch tool shall return content from commertial provider related to query.
While fetch URL content, when multiply targets requested, the Fetch tool shall limit parrallel load to configured value.

## Scope
### In Scope
- fetch tools for web research
- add "query" parameter to fetch to filter related content. tool signature `async def fetch(url: str, query: str|None)`
- connect crawl4ai to running browser
- Docker deployment as "docker compose" fleet, one container with obscura browser, one with mcp server
- evaluation of fetch tool performance compared to baseline from web_research_evaluation log
### Out of Scope
- Obsidian RAG
## Success Criteria
- [ ]
## Context
