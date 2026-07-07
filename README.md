
# Model Context Protocol ( MCP ) server for Obsidian vault

This is side research and learning project, mostly focused on AI Agents and information retrivial, as well as evaluation of AI tools.

The server allows access to [Obsidian.md](https://obsidian.md/) Vault with search and read tools. Obsidian is the plain text Markdown editor, and keeps all note files in the single folder. The format makes it perfect companion and knowledge storage for AI Agents.

I do use combination of AI coding agent ( Claude Code, Cursor ) with Obsidian.md Vault since 2024, similar to [Andrej Karpathy LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f), but a little bit more complicated with his proposal.
With > 1500 notes, agents often miss important information, so I decided to create search tool that aware of knowledge organization that I use

## Provided functionality
### Web deep research
`web_research` tool is an AI agent similar by the functionality to [Perplaxity.ai](https://www.perplexity.ai/) to answer questions based on the public information in Internet. It's optimised to answer technical or academic questions.
The core loop:
1. generate web search queries
2. fetch content from search results URLs. No attempts to bypass bot protections, but supports wide range of public sources: Arxiv.org articles, Github repositories, Reddit forums, Wikipedia, pdf documents - special cases to extract information from them, like switch to reddit API instead of read web pages, or use github raw format instead of html pages.
3. Use LLM to extract relevant information from fetch results, reduces main agent context
4. Reflection step that analyses result and decides to finish research or repeat loop to fill knowledge gaps
5. Final answer generator, creates short answer, long explanation how it was concluded, and relevant links to support answer

### Evaluation results
Evaluation were performed with small models like Gamini Flash lite or GPT 5.4 nano to save costs, with fraction of questions.
[GAIA dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA) - 60% on 20 questions
[DRACO dataset](https://huggingface.co/datasets/perplexity-ai/draco) - around 40%, mostly because answers are less detailed than expected in evaluation criteria

Evaluation code not in this repository, it is part of internal project for team wide AI Software development tools, created to optimize process for my startup [Jobsflow.ai](https://www.jobsflow.ai). The tool ported from that project.

## Obsidian Vault RAG
`obsidian_search` is hybrid vector + BM25 search engine, mostly classical RAG. It optimized to Vault organization that I use

### Note format assumption
I do keep all notes with the same pattern, and indexing tool uses expected format to split note into chunks and generate additional metadata. The format enforced by special Claude Code skill that used to process all new nortes.
1. Frontmatter properties:
   1. `title` - short sentence what is it about
   2. `description` - 3-4 sentences that describe the note content. Used as a summary similar to [RAPTOR](https://arxiv.org/html/2401.18059v1) combined summary of several chunks
   3. `tags` - I have pre-defined set of tags, that slice notes by 3 dimensions: knowledge area ( ai, programming, finances, ...), note type ( article, tutorial, action item, ...) and narrow subject ( programming language, tool type, activity ). Tool enables retrieve of tags taxonomy and filtering them
2. Note content - all notes follow scientific essay format, each section separated by first or second level header. Semantic chunking split notes by sections. Links between notes use `[[Wikilinks]] format and also extracted and stored in database. It allows graph like navigation
3. Note size - keep them no longer than 200 lines

## Indexing

Tool crawls vault folders by pattern, and extracts all markdown files. Each file parsed to extract frontmatter properties, splitted by headers ( no more than 500 tokens ) and saved to database with meta information. Chunk content and description also stored as vector embeddings. Chunks created without overlaps, positive search results combined with neibhours instead.

Additional **summary** chunk created by LLM from whole note content, to increase chances to get into search results.

Reindexing triggered by vault content change, or explicitly from command line.

## Search

The query passed through LLM to create *Hypotetical abswer* . Query itself used for full text search, and generated answer for vector search. This is similar to [qmd markdown search](https://github.com/tobi/qmd)

Database search results filtered by Reranker API call ( Cohere or Voiage.ai ), or Reciprocal Rank Fusion from LanceDB. RRFReranker fuses results by rank position instead of raw score, so it sidesteps having to make vector and full-text scores comparable.

The second filtering uses LLM to select chunks relevated to query. Result chunks combined with their neihbors ( so if 5th chunk from note selected, search return combination of 4-6th chunks)

### Additional filters
To narrow search, optional parameters:
- `tags` list of tags that must be present in result
- `path` file path pattern
I do have 2 special notes, **Tags.md** with taxonomy, and **Folders.md** that describe vault organization, MCP server provides tool and instructions to read them

### Evaluation
I do use a simple evaluation tool, that performs a query on My Vault snapshot, and counts number of expected words ( precision ), and unwanted words ( recall ). 25 questions total, F1 score ~0.9

## Usage

The server uses only a single LLM API provider. I do have [LiteLLM AI Gateway (LLM Proxy)
](https://docs.litellm.ai/docs/simple_proxy) , but it can work with [Openrouter](https://openrouter.ai/)

Clone repository, create `.env` file from `env.example` , and run
```sh
# Create vault index
uv run --project <local copy> mcps --vault <Vault Folder> --reindex
# run as HTTP MCP Server
uv run --project <local copy> mcps --vault <Vault Folder> --port 1234
```
With http protocol, a single server available to all AI tools. I do use it as shared Knowledge Base and memory across all projects.

## Docker

Build and run the server as a standalone container with streamable HTTP transport:

```sh
# Build image
docker build -t mcps:local .

# Run container (vault and secrets are mounted/injected explicitly)
docker run --rm --env-file .env -e VAULT=/vault -v <host-vault-path>:/vault -p 8000:8000 mcps:local
```

The server is reachable at `http://localhost:8000/mcp`.
