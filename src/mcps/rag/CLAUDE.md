# Information retrival from Obsidian.md vault.
## Problem statement
The package purpose is to store and search information from Obsidian.md Vault
### Source Structure
Obsidian vault is file system folder that contains plain text files with Markdown formatting.
Files organized by folders, where each folder contains information related to a broad topic.
Each folder may contain subfolders for more narrow subject, but in the same area as a parent folder.
Files linked by `[[Wikilinks]] , there link text is a target file name without extension and parent folders.
If several files have a same name, link may include parent folders to distinguish them.
Text file may contain `frontmatter` header with file properties in YAML format. Example:
```
---
title: "What is the note about"
description: " One sentence that describes note content"
source: "Url to the original page if note created from it"
tags:
  - area
  - project
  - kind
  - type
---
```
Those are the most important properties than can be used to search information
File text may contain tags in markdown format `#tag` , that marks section content.
The most common markdown text structure is the scintific essay, where one top level header describes content,
second level headers to separate introduction, thesises, conclusion, references.
Third and deeper headers usually marks individual statements inside second level section.
Therefore, the sections separated by second level headers is the atomic chunc of information.
### Search
Search system should be able to find information by file content, title, and description properties ( hybrid search with embeddings vector similarity and full text index ), with optional where conditions by file names ( `%like%` ), and tags ( exact match ). The search result is the text from relevant chunks, with additional metadata:
- source file location
- tags
- frontmatter properties
- outgoing links
## Implementation
1. Information processing done in four steps:
    - crawl the file system to get all markdown files, excluding unrelevant ones by the provided regular expressions
    - parse markdown content, extract text and frontmater properties. Keep `title`, `description`, and a list of `tags`. Also include the file last modification time.
    - split entire file content into chunks by first and second level headers. if chunk is too small, combine it with the next one. If it is too big, split into smaller parts. Too big and too small are configuration parameters for the text length. Extract markdown `#tags` and `[[Wikilinks]]` from chunk into metadata.
    - store chunks in the database.
2. Search done in the steps:
    - parse search request. In the next version, this step may rewrite search query for better match
    - search database by text query with optional restrictions by tags and file names
    - rerank search results to keep only specified result size with the most relevant answers. In the next version, it may repeat database search to refine result
    - format result and return it to caller
3. Information update
   - crawl the file system to get all markdown files with the last modification time
   - get all file locations from the database, whith the last modification time as well
   - compare those two collections. Delete all database records for files that were deleted or updated
   - store information for updated and new files, by the same steps ( starting from markdown processing ) as for initial processing.

### Classes and interfaces
each step of data processing and search implemented as separate class:
- update changed information - not yet implemented
- file system crawler `document_processing.py`
- markdown processor `document_processing.py`
- chunker `document_processing.py`
- database storage `database.py`
- search engine `search.py`
- output processor `search.py`
Abstract definitions and data model classes defined in `interfaces.py` module

### Performance considirations
- total number of markdown documents 2000-3000
- avwerge document size 10kb
- estimated number of chunks 10000
- one query per minute max, acceptable response time 5 seconds
### Security
This is fully controlled environment, no security or permission checks needed

### Libraries used
To verify documentation of libraries, use `Context7` tools

Python version 3.12 . Use all language features of this version, avoid deprecated,
prefer async operations using `asyncio` where possible for better performance.
markdown v3.8 to parse markdown text, python-frontmatter v1.1.0 to extract metadata
lancedb v0.23.0 as vector and data storage
pydantic v2.11.7 for data model classes
Full dependencies tree:
├── fastmcp v2.8.1
│   ├── authlib v1.6.0
│   │   └── cryptography v45.0.4
│   │       └── cffi v1.17.1
│   │           └── pycparser v2.22
│   ├── exceptiongroup v1.3.0
│   │   └── typing-extensions v4.14.0
│   ├── httpx v0.28.1
│   │   ├── anyio v4.9.0
│   │   │   ├── idna v3.10
│   │   │   ├── sniffio v1.3.1
│   │   │   └── typing-extensions v4.14.0
│   │   ├── certifi v2025.6.15
│   │   ├── httpcore v1.0.9
│   │   │   ├── certifi v2025.6.15
│   │   │   └── h11 v0.16.0
│   │   └── idna v3.10
│   ├── mcp v1.9.4
│   │   ├── anyio v4.9.0 (*)
│   │   ├── httpx v0.28.1 (*)
│   │   ├── httpx-sse v0.4.0
│   │   ├── pydantic v2.11.7
│   │   │   ├── annotated-types v0.7.0
│   │   │   ├── pydantic-core v2.33.2
│   │   │   │   └── typing-extensions v4.14.0
│   │   │   ├── typing-extensions v4.14.0
│   │   │   └── typing-inspection v0.4.1
│   │   │       └── typing-extensions v4.14.0
│   │   ├── pydantic-settings v2.9.1
│   │   │   ├── pydantic v2.11.7 (*)
│   │   │   ├── python-dotenv v1.1.0
│   │   │   └── typing-inspection v0.4.1 (*)
│   │   ├── python-multipart v0.0.20
│   │   ├── sse-starlette v2.3.6
│   │   │   └── anyio v4.9.0 (*)
│   │   ├── starlette v0.47.0
│   │   │   └── anyio v4.9.0 (*)
│   │   └── uvicorn v0.34.3
│   │       ├── click v8.2.1
│   │       └── h11 v0.16.0
│   ├── openapi-pydantic v0.5.1
│   │   └── pydantic v2.11.7 (*)
│   ├── python-dotenv v1.1.0
│   ├── rich v14.0.0
│   │   ├── markdown-it-py v3.0.0
│   │   │   └── mdurl v0.1.2
│   │   └── pygments v2.19.1
│   └── typer v0.16.0
│       ├── click v8.2.1
│       ├── rich v14.0.0 (*)
│       ├── shellingham v1.5.4
│       └── typing-extensions v4.14.0
├── lancedb v0.23.0
│   ├── deprecation v2.1.0
│   │   └── packaging v25.0
│   ├── numpy v2.3.0
│   ├── overrides v7.7.0
│   ├── packaging v25.0
│   ├── pyarrow v20.0.0
│   ├── pydantic v2.11.7 (*)
│   └── tqdm v4.67.1
├── markdown v3.8
├── ollama v0.5.1
│   ├── httpx v0.28.1 (*)
│   └── pydantic v2.11.7 (*)
├── python-dotenv v1.1.0
├── python-frontmatter v1.1.0
│   └── pyyaml v6.0.2
├── rank-bm25 v0.2.2
│   └── numpy v2.3.0
├── pytest v8.3.4 (group: dev)
│   ├── iniconfig v2.1.0
│   ├── packaging v25.0
│   └── pluggy v1.6.0
└── pytest-asyncio v0.25.3 (group: dev)
    └── pytest v8.3.4 (*)
