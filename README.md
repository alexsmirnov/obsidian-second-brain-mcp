
# Model Context Protocol ( MCP ) Python server
MCP server that exposes a customizable prompt templates, resources, and tools
It uses FastMCP python library to run as server application.

Dependencies, build, and run managed by uv tool.

## Provided functionality
### prompts
prompts created from markdown files in `prompts` folder. 
Additional content can be added by templating, by variable names in {{variable}} format
Initial list of prompts:
- review code created by another llm
- check code for readability, confirm with *Clean Code* rules
- Improve prompt provided as {{input}} parameter

### resources
- complete project structure and content, created by `CodeWeawer` or `Repomix`

### tools
- web search, using `serper` and `tivily` APIs. One parameter `query`
- deep research, by `perplexity.io` , gemini or other tools. First parameter `query`, second is level of research:  `SIMPLE`, `DETAIL`, and `DEEP`
- Retrivial Augmented Generation search in markdown files. For the query search string, return content that may contain answers. Optional parameter is starting folder
- Manipulate Files in Obsidian Vault, location from `VAULT` environment variable: 
    - get list of files and subfolders in folder
    - get file content by path.
    - rename or move Obsidian note, update [[Wikilinks]] references if needed. 2 parameters: old and new locations
- Task specific agents, backed by their own LLM and with set of tools
  - web application testing, controls browser and perform requested tasks.
  - review proposed code changes