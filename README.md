
# Model Context Protocol ( MCP ) Python server to use with continue.dev
MCP server that exposes a customizable prompt templates, resources, and tools
It uses FastMCP to run as server application.

Dependencies, build, and run managed by uv tool.

## Provided functionality
### prompts
prompts created from markdown files in `prompts` folder. 
Additional content can be added by templating, by variable names in {{variable}} format
Initial list of prompts:
- review code created by another llm
- check code for readability, confirm with *Clean Code* rules
- Use a conversational LLM to hone in on an idea
- wrap out at the end of the brainstorm to save it as `spec.md` file
- test driven development, to create tests from spec
- Draft a detailed, step-by-step blueprint for building project from spec

### resources
**NOTE: continue does not understand templates, so resource name should contain all information**
**resouce name left as is in prompt, so it should not confuse llm**
- extract url content as markdown
- full documentation about libraries, preferable from llms-full.txt
- complete project structure and content, created by `CodeWeawer` or `Repomix`

### tools
- web search, using `serper` or  
- web search results with summary, by `perplexity.io`
- find missed tests
- run unit tests and collect errors