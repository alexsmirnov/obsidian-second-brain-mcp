# FastMCP Server Project Specification
This document outlines the specification for a FastMCP server designed to provide prompts, resources, and tools to Language Model (LLM) clients, such as continue.dev.

1. Prompts
Source: Prompts are stored in Markdown files within a dedicated `prompts` directory in the Obsidian Vault
Prompt Identification: Each prompt is identified by its filename (without the .md extension). For example, a file named code_review.md corresponds to a prompt named code_review.
Prompt Templating: Prompt files can contain template variables in the format {{variable}}. These variables are placeholders that will be replaced with values provided by the client when requesting a prompt.
Templating Mechanism: Simple string replacement. The server will receive a dictionary of variable names and values from the client and replace all occurrences of {{variable}} with their corresponding values.
Client Interaction (MCP):
Listing Prompts: Clients can use the MCP listPrompts request to get a list of available prompt names. The server will scan the prompts directory and return a list of filenames (without extensions).
Retrieving Prompts: Clients can use the MCP getPrompt request to retrieve a specific prompt. The request must include:
name: The name of the prompt (filename without extension).
arguments: A dictionary where keys are variable names used in the prompt template, and values are the strings to replace the placeholders.
Server Processing: Upon receiving a getPrompt request, the server will:
Locate the Markdown file corresponding to the requested name in the prompts directory.
Read the content of the Markdown file.
Perform template replacement using the provided arguments dictionary.
Return the processed prompt content as a string within the MCP GetPromptResult response.
2. Resources
The server will provide the following resource types, identified by their URI schemes:

url: Resource (Fetch URL Content as Markdown)

URI Format: url:http://<host>/<page> (e.g., url:http://example.com/page)
Functionality:
Extract the URL from the URI (e.g., http://example.com/page).
Use the external service r.jina.ai to fetch and convert the URL content to Markdown by transforming the URL to https://r.jina.ai/<original_url> and making a request.
If the fetched content is plain text, return it as is.
Return the content (Markdown or plain text) as the resource.
Error Handling: Any errors from the r.jina.ai service or the response content itself will be returned as the resource content to the client.

3. Tools
The server will provide the following tools:

web_search Tool (Web Search using Serper)

Tool Name: web_search
Argument: query (string, required) - The search query.
Functionality: Uses the serper API to perform a web search using the provided query.
Output: Returns a plain text summary of the search results as a string. If no results are found, returns an empty string.
Error Handling: If the Serper API call fails, returns an error message string to the client. If no search results are found, returns an empty string.

deep_research Tool (Summarized Web Search)

Tool Name: deep_research
Argument: query (string, required) - The search query.
Functionality: Uses the perplexity.io API, or/and Gemini search grownding to perform a web search and get a summarized response for the query.
Output: Returns the summarized search result as a string. If no summary is available or an error occurs, returns an empty string.
Error Handling: If the Perplexity.io API call fails, returns an error message string to the client. If no summary is available or other issues occur, returns an empty string.

vault_search Tool ( hybrid search in Obsidian Vault)

Tool Name: vault_search 
Argument: query (string, required) - The search query.
Functionality: Perform hybrid search ( full text and embeddings ) inside Vault
Output: Returns the summarized search result as a string. If no summary is available or an error occurs, returns an empty string.

list_files Tool ( list files in Obsidian Vault)

Tool Name: list_files 
Argument: path (string, required) - The initial path.
Functionality: get files in Vault directory
Output: list of Markdoen files and directories

get_file_content Tool ( read note from Obsidian Vault)

Tool Name: get_file_content
Argument: name (string, required) - full path or only name of requested note.
Functionality: find file in Obsidian Vault by full or short name, return its content
Output: Returns file as a string. If no such note found, return empty string

rename_move_note Tool (moves note in Obsidian Vault, preserve links)

Tool Name: rename_move_note 
Arguments: from , to (string, required) - file name and destination. from can be full or partial name, to a directory of full file path
Functionality: Rename or move note ( as shell `mv` command ), update existing links to renamed note
Output: Return success or failure