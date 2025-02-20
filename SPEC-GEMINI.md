# FastMCP Server Project Specification
This document outlines the specification for a FastMCP server designed to provide prompts, resources, and tools to Language Model (LLM) clients, such as continue.dev.

1. Prompts
Source: Prompts are stored in Markdown files within a dedicated prompts directory on the server.
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
doc: Resource (Library Documentation)

URI Format: doc://<library_name> (e.g., doc://pandas)
Configuration: The server will have a configuration dictionary (library_docs) mapping library names to URLs of llms.txt files. This dictionary will be initially hardcoded in the Configuration class.
Functionality:
Extract the <library_name> from the URI.
Look up the <library_name> in the library_docs dictionary to get the corresponding llms.txt URL.
Fetch the content from the llms.txt URL.
Return the fetched content (assumed to be plain text, ready for LLM use) as the resource.
Error Handling: If the <library_name> is not found in the library_docs dictionary, the server will return an error to the client.
project: Resource (Project Structure and Content)

URI Format: project://<project_name> (e.g., project://my_project)
Configuration: The server will have a configuration dictionary (project_paths) mapping project names to local project root folder paths.
External Tool: "CodeWeawer" - assumed to be a command-line tool named codeweawer available in the system's PATH.
Functionality:
Extract the <project_name> from the URI.
Look up the <project_name> in the project_paths dictionary to get the project root folder path.
Execute the codeweawer command in the shell, passing the project root folder path as an argument (e.g., codeweawer /user/projects/my_project).
Capture the standard output (stdout) from the codeweawer command.
Return the captured stdout (plain text project structure) as the resource.
Error Handling:
If the <project_name> is not found in project_paths, return an error.
If the codeweawer command is not found in the system's PATH, return an error.
If the codeweawer command execution fails (non-zero exit code), return an error. In all error cases, an error response will be returned to the client.
3. Tools
The server will provide the following tools:

web_search Tool (Web Search using Serper)

Tool Name: web_search
Argument: query (string, required) - The search query.
Functionality: Uses the serper API to perform a web search using the provided query.
Output: Returns a plain text summary of the search results as a string. If no results are found, returns an empty string.
Error Handling: If the Serper API call fails, returns an error message string to the client. If no search results are found, returns an empty string.
perplexity_summary_search Tool (Summarized Web Search using Perplexity.io)

Tool Name: perplexity_summary_search
Argument: query (string, required) - The search query.
Functionality: Uses the perplexity.io API to perform a web search and get a summarized response for the query.
Output: Returns the summarized search result as a string. If no summary is available or an error occurs, returns an empty string.
Error Handling: If the Perplexity.io API call fails, returns an error message string to the client. If no summary is available or other issues occur, returns an empty string.
