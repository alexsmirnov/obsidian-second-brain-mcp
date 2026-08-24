# Add support for semantic link types in Obsidian vault

## What
Currently `extract_wikilinks()` (`document_processing.py:34-52`) returns a flat, untyped `list[str]` of wikilink note names, stored as `Chunk.outgoing_links` (`interfaces.py:52`). This loses the relation semantics vault notes already carry via Dataview-style inline fields (`field:: [[Target]]`, in standalone, hidden `(field:: ...)`, or visible `[field:: ...]` form).

When a chunk's body text contains an inline-field-prefixed wikilink, the document-processing pipeline shall extract the field name as the link's type and the wikilink as its target; when a wikilink has no such prefix, the pipeline shall default its type to `reference`. The type is a free-form string taken verbatim from the note text — the server enforces no fixed taxonomy and no reserved-name filtering; those are markdown-authoring conventions for humans/AI curators, out of scope for the indexer. The system shall also build a reverse (incoming) index of these typed links at index time, expose typed outgoing and incoming links through `obsidian_search`, and provide a new MCP tool to walk the typed link graph forward and backward from a given note, filtered by an OR-combination of caller-supplied type strings, up to a caller-supplied depth.

## Scope

### In Scope
- Parsing typed relation links from chunk body text (all 3 inline-field syntax forms), replacing the flat `outgoing_links: list[str]` with typed `{type, target}` pairs
- Defaulting untyped bare `[[wikilinks]]` to type `reference`
- Building and querying a reverse (backlink) index of typed links, computed at index time
- Exposing typed outgoing links and backlinks in `obsidian_search` results
- A new MCP tool that traverses typed links forward and backward from a note, to a caller-supplied depth, filtered by caller-supplied relation types (OR-combined)

### Out of Scope
- Parsing relation fields from YAML frontmatter (vault convention places them in body only)
- Extracting or typing external (non-wikilink) markdown-link targets
- Enforcing, validating, or filtering by a fixed relation-type taxonomy or a reserved-field-name list
- Batch reclassification/retyping of existing untyped links already written in vault note files (a separate vault-content migration script, not server code)
- Query rewriting, tag inference, or other unrelated `SearchAgent`/`SemanticSearchEngine` behavior

## Success Criteria
- [ ] A chunk containing `requires:: [[RAG Basics]]` (any of the 3 syntax forms) produces a link with type `requires` and target `RAG Basics`
- [ ] A bare `[[Some Note]]` with no inline-field prefix produces a link with type `reference`
- [ ] An inline field with an arbitrary/unrecognized name (e.g. `custom-field:: [[X]]`) is extracted with type `custom-field`, not rejected or dropped
- [ ] Typed links round-trip through the vector store (stored and retrieved with type preserved)
- [ ] `obsidian_search` results include typed outgoing links and typed backlinks for a note
- [ ] The new traversal tool returns correct forward and backward hops at depth 1 and depth >1, filtered to only the requested relation type(s)

## Context
- Affected area: `src/mcps/rag/document_processing.py`, `interfaces.py`, `database.py`, `obsidian_vault.py`
- Current behavior: `outgoing_links` is a flat, untyped `list[str]`; no backlink index; no traversal tool exists
- Constraint: relation-field syntax follows Dataview inline-field conventions (field name matches `[a-zA-Z][a-zA-Z0-9_-]*`); LanceDB schema must support the new link shape
- Source: `Obsidian RAG Tool`, `Relations`, `Vault Organization Optimization` notes in the user's knowledge base
