<!-- Context: read Goal, Specification, Out of Scope, Research Findings, Implementation Research Findings and Conventions in implementation-plan.md before starting this phase. -->

## Phase 6: Compose deployment and docs [NEW_FEATURE]

No automated tests; configuration only.

- `docker-compose.yaml:1` (NEW):
  - `browser`: `image: h4ckf0r0day/obscura:0.2.3@sha256:475def3ddf1ec513b3d1bc36e8ad15f0d192538cb15f814c77215aa70c418ca2`, `command: ["serve", "--stealth", "--allow-private-network"]`, `ports: ["127.0.0.1:8000:8000"]` (MCP port, published here because of the shared namespace), `restart: unless-stopped`.
  - `mcps`: `build: .`, `network_mode: "service:browser"`, `depends_on: [browser]`, `env_file: .env` (`required: false`), `environment: {BROWSER_CDP_URL: "ws://127.0.0.1:9222", VAULT: ""}`, `restart: unless-stopped`.
- `env.example:55-66` — describe BROWSER_CDP_URL (unauthenticated; fallback `obscura serve` on PATH; research disabled otherwise), keep SCRAPER_PROVIDER block, add `FETCH_MODEL=`, `FETCH_RESTRICTED_DOMAINS=`, `FETCH_CONCURRENCY=2`; remove the "optional extra" line.
- `src/mcps/config.py:57-59` comment — update to new behavior.
- Docs: `docs/config_environment.md:213-237` (new env, routing), `docs/dependencies_libraries.md:82-83` (crawl4ai mandatory), `docs/packages_modules.md:122-135` (filtering.py, removed reddit/wikipedia), `docs/architecture_overview.md:155` and `docs/deployment_infrastructure.md:154` (conditional `web_research`, Compose).

### VERIFY_GREEN
Full offline suite `test.sh "$(pwd)/tests"`. **Manual check (Docker host):** `docker compose config --quiet`, `docker compose up -d --build`, MCP `list_tools` includes `web_research`; `docker compose stop browser` then restart `mcps` → `web_research` absent.
