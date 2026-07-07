#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
ENV_FILE="${SCRIPT_DIR}/.env"

if ! docker image inspect vault-mcp >/dev/null 2>&1; then
    echo "Image vault-mcp not found. Building..."
    docker build -t vault-mcp "$SCRIPT_DIR"
else
    IMAGE_CREATED=$(docker inspect --type image --format '{{.Created}}' vault-mcp | cut -d. -f1 | tr 'T' ' ')
    CHANGED_FILES=$(find "$SCRIPT_DIR/src" "$SCRIPT_DIR/Dockerfile" -type f -newermt "$IMAGE_CREATED" -print -quit 2>/dev/null || true)
    
    if [ -n "$CHANGED_FILES" ]; then
        echo "Source files changed. Rebuilding image..."
        docker build -t vault-mcp "$SCRIPT_DIR"
    fi
fi

docker rm -f vault-mcp 2>/dev/null || true

# Wait until Docker completely frees the container name
while docker inspect --type container vault-mcp >/dev/null 2>&1; do
  sleep 0.5
done

docker run -d \
  --restart unless-stopped \
  --name vault-mcp \
  -v /Users/alsmirnov/Obsidian_Vault:/vault \
  --env-file "${ENV_FILE}" \
  -p 8000:8000 \
  vault-mcp
