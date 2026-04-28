#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: .env 파일이 없습니다 ($ENV_FILE)"
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

cd "$SCRIPT_DIR"
exec python main.py
