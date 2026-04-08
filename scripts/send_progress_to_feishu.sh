#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="${CC_CONNECT_PROJECT:-edge-fm-x}"

if ! command -v cc-connect >/dev/null 2>&1; then
  echo "cc-connect not found in PATH" >&2
  exit 1
fi

if [[ $# -gt 0 ]]; then
  cc-connect send -p "${PROJECT_NAME}" -m "$*"
  exit 0
fi

if [[ -t 0 ]]; then
  echo "usage: $0 <message...>  or  echo 'message' | $0" >&2
  exit 1
fi

cc-connect send -p "${PROJECT_NAME}" --stdin
