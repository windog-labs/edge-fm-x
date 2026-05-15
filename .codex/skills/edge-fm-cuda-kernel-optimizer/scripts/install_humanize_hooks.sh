#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
skill_root="$(cd "$script_dir/.." && pwd)"
runtime_root="$skill_root/vendor/humanize"
installer="$runtime_root/scripts/install-codex-hooks.sh"
codex_config_dir="${CODEX_HOME:-$HOME/.codex}"

if [[ ! -x "$installer" ]]; then
    echo "Error: vendored Humanize hook installer is missing or not executable: $installer" >&2
    exit 1
fi

args=(
    --codex-config-dir "$codex_config_dir"
    --runtime-root "$runtime_root"
)

if [[ "${1:-}" == "--dry-run" ]]; then
    args+=(--dry-run)
    shift
fi

if [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 1
fi

"$installer" "${args[@]}"

cat <<EOF

Humanize Codex hooks now point to:
  $runtime_root

This only installs or updates the Stop hook. It does not start an RLCR loop.
EOF
