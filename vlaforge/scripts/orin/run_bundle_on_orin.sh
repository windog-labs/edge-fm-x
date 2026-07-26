#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_bundle_on_orin.sh [--dry-run] BUNDLE_DIR [RUNNER_ARGS...]

Validates an arm64 VLAForge bundle on a real Jetson Orin and then executes its
generated no-Python runner. The bundle must already contain real SM87 TensorRT
engines and a generated runner built against their verified hashes.
EOF
}

dry_run=0
if [[ ${1:-} == "--dry-run" ]]; then
  dry_run=1
  shift
fi
if (($# < 1)); then
  usage >&2
  exit 2
fi

bundle=$(realpath "$1")
shift
runner=$bundle/bin/vlaforge_generated_runner
manifest=$bundle/bundle.json
if [[ ! -x "$runner" || ! -f "$manifest" ]]; then
  echo "bundle must contain bin/vlaforge_generated_runner and bundle.json" >&2
  exit 2
fi
if [[ $(uname -m) != "aarch64" ]]; then
  echo "this runner must execute on aarch64 Jetson Orin" >&2
  exit 2
fi

compatible=""
if [[ -r /proc/device-tree/compatible ]]; then
  compatible=$(tr '\0' ',' < /proc/device-tree/compatible)
fi
if [[ "$compatible" != *"tegra234"* ]]; then
  echo "warning: tegra234 was not found in device-tree compatibility" >&2
fi

machine=$(file -b "$runner")
if [[ "$machine" != *"ARM aarch64"* ]]; then
  echo "runner is not an AArch64 ELF: $machine" >&2
  exit 3
fi
linked=$(ldd "$runner")
if grep -q "not found" <<<"$linked"; then
  echo "$linked" >&2
  echo "runner has unresolved shared-library dependencies" >&2
  exit 3
fi
if grep -qi "python" <<<"$linked"; then
  echo "runner unexpectedly links Python" >&2
  exit 3
fi
runtime_libraries=$(ldconfig -p)
if ! grep -q "libnvinfer.so.10" <<<"$runtime_libraries"; then
  echo "TensorRT 10 runtime was not found" >&2
  exit 3
fi

echo "platform=aarch64"
echo "compatible=$compatible"
echo "runner=$runner"
echo "python_linked=false"
echo "shared_libraries=resolved"
if [[ $dry_run -eq 1 ]]; then
  exit 0
fi

cd "$bundle"
exec env \
  PYTHONHOME=/definitely/not/a/python/home \
  PYTHONPATH=/definitely/not/a/python/path \
  "$runner" "$@"
