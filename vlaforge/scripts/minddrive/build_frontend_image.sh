#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <minddrive-source-root> [image-tag]" >&2
  exit 2
fi

minddrive_source_root=$(realpath "$1")
image_tag=${2:-vlaforge-minddrive:torch2.4.1-cu118-sm86}
expected_revision=1a4085dab1c20895a0c8d2b67b4f8e65712fa8de
actual_revision=$(git -C "$minddrive_source_root" rev-parse HEAD)
repository_root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
dockerfile="$repository_root/vlaforge/docker/minddrive/Dockerfile"
requirements="$repository_root/vlaforge/docker/minddrive/vlaforge-runtime-requirements.txt"
force_cuda_patch="$repository_root/vlaforge/docker/minddrive/force_cuda_build.patch"

if [[ "$actual_revision" != "$expected_revision" ]]; then
  echo "MindDrive revision mismatch: expected $expected_revision, got $actual_revision" >&2
  exit 3
fi

temporary_context=$(mktemp -d /tmp/vlaforge-minddrive-context.XXXXXX)
cleanup() {
  rm -rf "$temporary_context"
}
trap cleanup EXIT

cp -a "$minddrive_source_root/." "$temporary_context/"
cp "$requirements" "$temporary_context/vlaforge-runtime-requirements.txt"
cp "$force_cuda_patch" "$temporary_context/vlaforge-force-cuda-build.patch"

docker build \
  --file "$dockerfile" \
  --build-arg "MINDDRIVE_REVISION=$expected_revision" \
  --tag "$image_tag" \
  "$temporary_context"

docker run --rm --gpus all "$image_tag" python - <<'PY'
import torch

assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
print("compute_capability", torch.cuda.get_device_capability(0))
PY
