#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: cross_compile_bundle.sh --bundle DIR --output DIR [--image IMAGE]

Copies a verified TensorRT/SM87 Compile Bundle, rebuilds its generated runner
for JetPack ARM64, updates the binary record in bundle.json, and verifies the
result. TensorRT engines are not executed in the driverless Docker container.
EOF
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
vlaforge_root=$(cd -- "$script_dir/../.." && pwd)
repository_root=$(cd -- "$vlaforge_root/.." && pwd)
bundle=""
output=""
image=vlaforge-orin-backend:jetpack-r36.4.0

while (($#)); do
  case "$1" in
    --bundle)
      bundle=$2
      shift 2
      ;;
    --output)
      output=$2
      shift 2
      ;;
    --image)
      image=$2
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$bundle" || -z "$output" ]]; then
  usage >&2
  exit 2
fi
bundle=$(realpath "$bundle")
output=$(realpath -m "$output")
if [[ "$output/" == "$bundle/"* || "$bundle/" == "$output/"* ]]; then
  echo "input and output bundle directories must not contain each other" >&2
  exit 2
fi
if [[ ! -f "$bundle/bundle.json" ||
      ! -f "$bundle/generated/CMakeLists.txt" ]]; then
  echo "input must be a generated Compile Bundle: $bundle" >&2
  exit 2
fi
if [[ -e "$output" ]] &&
   [[ -n "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "output directory must be absent or empty: $output" >&2
  exit 2
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required" >&2
  exit 2
fi
if [[ ! -e /proc/sys/fs/binfmt_misc/qemu-aarch64 ]]; then
  echo "arm64 binfmt is not registered" >&2
  exit 2
fi
if ! docker image inspect "$image" >/dev/null 2>&1; then
  echo "missing Docker image; run build_backend.sh first: $image" >&2
  exit 2
fi

PYTHONPATH="$vlaforge_root/python${PYTHONPATH:+:$PYTHONPATH}" \
  python3 - "$bundle" <<'PY'
import pathlib
import sys

from vlaforge.deployment import load_bundle_manifest

root = pathlib.Path(sys.argv[1])
manifest = load_bundle_manifest(root / "bundle.json")
manifest.verify_files(root)
for artifact in manifest.region_artifacts:
    if artifact.capability.backend != "tensorrt":
        raise ValueError(
            f"{artifact.region_name}: only TensorRT bundles are supported"
        )
    if artifact.capability.target != "sm_87":
        raise ValueError(
            f"{artifact.region_name}: expected target sm_87, "
            f"got {artifact.capability.target}"
        )
cmake = (root / "generated/CMakeLists.txt").read_text(encoding="utf-8")
if "VLAFORGE_BUILD_TENSORRT_BACKEND ON" not in cmake:
    raise ValueError("generated source does not select the TensorRT backend")
PY

mkdir -p "$output"
cp -a --reflink=auto "$bundle/." "$output/"

docker run --rm --platform linux/arm64 \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "$repository_root":/workspace/edge-fm:ro \
  -v "$output":/workspace/bundle \
  -w /workspace/bundle \
  "$image" \
  bash -lc '
    set -euo pipefail
    cmake -S generated -B .orin-build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=OFF \
      -DVLAFORGE_RUNTIME_ROOT=/workspace/edge-fm/vlaforge \
      -DVLAFORGE_TENSORRT_DRIVERLESS_COMPILE_ONLY=ON \
      -DCUDAToolkit_ROOT=/usr/local/cuda \
      -DCUDA_CUDART=/usr/local/cuda/lib64/libcudart.so
    cmake --build .orin-build --parallel 1
    install -m 0755 .orin-build/vlaforge_generated_runner \
      bin/vlaforge_generated_runner
    cmake --version | sed -n "1p" > .orin-cmake-version
    c++ --version | sed -n "1p" > .orin-cxx-version
    readelf -h bin/vlaforge_generated_runner > .orin-runner-elf
    readelf -d bin/vlaforge_generated_runner > .orin-runner-dynamic
  '

revision=$(git -C "$repository_root" rev-parse HEAD)
dirty=$(git -C "$repository_root" status --short --untracked-files=no)
source_dirty=false
if [[ -n "$dirty" ]]; then
  source_dirty=true
fi
image_id=$(docker image inspect "$image" --format '{{.Id}}')
python3 - "$output" "$revision" "$source_dirty" "$image" "$image_id" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
manifest_path = root / "bundle.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
runner = root / "bin/vlaforge_generated_runner"
payload = runner.read_bytes()
runner_digest = hashlib.sha256(payload).hexdigest()
for record in manifest["binaries"]:
    if record["path"] == "bin/vlaforge_generated_runner":
        record["sha256"] = runner_digest
        record["size_bytes"] = len(payload)
        record["executable"] = True
        break
else:
    raise ValueError("bundle manifest has no generated runner record")

versions = {
    "cmake": (root / ".orin-cmake-version").read_text().strip(),
    "cxx": (root / ".orin-cxx-version").read_text().strip(),
}
manifest["toolchain_versions"] = [
    {"name": name, "version": version}
    for name, version in sorted(versions.items())
]
reproducibility = manifest["reproducibility"]
reproducibility["source_revision"] = sys.argv[2]
reproducibility["source_dirty"] = sys.argv[3] == "true"
reproducibility["build_commands"] = [
    "cmake -S generated -B .orin-build "
    "-DVLAFORGE_RUNTIME_ROOT=<source>/vlaforge "
    "-DVLAFORGE_TENSORRT_DRIVERLESS_COMPILE_ONLY=ON "
    "-DCUDAToolkit_ROOT=/usr/local/cuda",
    "cmake --build .orin-build --parallel 1",
]
environment = {
    item["name"]: item["value"]
    for item in reproducibility.get("environment", [])
}
environment.update(
    {
        "VLAFORGE_ORIN_CROSS_COMPILE_IMAGE": sys.argv[4],
        "VLAFORGE_ORIN_IMAGE_ID": sys.argv[5],
        "VLAFORGE_ORIN_TARGET": "sm_87",
    }
)
reproducibility["environment"] = [
    {"name": name, "value": value}
    for name, value in sorted(environment.items())
]
manifest_path.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
report = {
    "schema": "vlaforge.orin_bundle_cross_compile/1",
    "status": "passed",
    "classification": "driverless JetPack arm64 compile evidence",
    "runner_sha256": runner_digest,
    "runner_size_bytes": len(payload),
    "source_revision": sys.argv[2],
    "source_dirty": sys.argv[3] == "true",
    "image": sys.argv[4],
    "image_id": sys.argv[5],
    "target": "sm_87",
    "real_gpu_execution": False,
}
(root / "metadata/orin_cross_compile.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

PYTHONPATH="$vlaforge_root/python${PYTHONPATH:+:$PYTHONPATH}" \
  python3 - "$output" <<'PY'
import pathlib
import sys

from vlaforge.deployment import load_bundle_manifest

root = pathlib.Path(sys.argv[1])
manifest = load_bundle_manifest(root / "bundle.json")
manifest.verify_files(root)
PY

machine=$(awk -F: '/Machine:/{gsub(/^[ \t]+/,"",$2); print $2}' \
  "$output/.orin-runner-elf")
if [[ "$machine" != "AArch64" ]]; then
  echo "cross-compiled runner is not AArch64: $machine" >&2
  exit 3
fi
if grep -qi "python" "$output/.orin-runner-dynamic"; then
  echo "cross-compiled runner unexpectedly links Python" >&2
  exit 3
fi

rm -rf "$output/.orin-build"
rm -f "$output/.orin-cmake-version" "$output/.orin-cxx-version" \
  "$output/.orin-runner-elf" "$output/.orin-runner-dynamic"

echo "Cross-compiled Orin bundle: $output"
echo "Next: run_bundle_on_orin.sh --dry-run $output"
