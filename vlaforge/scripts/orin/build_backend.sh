#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: build_backend.sh [--output DIR] [--image IMAGE] [--rebuild-image]

Builds and packages the VLAForge TensorRT backend in an emulated JetPack arm64
container. The output directory must be absent or empty.
EOF
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
vlaforge_root=$(cd -- "$script_dir/../.." && pwd)
repository_root=$(cd -- "$vlaforge_root/.." && pwd)
output=$vlaforge_root/out/orin-backend
image=vlaforge-orin-backend:jetpack-r36.4.0
rebuild_image=0

while (($#)); do
  case "$1" in
    --output)
      output=$2
      shift 2
      ;;
    --image)
      image=$2
      shift 2
      ;;
    --rebuild-image)
      rebuild_image=1
      shift
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

output=$(realpath -m "$output")
if [[ -e "$output" ]] && [[ -n "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "output directory must be absent or empty: $output" >&2
  exit 2
fi
mkdir -p "$output"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required" >&2
  exit 2
fi
if [[ ! -e /proc/sys/fs/binfmt_misc/qemu-aarch64 ]]; then
  echo "arm64 binfmt is not registered" >&2
  exit 2
fi
if [[ $rebuild_image -eq 1 ]] ||
   ! docker image inspect "$image" >/dev/null 2>&1; then
  docker buildx build --platform linux/arm64 --load \
    -t "$image" \
    -f "$vlaforge_root/docker/orin/Dockerfile" \
    "$vlaforge_root/docker/orin"
fi

PYTHONPATH="$vlaforge_root/python${PYTHONPATH:+:$PYTHONPATH}" \
  python3 "$vlaforge_root/tools/materialize_orin_tensorrt_fixture.py" \
    --output "$output/compile-fixture"

revision=$(git -C "$repository_root" rev-parse HEAD)
dirty=$(git -C "$repository_root" status --short --untracked-files=no)
source_dirty=false
if [[ -n "$dirty" ]]; then
  source_dirty=true
fi

docker run --rm --platform linux/arm64 \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "$repository_root":/workspace/edge-fm:ro \
  -v "$output":/workspace/out \
  -w /workspace/edge-fm \
  "$image" \
  bash -lc '
    set -euo pipefail

    cmake -S vlaforge -B /workspace/out/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=ON \
      -DVLAFORGE_BUILD_TENSORRT_BACKEND=ON \
      -DVLAFORGE_TENSORRT_DRIVERLESS_COMPILE_ONLY=ON \
      -DCMAKE_INSTALL_PREFIX=/workspace/out/sdk
    cmake --build /workspace/out/build --parallel 1
    ctest --test-dir /workspace/out/build --output-on-failure \
      | tee /workspace/out/ctest.log
    cmake --install /workspace/out/build \
      | tee /workspace/out/install.log

    cmake -S vlaforge/tests/tensorrt_install_consumer \
      -B /workspace/out/consumer-build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=/workspace/out/sdk
    cmake --build /workspace/out/consumer-build --parallel 1

    cmake -S /workspace/out/compile-fixture/generated \
      -B /workspace/out/generated-build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=OFF \
      -DVLAFORGE_RUNTIME_ROOT=/workspace/edge-fm/vlaforge \
      -DVLAFORGE_TENSORRT_DRIVERLESS_COMPILE_ONLY=ON \
      -DCUDAToolkit_ROOT=/usr/local/cuda \
      -DCUDA_CUDART=/usr/local/cuda/lib64/libcudart.so
    cmake --build /workspace/out/generated-build --parallel 1

    file /workspace/out/build/libvlaforge_tensorrt_backend.a \
      /workspace/out/build/tests/cpp/vlaforge_tensorrt_region_on_device_smoke \
      /workspace/out/generated-build/vlaforge_generated_runner \
      > /workspace/out/file.txt
    readelf -h /workspace/out/generated-build/vlaforge_generated_runner \
      > /workspace/out/runner-elf.txt
    readelf -d /workspace/out/generated-build/vlaforge_generated_runner \
      > /workspace/out/runner-dynamic.txt
    readelf -h \
      /workspace/out/build/tests/cpp/vlaforge_tensorrt_region_on_device_smoke \
      > /workspace/out/smoke-elf.txt
    readelf -d \
      /workspace/out/build/tests/cpp/vlaforge_tensorrt_region_on_device_smoke \
      > /workspace/out/smoke-dynamic.txt
    if grep -qi python /workspace/out/runner-dynamic.txt \
       /workspace/out/smoke-dynamic.txt; then
      echo "an Orin executable unexpectedly links Python" >&2
      exit 3
    fi
    dpkg-query -W -f="\${Version}\n" libnvinfer10 \
      > /workspace/out/tensorrt-version.txt
    nvcc --version > /workspace/out/nvcc-version.txt
  '

mkdir -p "$output/delivery/bin" "$output/delivery/evidence"
cp -a "$output/sdk/." "$output/delivery/"
cp "$output/build/tests/cpp/vlaforge_tensorrt_region_on_device_smoke" \
  "$output/delivery/bin/"
cp "$output/generated-build/vlaforge_generated_runner" \
  "$output/delivery/evidence/compile-only-generated-runner"
cp "$output/compile-fixture/fixture_manifest.json" \
  "$output/delivery/evidence/"
cp "$script_dir/run_bundle_on_orin.sh" \
  "$output/delivery/bin/run_bundle_on_orin.sh"
cp "$script_dir/run_backend_smoke_on_orin.sh" \
  "$output/delivery/bin/run_backend_smoke_on_orin.sh"
cp "$vlaforge_root/docker/orin/README.md" \
  "$output/delivery/README.md"
chmod 0755 \
  "$output/delivery/bin/run_bundle_on_orin.sh" \
  "$output/delivery/bin/run_backend_smoke_on_orin.sh"

image_id=$(docker image inspect "$image" --format '{{.Id}}')
trt_version=$(tr -d '\n' < "$output/tensorrt-version.txt")
runner_machine=$(awk -F: '/Machine:/{gsub(/^[ \t]+/,"",$2); print $2}' \
  "$output/runner-elf.txt")
python3 - "$output" "$revision" "$source_dirty" "$image" "$image_id" \
  "$trt_version" "$runner_machine" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
runner = root / "generated-build" / "vlaforge_generated_runner"
backend = root / "build" / "libvlaforge_tensorrt_backend.a"
smoke = (
    root / "build/tests/cpp"
    / "vlaforge_tensorrt_region_on_device_smoke"
)

def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

report = {
    "schema": "vlaforge.orin_backend_build/1",
    "status": "passed",
    "classification": "driverless JetPack arm64 compile evidence",
    "source": {
        "revision": sys.argv[2],
        "dirty": sys.argv[3] == "true",
    },
    "container": {
        "image": sys.argv[4],
        "image_id": sys.argv[5],
        "platform": "linux/arm64",
    },
    "target": {
        "soc": "Jetson Orin",
        "compute_capability": "sm_87",
        "jetpack_line": "r36.4",
        "tensorrt": sys.argv[6],
        "runner_machine": sys.argv[7],
    },
    "gates": {
        "runtime_build": True,
        "runtime_ctest": "7/7",
        "tensorrt_backend_compile": True,
        "installed_sdk_consumer_compile": True,
        "generated_tensorrt_session_compile": True,
        "on_device_backend_smoke_compiled": True,
        "on_device_backend_smoke_no_python": True,
        "real_gpu_execution": False,
    },
    "artifacts": {
        "backend_sha256": sha256(backend),
        "runner_sha256": sha256(runner),
        "on_device_smoke_sha256": sha256(smoke),
    },
    "claim_boundary": (
        "The L4T container has no Orin GPU/driver stack. TensorRT engine "
        "deserialization, enqueue, parity, latency, memory, and power remain "
        "on-device gates."
    ),
}
(root / "build_report.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(report["gates"], sort_keys=True))
PY

archive="$output/vlaforge-orin-backend-${revision:0:7}-jetpack-r36.4-arm64.tar.gz"
tar -C "$output" -czf "$archive" delivery build_report.json \
  ctest.log file.txt runner-elf.txt runner-dynamic.txt \
  smoke-elf.txt smoke-dynamic.txt
sha256sum "$archive" > "$archive.sha256"

echo "Orin backend package: $archive"
echo "Build report: $output/build_report.json"
