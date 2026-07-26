#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
smoke=$script_dir/vlaforge_tensorrt_region_on_device_smoke

if [[ $(uname -m) != "aarch64" ]]; then
  echo "the TensorRT backend smoke must run on aarch64 Jetson Orin" >&2
  exit 2
fi
if [[ ! -x "$smoke" ]]; then
  echo "missing executable: $smoke" >&2
  exit 2
fi

compatible=""
if [[ -r /proc/device-tree/compatible ]]; then
  compatible=$(tr '\0' ',' < /proc/device-tree/compatible)
fi
if [[ "$compatible" != *"tegra234"* ]]; then
  echo "tegra234 was not found in device-tree compatibility" >&2
  exit 3
fi

linked=$(ldd "$smoke")
if grep -q "not found" <<<"$linked"; then
  echo "$linked" >&2
  echo "the TensorRT backend smoke has unresolved dependencies" >&2
  exit 3
fi
if grep -qi "python" <<<"$linked"; then
  echo "the TensorRT backend smoke unexpectedly links Python" >&2
  exit 3
fi

trt_version=unknown
if command -v dpkg-query >/dev/null 2>&1; then
  trt_version=$(dpkg-query -W -f='${Version}' libnvinfer10 2>/dev/null ||
    true)
fi

echo "platform=$(uname -m)"
echo "compatible=$compatible"
echo "tensorrt=$trt_version"
echo "python_linked=false"
echo "test=build identity engine + deserialize + bind + enqueueV3 + verify"

exec env \
  PYTHONHOME=/definitely/not/a/python/home \
  PYTHONPATH=/definitely/not/a/python/path \
  "$smoke"
