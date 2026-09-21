# H100 pi0.5 ATen Native Path

2026-09-09 04:30 UTC. H100 pi0.5 native failures are now localized:

- `pi05-aoti-eager-010-native-trace` executes a no-Python Session and exits 0,
  but normalized action differs from the same AOTI Python artifact by
  MSE 2.43e-7 and max-abs 2.81e-3 (`failed-numerics`).
- `pi05-aoti-eager-005-native-trace` failed earlier at CMake build.
- `pi05-aoti-aten-012-public-full-audit` passed full AOTI artifact execution
  with the `aten-preserving` profile, but `no_python_deployment="not-run"`.

The recommended H100 path is therefore to rebuild the native Session from the
already-passing `aten-preserving` artifacts instead of trying to relax the eager
profile tolerance. This avoids rerunning capture/reference and keeps an exact
numeric contract.

Required actions:

1. Transfer/refresh the current VLAForge runtime/source into a fresh H100
   isolated run directory.
2. Use `pi05-aoti-aten-012-public-full-audit` compile reports and the existing
   `pi05-recorded-capture-002-image-layout` reference.
3. Produce one no-Python full-Session output and compare it byte-for-byte to
   the ATen Python artifact before any three-policy replay.

This still requires H100 SDK/env compatibility work; it is a diagnosis plus
actionable path, not a new passed H100 result.
