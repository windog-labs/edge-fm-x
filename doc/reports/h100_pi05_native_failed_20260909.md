# H100 pi0.5 Native Attempt Status

2026-09-09 04:00 UTC. Several H100 pi0.5 AOTI/native directories contain
bundles and run directories, but their native audit reports are **not passed**.

- `pi05-aoti-eager-005-native-trace/report.json`: `failed`.
- `pi05-aoti-eager-007-native-trace/report.json`: `failed`.
- `pi05-aoti-eager-010-native-trace/report.json`: `failed-numerics`.

These are real attempts, not successful no-Python H100 validation. They must
not be used to fill the same-platform matrix. The saved Python replay
`pi05-saved-only-replay-003` remains exact but is also not no-Python.

Next H100 pi0.5 work must start from the retained failed receipts, reproduce a
single full-action no-Python call, diagnose the failure, and only then attempt
formal three-policy replay.
