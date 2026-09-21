# Local Artifact Deduplication

2026-09-09. User-authorized storage reduction, limited to immutable experiment
payloads under `artifacts/edgefm-vla-goal/20260906-044509`.

- 22 duplicate payloads consolidated with the system `hardlink` utility.
- Released 19,962,724,352 allocated bytes (18.59 GiB).
- Project disk usage decreased from about 136 GiB to 117 GiB; filesystem free
  space increased to about 320 GiB. Filesystem totals include unrelated activity.
- All 185 selected payload paths, sizes and SHA-256 digests match before and
  after. Source, documentation, tracked diff, Git HEAD and index are unchanged.
- No unique model package, input, raw output, report, failure evidence, source
  snapshot or environment was deleted. No remote data or process was changed.
- Files sharing a hardlink must remain immutable. New experiments must create
  new files; in-place edits would affect linked copies. Inode/ctime change,
  and equal-content files can share their retained copy's mtime.

Evidence: `artifacts/storage-dedup-20260909/{plan.json,preview.log,apply.log,report.json}`.
Report SHA-256: `fc8d6a6f98cdf8109058681048110bceaf6ce1db3c40b44c515a84925372f1d3`.
The checked driver is retained as `maintain.py`. It is a one-shot maintenance
record, not a model validation or a paper-completion result.
