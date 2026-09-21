# Formal Session Campaign Summary

`tools/summarize_session_campaigns.py` produces a scanning table from completed
multi-output Session benchmark archives. The input is a data-only index:

```json
{
  "schema": "vlaforge.session_campaign_index/1",
  "campaigns": [
    {
      "id": "experiment-id",
      "prepared": "/absolute/local/prepared",
      "formal_audit": {
        "path": "/absolute/local/independent-formal-audit.json",
        "sha256": "replace-with-the-actual-audit-file-sha256"
      }
    }
  ]
}
```

```sh
python vlaforge/tools/summarize_session_campaigns.py \
  --manifest /absolute/index.json --output /absolute/new-summary
```

The tool requires a passed formal audit, at least five independent processes and
1,000 measured calls each, the v2 complete-output protocol and bitwise reference
validation. It binds the protocol, frozen manifest and complete tensor contract;
checks original/direct serialized reference hashes; verifies every raw output
chunk; and recalculates each worker, policy aggregate and CDF from raw timings.
It retains exact integer outputs and complete floating storage, including values
outside any active-action subset. Floating metrics include finite/near-zero
handling from the shared comparator. High cosine alone never grants acceptance.

Worker and aggregate timing boundaries must match the protocol. Host-model-tensor
campaigns additionally require a bound complete `host-timing.csv`; the summary
rechecks every segment sum against the main latency CSV and includes the sidecar
hash in its evidence map. This does not add preprocessing to a model-tensor run.

Actual GPU UUID/type/driver and owners come from telemetry. Process-map hashes,
no-Python deployment checks, recorded replay counters and numerical-worker
markers are retained. Marker observation is not a new proof of all remote model
payloads or provider implementation: those remain covered by the separately
bound native execution/artifact audits. The summary does not relocate binaries
or reinterpret checkpoint identity.

Campaigns remain separate even when labels or GPU types match. There is no model
name dispatch, cross-campaign speedup, pooled cross-device CDF, automatic removal
of outliers, board substitution or inference of an absent baseline. Input hashes
are checked again before writing. Existing output directories are not overwritten.

Outputs are `cuda-session-table.csv` and `report.json`. Memory is sampled NVML
device use across the whole worker, not allocator peak. This is a CUDA evidence
summary, not the final paper main table: checkpoint/parameter provenance, locked
precision/scheduler profiles, full-input boundary, vendor baselines and board
results still need their corresponding evidence. Original reference limitations
remain visible, including historical notes that may require a separately scoped
new result rather than rewriting the original protocol.
