# TorchScript Numerical Compile Binding

Update at 2026-09-08 15:08 UTC: the fresh numerical-bound candidate's formal
five-process-per-policy campaign, fixed audit-only recovery, local retrieval
and CDF are complete. See
`doc/reports/cogact_numerical_formal_20260908.md` for the table and SHA chain.
The provider-bound numbers do not replace or inherit the earlier unbound run.

2026-09-08 12:22 UTC. Shared compilation, a new real CogACT candidate native
pilot and local evidence verification are complete. New formal timing, original
model configuration provenance, autonomous RNG and paper acceptance are not.

## Shared Interface

`vlaforge.deployment.torchscript_export.compile_torchscript_region` accepts an
export path, new output path, observed reference context, optional validation
cases and optional target. It measures the input archive, loaded graph and output
archive identities itself, observes the process policy before/after compilation,
and publishes only after checks pass. It does not restore global flags or upgrade
legacy contexts. CPU artifacts cannot be relabeled as CUDA. Failed compilation
and concurrent destination writers do not overwrite an existing output.

The CLI opts in with `--numerical-context` and optionally `--target`. Its default
Region-only path is preserved. Neither API advertises runtime enforcement or
full-model parity merely because compilation passed. Models do not appear in
dispatch in this shared module; runtime bindings continue to use the existing
typed `NumericalCompileRecord`, `RegionNumericalBinding` and generated Session.

The frontend now exposes `exported_graph_digest`, using the unchanged capture
hash algorithm; the previous diagnostic helper name remains available.

## Verification

Evidence root: `artifacts/edgefm-vla-goal/20260906-044509/`.

- Initial five missing-interface tests failed, then passed after implementation.
  A strict JSON conversion failure and a cross-module graph-hash mismatch were
  also reproduced and corrected; all red reports are retained.
- Full CPU regression `resume-cpu-regression-021`: **2113 passed, 83 skipped,
  zero failed**. Source snapshot before/after execution is unchanged. Report SHA:
  `5620c9fcbdd10938e9fda1bbf9454b89e8ba92afe7831353822b8740ada293a1`.
  Optional CUDA/native/dependency tests remain skips, not hardware evidence.
- Separately, `native-numerical-cpu-current-001` passed both opt-in real C++
  numerical-policy tests, including explicit bootstrap, drift rejection,
  multiple Sessions and the legacy control. These are generated CPU fixture
  Sessions, not real-model validation. Report SHA:
  `fafd0b0cdd7fa106f4fda3d7c77d47a4d8460a672a2eecd118003b27b012c37b`.
- Regression 019 failed because the new legacy-context test retained a v2-only
  `torch_version` field. The test now constructs a valid legacy object before
  testing the compiler's refusal; the failing report is not rewritten.

## Real Graph Identity Reproduction

H20-2 run `cogact-numerical-h20-20260908-001` compiled finish/initialize, then
stopped at a driver assertion comparing an in-memory capture graph digest with
the loaded export graph digest. It did not reach full-model or native acceptance.

The real step reproduction `graph-identity-probe-001/data/report.json` has SHA
`b1f8c49f2aec8fa986bdb2695217ffd988544209fc7f3981eb173258a710990e`.
It re-captures and serializes the pinned 356736978-byte step export:

- Original export SHA: `fb3eea630be14758111f65c1e7ea41522f0e3a1023a7ec5f7d990529716c10b2`.
- Capture/re-capture graph: `d1d6e1f43ff197e0996a92e7cc74eeb1c908ee2e08041559566a480ee932556f`.
- Source/reloaded graph: `a28bafcf8f82b8a6a55d9d21a5f11fe841ad650af8e4f432022db36440c23f6f`.
- Both graphs have 480 nodes. Serialization renames `unbind` to `unbind_int`
  and `_assert_tensor_metadata` to `_assert_tensor_metadata_default`, with
  corresponding references renamed. Canonical node-index representations match.
- State-dictionary tensors and the supplied complete Region outputs match;
  the original export bytes remain unchanged. This is not a full-model result.

A minimal real Torch CPU `unbind` save/load regression reproduces the naming
change. New artifact identities use the actually compiled, loaded graph digest;
the original capture hash and exact export-file hash remain separate provenance.
Old capture records and contracts are never rewritten to conceal the difference.

## New Native Evidence

Recovery run `cogact-numerical-h20-20260908-002` uses a fresh local source snapshot
and four newly compiled archives. The driver keeps capture and loaded graph
identities separate, checks the pinned export bytes and binds each new artifact
to its actual compile record. No old capture or failed run was rewritten.

All 16 inputs produce five complete outputs matching the archived official-code
public-dependency candidate references byte-for-byte. The new direct report SHA
is `7322bad8a3fb5f0358b8bad5ce0913c2bf98e1b817f7d3b2c0bac9f866225d4f`.
This re-executes the new artifact through full IR, not the official reference
producer. Global Python RNG state remains unchanged; the external random tape
is still an explicit model input.

Fresh off/batch-only/required C++ bundles and native pilots passed on H20-2 GPU1
`GPU-29fb9c6e-2a11-e303-5f26-ad079fa86fa5`. Independent audit validates all four
provider-required numerical bindings, actual configured native workers, replay
counters and all 144 calls / 720 complete outputs / 50832 values. Ports retain
raw F32[16,7], normalized F32[16,7], native F64[16,7], RNG U8[16] and draws I64[1].

- Native audit SHA:
  `f5244f434a509285fca8af04aac8110138da3e4b4536db6ca05cd6926ef1031d`.
- The original pilot pipeline remains **failed** at its last test step because
  the model environment lacked pytest. Its SHA is
  `246658ea61b23242e9200c77356611e7be5105eba79752596491d900dc7bb5b8`.
  Model build, execution and independent audit had already passed.
- Tests were recovered with 487 hash-verified pure-Python support files in an
  isolated `test-support-002` directory, using the unchanged CogACT environment
  for numerical dependencies. The first support copy omitted pytest's `py.py`
  compatibility module; that partial directory and observed failure are kept.
  No model environment, native bundle or model run was changed/restarted.
- `test-recovery-001`: **12 passed, zero failures/errors/skips**. Recovery SHA:
  `f5cf247ec95989146114a572feac85dac268ae10a66dbae0780ab9572247def6`.
- `pilot-retrieval-001`: 548 files / 23080022 bytes, evidence only. Local raw
  audit SHA `48284cbddaa7be5ab77ad23a9c765ec29a606ddb003374dd031ec0c8ae744f0c`.
  All outputs match both references exactly; 423 cosine/norm metadata entries
  differ by only one ULP across CPUs. MSE/max-abs and integer bytes stay exact.
- `pilot-closeout-001.json` SHA:
  `c8430b442226c2a4ca73b320b1ac5fcfadbaa4d06f8473414e68e2b6d33e2868`.

This closes the numerical-provider gap only for this newly compiled, scoped
candidate and its validated workload. Actual model size remains 7.63B, original
Meta configuration remains unverified, autonomous C++ RNG remains absent and
board validation remains deferred. Earlier unbound formal timing does not become
numerically bound retroactively; fresh formal timing for this version is pending.

## Next Execution

Do not rerun `finish_pilot.py` or overwrite its failed receipt. Before formal
timing, verify the native audit and recovery hashes above, confirm the 12-test
XML and check GPU ownership. The prepared directory currently contains pilots,
not formal runs. On the remote host, with `R` set to the run directory:

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  /xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/envs/cogact-cu128-py311/bin/python \
  "$R/source/vlaforge/tools/benchmark_session.py" --stage run --output "$R/prepared"
```

A new formal controller must record these preconditions and subsequent report,
independent audit and retrieval. The completed pilot is not itself a formal CDF.
