# Real OpenVLA generated C++ audit

> **Archived v0.1 evidence.** The real checkpoint result below predates the
> passive Invocation IR v0.2 ABI. It must be reproduced before claiming real
> OpenVLA v0.2 L3/L4.

## Result

Gate G3 passed for the pinned local OpenVLA-7B checkpoint. The generated C++17
runner consumes preprocessed BF16 pixels and integer token/mask tensors, runs
multimodal prefill, six explicit cached decode steps, token extraction,
detokenization, validation, transaction commit, and publication for three
control ticks. The hot path does not start or link Python.

```text
checkpoint revision: 47a0ec7fc4ec123775a391911046cf33cf9ed83f
checkpoint shards: 3, approximately 15 GB BF16
torch: 2.6.0+cu126
deployment device: CPU, 16 threads
persistent StateSlots: none
```

The official BF16 `model.generate`, explicit Python token loop, Semantic IR,
Plan Executor, Python-loaded archive, and generated C++ all produce:

```text
31857, 31864, 31900, 31840, 31860, 31868, 31872
```

The continuous action on all three ticks is:

```text
 0.003146326296469725
 0.002165956051910577
-0.000241243806393714
 0.020934728743983083
 0.000559131567969023
 0.004768682880728764
 0
```

## AOTInductor failure and no-Python fallback

The original exported program was valid in memory but PyTorch 2.6 could not
deserialize its Transformers RoPE higher-order wrappers. Replacing only the
fixed-profile `no_grad/autocast` wrapper with its pure tensor expression was
bit-exact and made the 4,747-node saved program loadable.

AOTInductor then exceeded the 32 GB host memory budget immediately after
loading the 15 GB ExportedProgram and was killed with exit 137. This failure
was reproduced after the serialization issue was removed.

The required no-Python fallback is a production `RegionExecutable` backend
over LibTorch TorchScript, not a Python callback. One 15,085,415,106-byte
archive exposes three entrypoints:

```text
prefill
decode
detokenize
```

The backend parses the generic `archive#entrypoint` only during load, stores a
resolved method, and shares one module instance across all RegionExecutable
objects. The 7B weights therefore reside once. The fixed six decode KV banks
are allocated before `RunTick` and reused across ticks.

Archive SHA-256:

```text
f77f68374187adade017e6f5d9e35ba0d97936f144ca7ae4fc5711b4a4c2eaec
```

## Evidence

```text
generated source digest:
  cefbb5b403dce15ea675d7f2d0b4696256a8b7f4d6dfae0199b9e877ec111e3d
Python archive vs C++ region tensors: 460/460 exact
maximum absolute error: 0
maximum relative error: 0
Semantic vs Plan trace: exact
Semantic vs Plan vs C++ integer trace: 54/54 exact
transaction IDs: 0, 1, 2
state commit events: 0
binary evidence: 1,021,162,520 bytes, exact expected size
observed runner RSS during execution: approximately 17.3 GB
```

The C++ runner was built in a clean directory with `-Werror`, ran with invalid
`PYTHONHOME/PYTHONPATH`, and has no `libpython` dependency. CUDA libraries
appear in `ldd` because this LibTorch distribution is CUDA-enabled, but every
OpenVLA tensor and operation in this audit is on CPU.

## Input ABI

The runner receives three raw, typed tensors rather than invoking a tokenizer,
PIL, or processor in its hot path:

```text
pixel_values:    1x6x224x224 bf16, 602,112 bytes
input_ids:       1x19 i64, 152 bytes
attention_mask:  1x19 i64, 152 bytes
```

Their hashes are recorded by
`vlaforge/tools/materialize_real_openvla_inputs.py`.

## Reproduction

```text
vlaforge/tools/audit_real_openvla_frontend.py
vlaforge/tools/materialize_real_openvla_inputs.py
vlaforge/tools/generate_real_openvla_cpp.py
vlaforge/tools/audit_real_openvla_cpp.py
```

`audit_real_openvla_cpp.py` loads the exact same archive through Python
LibTorch only as an offline reference, compares every prefill/decode output
against the C++ evidence, and separately proves Semantic/Plan/C++ control
trace equality.
