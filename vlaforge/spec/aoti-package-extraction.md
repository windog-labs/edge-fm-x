# Explicit AOTI Package Extraction

The existing loader is unchanged when no extraction option is specified.
The opt-in route supports compiled Linux LibTorch 2.10 packages in archive
format `pt2`, archive version `0`, with the conventional `model` entry. It does
not compile source-only packages and does not create a shared persistent cache.
`model` is the upstream package entry name, not a model-family dispatch rule.

## Native Configuration

`build_artifact_compile_bundle(..., aoti_package_extraction_root="/nas/private")`
records the literal root and extraction implementation hashes in the bundle's
hashed build configuration. It forwards the immutable build default through
`VLAFORGE_AOTI_PACKAGE_EXTRACTION_ROOT`. The default is an empty string. Existing
Region ABI v1/v2 tables do not change. Before load, native callers can override
an individual AOTI Region with `vlaforge_aoti_set_package_extraction_root`.

The root must already exist, be an absolute canonical path, belong to the
current effective user, and have mode 0700. The loader creates only exclusive
0700 `vlaforge-aoti-*` children. It neither creates nor owns the root itself.
No environment variables, installed SDK files, or foreign temporary files are
modified. Different Regions and live Sessions have separate owned children.

Explicit extraction requires the full artifact SHA-256 and size descriptor.
The package stream is hashed before extraction and again, on the same open
file, before loading any library. Generated Session verification remains in
place. Sequence artifacts retain manifest and per-member verification.
Streaming uses the installed LibTorch ZIP reader with a 1 MiB data buffer;
weights and the full archive are not copied into a multi-GB staging allocation.
The metadata and ZIP reader's own bookkeeping are separate bounded-format
overheads, not part of a zero-memory claim. Concurrent mutation of trusted
artifact storage remains forbidden, including deliberate mutate-and-restore.

Record names must be canonical relative paths. Duplicate names, parent/path
escapes, multiple shared libraries, ambiguous blobs, and constants-flattening
collisions are rejected. The whole selected entry is extracted, including the
unchanged proxy JSON, generated source/metadata, CUDA binaries, constants and
external blob. Constants use upstream basename placement. The C++ reader never
interprets ZIP link/permission metadata: every member is written as regular
bytes inside its private child. Source SHA verification, not ZIP metadata, is
the artifact integrity authority. The explicitly bound backend device remains
authoritative; wrapper metadata is retained, not used to silently select a
different device.

The official raw runner initializes the same-stem OSSProxyExecutor and uses
the extracted CUDA-binary directory. External blobs are installed via the
official runner API. CUDA uses single-threaded execution as before. An absent
required proxy produces an official runtime error, never a successful fallback.

Runner destruction precedes extraction cleanup. Partial construction failures
remove owned children. Cleanup uses fd-relative POSIX operations and does not
follow symlinks. Cleanup failure is diagnosed on stderr; destructors cannot
return a checked cleanup status. A killed process or failed cleanup can leave
its owned child behind; no automatic stale-directory scavenger is provided.
External CUDA graphs must be destroyed before their Regions, as required by the
existing Session ownership contract.

## Python Reference Route

`vlaforge.deployment.aoti_load.load_aoti_package` is a separate explicit helper:

```python
with load_aoti_package(
    package, extraction_root=private_root, sha256=expected_sha256,
    size_bytes=expected_size, device="cuda:0",
) as run:
    result = run(*inputs)
    extraction_evidence = run.extraction
```

It uses Python's streaming ZIP API, verifies full package identity twice,
rejects link/special-file entries, preserves per-member hashes, validates device
metadata, and delegates to the official raw runner. Input/output pytrees use
Torch's call spec and keyword ordering. Specialized non-Tensor arguments follow
the ordinary Torch compiled-model ABI; this is not a new dynamic scalar ABI.
The CUDA raw binding supports explicit single-threaded mode; the installed CPU
Python binding does not expose that flag and uses its ordinary runner.

Use a context manager or explicit `close()`. CUDA close drains the device,
releases the runner, then removes the owned child. Failed drain must not delete
its backing directory. Calls and close on one owner are serialized. This loader
neither enforces numerical requirements nor prevents external global setters;
the caller's existing numerical guard remains required. It does not claim
whole-model parity, CUDA replay compatibility, or board execution from an
extraction/CPU test alone.

## NFS Lifecycle Limit

The H20 real CUDA output package exposed a limit not covered by the successful
CPU extraction tests: after the raw runner was destroyed, its ELF library
remained mapped. NFS converted the unlinked file into a busy `.nfs` entry, so
directory cleanup failed. Dropping outputs, GC, CUDA synchronization and two
delayed retries did not release the mapping. This is a failed close operation,
not a successful resource-release or complete pi0 deployment result.

The observed library contains `STB_GNU_UNIQUE` symbols. This is consistent with
GCC's documented effect on DSO unloading; it is not proof that every AOTI
library has that behavior. [GCC code-generation options](https://gcc.gnu.org/onlinedocs/gcc/Code-Gen-Options.html#index-fno-gnu-unique)
explain this binding's interaction with `dlclose`. The existing Torch runner
and DynamicLibrary destructors do request ordinary cleanup, but destructor
order alone cannot prove that the operating system unmapped the library.

The optional materialized path below retains package files as deployment
assets. It does not turn the earlier cleanup failure into a successful close.
Do not silently ignore cleanup errors, force extra `dlclose` calls, remove
foreign files or claim NFS cleanup passed.

## Materialized Deployment Assets

`deployment/aoti_materialized.py` publishes a create-only directory containing
`model.vfaoti` and the complete selected payload. Materialization verifies the
original package stream before and after extraction and never loads code or
recompiles it. The canonical versioned manifest binds source package identity,
target device, library/blob names and every member's SHA-256 and byte count.
These are integrity hashes, not public-key signatures or an untrusted-code
sandbox. Trusted deployment storage must not be modified concurrently.

```python
manifest = materialize_aoti_package(
    package, output_directory, sha256=package_sha, size_bytes=package_size,
)
contract = materialized_region_contract(
    original_contract, manifest, artifact_path="artifacts/region/model.vfaoti",
)
```

`build_artifact_compile_bundle` automatically includes and checks all payload
members for `ArtifactKind.AOTI_MATERIALIZED`; callers provide the manifest as
the artifact source. Model-specific C++ copying/loading code is not required.
The numerical binding preserves original policies and graph identity while
binding the new manifest identity and the original compile-record digest.
This byte-preserving representation change is not a new optimization result.

The ordinary C++ Region descriptor supplies the manifest SHA-256 and size.
Before constructing the raw runner, the loader checks every declared payload,
canonical relative paths, library/blob uniqueness and the explicit device.
No runtime extraction-root option is needed. Python reference execution uses
`load_materialized_aoti(..., sha256=..., size_bytes=..., device="cuda:0")` with
the same runner, proxy, kernel directory and external-blob APIs. The Python
route additionally checks metadata JSON's device field. Linux Torch 2.10.0
is the audited implementation; unsupported versions fail rather than fall back.

Session/runner destruction releases owned runtime resources but does not
unlink immutable deployment files. Removing a deployment directory requires
an external lifecycle decision after its processes have exited; this interface
does not claim DSOs are unmapped at close or provide automatic garbage collection.
CPU tests execute compiled proxy and embedded/blob packages, relocated complete
generated Sessions, actual file-corruption rejection and unchanged default
package loading. CUDA full-model and replay evidence must be recorded separately.

## Primary Sources

- [Torch 2.10 package loader](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/csrc/inductor/aoti_package/model_package_loader.cpp)
- [Torch 2.10 raw runner and proxy initialization](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/csrc/inductor/aoti_runner/model_container_runner.cpp)
- [Torch 2.10 streaming reader](https://github.com/pytorch/pytorch/blob/v2.10.0/caffe2/serialize/inline_container.cc)
