# MindDrive 0.5B real L3/L4 evidence

This directory records the compact, Git-tracked index for the complete
MindDrive 0.5B real L3 and L4 evidence. The large capture, AOTInductor artifacts,
held-out inputs, output tensors, and full reports remain in the durable
external archive:

`/home/zhangzimo/Archives/vlaforge-minddrive-0.5b-20260726`

`minddrive_l3.json` was frozen after a clean-worktree L3 rerun from commit
`99f0192da2df4704614b3c4eebe086aaec35c4c5`. The run covers five consecutive
real frames, all 64 executed `sm_86` physical artifacts, 10 transactional
named outputs, and 16 authoritative state slots. Its evidence level is
`real-L3-compiled-held-out-end-to-end`.

The aggregate artifact manifest rejects missing, unexpected, stale,
hash-mismatched, non-contiguous, or mutable hard-linked artifacts. The
held-out validator checks the same manifest and its complete artifact-set
digest before execution.

The first L4 feasibility gates recorded in `minddrive_l3.json` were:

- the backend builds against both LibTorch 2.4 and 2.10 and executes a raw
  PyTorch 2.4 AOTI `.so` through the stable RegionExecutable ABI;
- all 64 verified raw artifacts can be resident together on the RTX 3060;
- an ATen SDPA provider stays within every frozen contract-v3 output/state
  threshold on the five-frame development rerun, so generated C++ does not
  need the Python FlashAttention wrapper.

Those records remain the provenance for the provider decision. The actual
promotion is indexed by `minddrive_l4.json`: commit
`b9372682abd9af484a046014be6305dd2ae1ee45` cleanly builds a verified bundle
containing two static AOTI sequences and six direct artifacts, for 66 physical
artifacts total. The generated runner:

- executes both typed C++ and generic C ABI paths with invalid Python
  environment variables and no `libpython` link;
- returns all 10 outputs bit-exact to the compiled reference and bit-exact
  across APIs;
- records exact revision hit/miss, 128 state commits, 8 output/transaction
  commits, one validation abort, retry, and episode reset.

The durable paths are `l4/bundle` and `l4/evidence`; the full L4 report SHA256
is `20cef99bef75ee60a5d43bd62b731d25451640b1f694be9dfe3c339eba885a6e`.
