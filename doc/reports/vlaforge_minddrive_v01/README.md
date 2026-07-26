# MindDrive 0.5B real L3 evidence

This directory records the compact, Git-tracked index for the complete
MindDrive 0.5B real L3 evidence. The large capture, AOTInductor artifacts,
held-out inputs, output tensors, and full reports remain in the durable
external archive:

`/home/zhangzimo/Archives/vlaforge-minddrive-0.5b-20260726`

`minddrive_l3.json` was frozen after a clean-worktree rerun from commit
`99f0192da2df4704614b3c4eebe086aaec35c4c5`. The run covers five consecutive
real frames, all 64 executed `sm_86` physical artifacts, 10 transactional
named outputs, and 16 authoritative state slots. Its evidence level is
`real-L3-compiled-held-out-end-to-end`; it is not generated no-Python C++ L4
evidence.

The aggregate artifact manifest rejects missing, unexpected, stale,
hash-mismatched, non-contiguous, or mutable hard-linked artifacts. The
held-out validator checks the same manifest and its complete artifact-set
digest before execution.

The next promotion gate is a verified Compile Bundle and generated no-Python
C++ Session that executes the same physical artifacts and compiled CUDA
attention provider without Python orchestration.
