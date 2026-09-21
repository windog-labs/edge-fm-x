# H100 pi0 Dual-Output Native Session

2026-09-08 21:10 UTC. The first H100 physical/double-output no-Python
validation is now closed for pi0. The previous H100 capture was legacy/1 and
could not be extended in place, so the H100 pi0 chain was rebuilt with the
current adapter as a clean v2 capture.

## What Was Done

1. Generated a fresh H100 v2 pi0 capture from the same official frame0 input
   and checkpoint, with `processor_config` and enforceable `/2` numerical
   context.
2. Ran independent saved-only reload and validated the capture report.
3. Recompiled all four real Regions through the public AOTI path
   (`aten-preserving`, sm_90) and passed the complete AOTI audit.
4. Rebuilt the no-Python native normalized Session from that v2 capture and
   ran it successfully.
5. Compiled and validated the native output-stage processor.
6. Attached the output stage to the normalized Session and produced a
   dual-output no-Python bundle through `openpi_output_bundle`.
7. Executed the dual-output bundle with owner-registration monitoring. Three
   complete normalized F32 and native F64 chunks are byte-exact to both the
   official H100 references and the direct full-IR Python outputs. Process
   maps contain no Python.

## Evidence

The small evidence tree is archived under
`artifacts/edgefm-vla-goal/20260906-044509/h100-pi05-aten-recovery-20260909/pi0-h100-dual-output-20260908-evidence/`.

- `verify.json` SHA `e71897a41942ad77dde04ad2860e3dc615de74d79eeb0cbb6736a2b8b5540efa`
- `build.json` SHA `d459b323519d8ad5f9368d3c3e4f15fff29fc87202e7b537e737c9915c78396b`
- `direct.json` SHA `09de9ec630aec5dfc91c1d1cbcd4f87767b52987342b7a5785d5e48c5002a3f2`
- `prepare.json` SHA `e644dd5da077d8cda86fedd3b1e8177fea8063d87b5bc7828e997aafb83468c2`
- Native normalized raw SHA `d91ae5db7a1b3063429a69627c6460005b2c65e2728d822fd4a0c0dbfcb7cca0`
- Native physical F64 raw SHA `0cdb72e592be6c29522082b15d712145b05db4494ed20262f1d8ed19a4e6c43b`
- Upstream v2 capture validated report SHA
  `9f22db202ce34e668ec943f65db441fa84e63db8c651de92461a79ff48df3e5f`
- AOTI audit SHA `9f8977694c186e6352d948774386dc5e2fdacb6a965610ca695704d524e52c06`
- Output processor report SHA
  `4e72416f06bd7bbfc2e24e29421fa08d32a391abf747f6a019dba0453088153f`
- Native normalized v2 Session report SHA
  `edd5fb0159264e8886091abc5af1a0de72a6a8de98c251e6f271af5827ef6e97`

## Scope

This is a three-call diagnostic native dual-output validation on one real
frame, not a formal multi-frame replay/CDF and not Orin/BPU evidence.
`full_paper_acceptance=false`.
