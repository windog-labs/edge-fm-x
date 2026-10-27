# CogACT Two-Java Window Failure

This is a terminal **incomplete** formal attempt, not an accepted E2E/CDF row.
Only1/10processes completes; no second worker is launched. Formal coverage
remains4/5. The old109partial data and this03attempt stay separate.

## Actual Result

| Item | Verified Evidence |
|---|---|
| First worker |128warmup+1024measured,1152fresh complete calls |
| Complete public outputs |5760tensors independently read; all byte-exact, including typed opaque/integer outputs |
| Primary normalized quality |MSE0,cosine1at unchanged primary gate |
| Native error |Supplemental MSE0; high cosine is not robot success evidence |
| First post-stop MemAvailable |29334088KiB, above25165824KiBstartup gate |
| Minimum running guard |7792488KiB, above4194304KiBfloor |
| First exit MemAvailable snapshot |20916524KiB, below next-worker startup gate |
| Terminal failure |`formal memory admission failed`; exact refused instantaneous value was not persisted |
| Restoration |Only the two authorized Java programs, through original Supervisor |
| Outage |891.990915904s, within original four-hour bound |
| Cleanup |Both owned units terminal/MainPID0/empty compute cgroup; five recorded birth identities terminal |
| Fresh restored Java |1465074/start28963239 and1465114/start28963358, argv/cwd/parent/JAR/config independently checked |

Both whole-cohort model and resource auditors reject this partial cohort.
Twenty-six checks cover the actual output/resource data, partial-cohort refusal,
bad restoration rejection and unchanged admission.5804target file hashes are
independently checked. The failed window actually exercises the unconditional
failure restoration path, not just a synthetic unit test. No application-health
or complete shared16GBSDK/model-resource re-audit claim follows.

[Closure](../../artifacts/orin/20261024-03/failed-window-closure-001.json),
[complete first-worker audit](../../artifacts/orin/20261024-03/partial-model-audit-001.json),
[resource audit](../../artifacts/orin/20261024-03/partial-resource-audit-001.json),
[26checks](../../artifacts/orin/20261024-03/failed-window-tests-003.xml).

## Read-Only Diagnosis Boundary

An actual admission feedback command imports the pinned collector's policy and
calls its real resource require function. Two snapshots return exit2 with the
same refusal, MemAvailable19308964/19180064KiBversus25165824KiB. No model,
service stop or parameter write occurs. `/proc/meminfo`across the original
window shows an8417564KiBavailability reduction, but only109376KiBAnonPages
increase and108612KiBCached reduction. Those counters do not prove a leak or
identify its owner; RSS/PSS and shared CPU/GPU memory are not double summed.

NvMap exposes `enable_page_pools=Y`, raw `pool_size=2009603`, but actual
debugfs clients/pagepool are permission-denied. The raw parameter is not
treated as actual occupied/reclaimable bytes. NVIDIA describes reusable NvMap
pools as a possible general mechanism, not proof of this board's cause.
[NVIDIA's explanation](https://forums.developer.nvidia.com/t/cpu-ram-vs-gpu-ram/127846).
No drop_caches, pool shrink, GPU reset, reboot or SDK update was executed.
Root SSH is unavailable and read-only sudo still requests a password.
[Read-only red feedback](../../artifacts/orin/20261024-04/admission-feedback-002.json).

## New Approved Path

The user subsequently authorizes **different** temporary programs:
`router_manager` and `compotask_service`, each roughly6.9GiBRSS, through the
existing user-owned Supervisor. No permanent startup disable is approved.
Only those two programs are stopped in a fresh05window; Java/native starter,
root device-sharing, other programs and hardware settings stay unchanged.
Two short complete-process probes exit with32482752/32441996KiBavailable,
so a fresh full formal cohort starts without lowering24GiB/4GiBor using root.
This is a capacity alternative, not a proven fix for an SDK/driver leak.
[New scope and evidence](../../artifacts/orin/20261024-05/README.md).
