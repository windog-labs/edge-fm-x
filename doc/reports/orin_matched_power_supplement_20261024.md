# Orin Same-Engine Matched Power Supplement

## Second-Mode Permission Assessment

After all five primary-mode VLA cohorts pass, the actual non-root
`/usr/sbin/nvpmodel -q --verbose`query reports current MAXN/ID0. The installed
configuration lists MAXN,MODE_15W,MODE_30W,MODE_50W; listing is not validation
that a switch is safe or that those wattages are measured caps. Actual existing
CPU/GPU limits differ from some configuration values and are left untouched.
Query exit0still includes EACCES13for EMC cap, retained as a missing read,
not zero or a fully observed profile.

UID1000cannot write the status file, active CPU max-frequency or GPU
max-frequency paths. The CLI binary is root-owned0755, not setuid, with no
file capability reported. No verified unprivileged switching route exists in
this execution; no sudo/root or mode/clock write is attempted. Therefore no
second-mode performance is claimed. This is the concrete permission boundary
allowed by GoalD4.5, not a declaration that Orin lacks alternate power modes.
All managed programs remain unchanged. MAXN is not renamed60W.
[Actual query/config/permissions](../../artifacts/orin/20261024-06/power-mode-assessment-001.json).

## Complete Representative Result

Terminal verification: 2026-10-23 23:01 UTC. **All15workers and the matched
client-power supplement pass their full scoped audits; the whole Goal and
CogACT remain incomplete.** Native full model/precision/collector/inputs/noise
stay unchanged.17,280total calls include128warmup+1024measured per process;
all34,560complete normalized/native action tensors are independently byte-exact,
MSE0/cosine1, including warmups. The existing frozen source gate is unchanged.

| Mode | Measured | Mean E2E ms | p99 ms | Std ms | Chunks/s | Matched Input-Rail W | Eligible Power Samples |
|---|---:|---:|---:|---:|---:|---:|---:|
| Conventional TRT |5120|132.803564|140.117911|2.094343|7.529918|38.438594|655|
| Shared stream |5120|132.111825|137.898456|1.801593|7.569345|38.552769|650|
| Full-loop replay |5120|112.148095|117.771376|1.713163|8.916781|42.267198|549|

Power is the equal-weight mean of five worker-window client-time-weighted
means. Across-worker population std is0.115188/0.089732/0.083017W respectively,
not a confidence interval. Replay's observed window power is about10%higher,
not lower, while its same-cohort mean runtime ratio is1.184180291x. Allfive
paired replay power deltas are positive. No significance, causal isolated-power
or exact request-energy claim is made; power windows do not exactly coincide
with individual timed calls. The new replay std1.713163ms still misses the
draft1.2ms target. Negative/tail results remain intact.

All2423raw power samples are retained;1854meet the frozen measured-window rules.
The other569are84preflight/between-worker and485warmup/guard/birth/transition
rows, retained in the trace rather than deleted. Per-case eligible counts are
109--132and spans108.063--131.083s; max eligible gap1.02179072s, within the
unchanged80sample/5second gates. Daemon cache/response delay and background
work remain explicit limitations. This approximately40.413minute multi-worker
campaign is **not** a30minute continuous single-worker soak; the old35minute
soak stays separate. Original main-table134.409630/113.139929msand1.187995x
are unchanged, not replaced by these differently observed power-campaign times.

| Sampled Whole-Worker Envelope | Conventional | Shared | Replay |
|---|---:|---:|---:|
| Maximum process VmHWM (KiB) |3155124|3154168|3156308|
| Minimum system MemAvailable (KiB) |21402632|21459104|21380348|
| Highest valid sampled temperature (C) |55.593|55.968|56.218|
| Resource snapshots |621|618|536|

These include initialization/warmup and are not isolated physical GPU peaks,
exhaustive private-SDK allocation or CPU-heap freedom. Shared CPU/GPU physical
memory is not summed. Permission-denied device-owner/sensor entries stay in
raw telemetry and do not certify device exclusivity.

The own unit finishes success/MainPID0; post observes an empty cgroup. All33
tracked monitor/collector/native/tegrastats/post births are terminal and192
actual target paths are independently rehashed, including new captures,
native binaries/engines and pinned helper/SDK inputs. Java942765/942805retain
birth/argv/cwd/original Supervisor parent. No original application stop/start,
expired service grant, SDK or power-mode change is used. Eighteen source/
actual/full-cohort/figure tests pass. The figures retain all2423raw reads and
pass PNG pixel-variance/dimension and PDF signature checks; visual review is
also complete. No original paper PDF is edited.

Fresh post-campaign capacity reads23,491,872--23,494,288KiB(about22.4GiB),
all below CogACT's unchanged25,165,824KiB(24GiB)admission. Visible GPU holders
are the two existing gnome-shell processes, with permission limits retained.
There is no model job left running and no CogACT retry or background stop.

- [Joint complete model/power/source/process closure](../../artifacts/orin/20261023-111/joint-closure-001.json).
- [Original full34560tensor/CDF auditor result](../../artifacts/orin/20261023-111/target-archive-001/formal-001/audit-output/audit.json).
- [Power windows and paired rounds](../../artifacts/orin/20261023-111/power-audit-001.json).
- [Power PNG/PDF render receipt](../../artifacts/orin/20261023-111/power-figures-001/receipt.json).
- [Power figure](../../artifacts/orin/20261023-111/power-figures-001/matched-power.png).
- [Eighteen complete source/actual checks](../../artifacts/orin/20261023-111/complete-cohort-tests-006.xml).
- [Terminal snapshot](../../artifacts/orin/20261023-111/checkpoint-013.json).
- [Post-campaign capacity/application observation](../../artifacts/orin/20261023-111/post-supplement-resources-001.json).

Do not rerun the one-shot launcher or completed archive/closure pipeline.
Read-only host rechecks use `test_matched_power.py` and fresh output paths.
Remaining Goal gates include CogACT's complete cohort, unresolved operator/
memory boundaries and a supported authorized alternate power mode after all
five primary-mode models. Formal model coverage remains4/5.

## Historical Preparation And Live Checkpoints

Counter refresh, 2026-10-23 22:33 UTC: same batch7/15complete workers,
8071measured and1262raw power rows, current ordinary worker still live.
[Checkpoint006](../../artifacts/orin/20261023-111/checkpoint-006.json).
No full-cohort or power acceptance follows from these running counts.

Latest live checkpoint: 2026-10-23 22:28 UTC. Two rounds complete:
6/15workers,6144measured,985raw power rows. Same unit/monitor/collector remain
live, no restart. Fifteen source/actual-window checks pass, including actual
live-archive refusal. First-worker quality remains selected evidence only;
all15workers, complete power and terminal closure remain pending.

Checkpoint: 2026-10-23 22:20 UTC. **Sampling is live, not accepted.** Three of
15 independent workers complete; 3,072 measured requests are recorded. The
first ordinary worker independently checks all 2,304 complete action tensors
byte-exact, including 128 warmups and 1,024 measurements. Normalized MSE0/
cosine1 and native supplemental error0. This is selected-worker evidence,
not full-cohort fidelity, power acceptance or increased five-model coverage.

## Frozen Workload

This required D4 supplement uses the same seven previously qualified SmolVLA
engines and original native workers, collector, real16observations/noise and
precision. Ordinary TRT, shared-stream and full-loop replay each retain
five independent serial processes,128warmup+1024measured,batch1, with rotating
mode order. The full prefix and alltenaction iterations execute freshly for
every request; no precomputed context or reduced solver is introduced.
The original source gate and official H20 supplemental failure stay unchanged.

The original collector performs its actual engine/runner/data checks before
inference; the new observer neither modifies its files nor replaces its timing
boundary. The old formal main-table times are not substituted by this new
Java-on power campaign. A full original 15-worker output/CDF audit, separate
power-window audit and terminal-process/source closure are still mandatory.

## Power Measurement Boundary

The existing jtop4.2.3daemon and hash-verified installed client supply power;
no clock/power/fan setter, direct sysfs write, SDK update or daemon restart is
used. The client interval reports1second. Raw rails, voltage/current/power,
daemon averages and the raw total are all retained, but only instantaneous
`VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0` form the selected input-rail estimate.
The overlapping DDR rail is excluded because it is already included in
`VIN_SYS_5V0`. [NVIDIA R35.4.1 rail documentation](https://docs.nvidia.com/jetson/archives/r35.4.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#software-based-power-consumption-modeling).

The frozen selection requires the same live worker PID/birth/argv on both sides
of the read, completed call counts in the measured range, and a five-second
guard after warmup completion is first observed. All raw rows remain, including
preflight/warmup/transition rows outside that analysis window. Each completed
case needs at least80eligible samples and no eligible adjacent gap above5s;
the first ordinary case has130eligible samples. Client monotonic timestamps
are not hardware rail acquisition times; sensor/cache response delay is not
independently measured. No precise request-energy, wall-plug, idle-subtracted
or exclusive model-power claim is made. Equal-worker means and paired round
contrasts will retain negative outcomes and background/temperature differences.

This is not a new continuous30minute same-worker thermal test. The previous
[35minute qualified soak](orin_smolvla_soak_memory_20261021.md) remains separate;
neither its partial power samples nor new samples are merged across campaigns.

## Resource And Application Scope

Only a new owned transient user compute unit is launched. It has an80minute
outer watchdog,30second stop timeout/control-group cleanup, and no restart.
The original SmolVLA8GiBadmission is unchanged; a separate2GiBown-job floor
is monitored. This does not lower CogACT's24GiBadmission/4GiBruntime limits.
Original Java PIDs942765/942805and birth/argv/cwd remain under read-only checks;
no application-stop authority is used, invented or extended. Existing MAXN
remains MAXN, not60W. No power-mode change is attempted before the five core
models close, and CogACT remains waiting for stable capacity or new explicit
interruption scope.

At22:20the owned unit/monitor and collector are confirmed live by actual
PID/birth/argv. The third native worker has just exited while the controller
continues; that between-worker snapshot is not a failed campaign or a restart
reason. The first worker's sampled maxRSS2,549,992KiB/maxVmHWM3,151,796KiB,
minimum system availability21,526,616KiBand275deniedFDdirectories remain
bounded observations, not complete physical GPU peaks or exclusive ownership.

## Early Audits And Retained Rejections

Fourteen source/actual-window tests pass, including all complete first-worker
outputs and real power-window mutations: insufficient coverage, wrong birth/
mode, overlapping rail sums, post-final-call readings and missing warmup guard.
The unchanged whole-model auditor rejects the actual running/incomplete cohort.
It is not bypassed using the successful selected-worker report.

The first host subset audit supplies an incorrect local reference directory;
the unchanged source-gate identity check rejects it before output comparison.
The corrected interface requires an explicit reference directory with the
original frozen manifest SHA41ad30...; no new baseline/reference is selected.
The first test report also expects the later count guard, while the actual
original auditor rejects earlier at running status. Both negative artifacts
remain; the corrected test checks the actual refusal, not a changed auditor.
The target experiment/native binaries are unchanged throughout.

## Evidence

- [Frozen complete workload and power-window rules](../../artifacts/orin/20261023-111/plan-001.json).
- [Owned-unit launch](../../artifacts/orin/20261023-111/launch-receipt-001.json).
- [Actual22:20checkpoint](../../artifacts/orin/20261023-111/checkpoint-003.json).
- [First2304complete outputs](../../artifacts/orin/20261023-111/first-worker-quality-001.json).
- [Fourteen source/actual-window checks](../../artifacts/orin/20261023-111/actual-window-tests-004.xml).
- [Fifteen checks including actual live-archive refusal](../../artifacts/orin/20261023-111/actual-window-tests-005.xml).
- [Latest22:28same-job checkpoint](../../artifacts/orin/20261023-111/checkpoint-005.json).
- [Actual live archiver refusal without restart](../../artifacts/orin/20261023-111/closed-audits-live-refusal-001.json).
- [Earlier13pass/one wrong expected-guard rejection](../../artifacts/orin/20261023-111/actual-window-tests-003.xml).
- [Incorrect reference precondition rejection](../../artifacts/orin/20261023-111/host-reference-precondition-rejection-001.json).

Do not relaunch the one-shot unit or merge partial cohorts. Resume by reading
the same actual unit and native PID/start/argv. Wait for all15workers and then
run the original full model audit, new power audit and terminal/source closure.
The full goal remains active and incomplete, with formal model acceptance4/5.

Prepared `run_closed_audits.py` refuses live/failed units, archives fresh
terminal files, invokes the original full auditor and the separate power and
source/process closure. It never launches inference or controls applications.
Never re-run the experiment launcher. Use fresh checkpoint filenames.
