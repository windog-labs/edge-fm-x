# Orin Formal Window Mechanism Review

Checkpoint: 2026-10-23 20:42UTC. **Three CPU-only NONMODEL lifecycle probes pass
on Orin; no VLA inference, original Java interruption or formal samples occur.**
This qualifies an operating-system mechanism, not a production service lease,
real application recovery or permission to start CogACT formal sampling.

## Review Findings

The existing formal wrapper31uses a six-hour outer alarm and16GiBpost-stop
admission; its inner service guard has a36-hour fallback. The outer alarm is
normally the tighter bound:the36-hour fallback does not mean earlier campaigns
ran that long. Neither default directly implements the proposed four-hour,
24GiB/4GiBCogACTwindow. Old sources/accepted campaigns remain unchanged.

Nested workers also require explicit ownership cleanup. A native worker can
start a separate process group/session under a memory guard; stopping only the
collector group is not by itself proof that every such descendant is gone.
The existing guards normally cooperate on TERM. No actual orphan or overrun is
asserted for previous accepted runs. The remaining review concerns future
failure handling and the final production integration.

## Actual Platform And Tested Direction

Read-only checks find systemd245.4-4ubuntu3.21with the hybrid cgroup hierarchy.
System-level `sudo -n -l /usr/bin/systemd-run` requires a password and is not used.
The existing UID1000user manager and `/run/user/1000/bus` are usable. Its initial
state is degraded; no unrelated unit/configuration is repaired or reset.

The tested direction uses newly named transient **user** service units,
`Type=exec`, `RuntimeMaxSec`, `TimeoutStopSec`, `KillMode=control-group` and an
`ExecStopPost` marker. This keeps experimental descendants together even when
the child changes session. The original application programs remain under their
original Supervisor; they are not migrated to systemd. The version-matched
systemd documentation defines cgroup-wide stop and post-stop behavior.
[systemd245service documentation](https://raw.githubusercontent.com/systemd/systemd/v245/man/systemd.service.xml),
[systemd245kill documentation](https://raw.githubusercontent.com/systemd/systemd/v245/man/systemd.kill.xml).

Only task-owned random-named units are created/stopped/cleared. Before creating
one, its name must be absent; before controlling it, its unique description must
match. The source performs no original-application start/stop RPC. No persistent
unit is installed/enabled, no SDK or power setting is changed, and no root
privilege is used.

## Three Real NONMODEL Cases

All tests use a6-second runtime and2-second stop timeout. Main and child are
ordinary Python CPU sleepers. The detached child has PGID=SID=its own PID while
remaining in the same unit cgroup. A separate outside process must survive the
unit's cleanup, then is explicitly cleaned by its own Popen owner.

| Case | Actual SERVICE_RESULT | Main Exit | Post-Hook Other Cgroup PIDs | Outside Process |
|---|---|---|---|---|
| Normal completion | success | exited0 | Empty | Survives |
| Runtime expires; main/child ignore TERM | timeout | killedKILL | Empty | Survives |
| Own main process explicitly SIGKILLed | signal | killedKILL | Empty | Survives |

The post markers and final PID/birth checks are retained. A later independent
read-only query finds allthreeunits `LoadState=not-found`, `ActiveState=inactive`,
`MainPID=0`. The original Java PIDs942765/942805remain Running before/after under
the original manager. This is process observation, not application health.

The original six-hour build/pilot authority is now expired. These tests do not
use that authority:they do not interrupt either application or run a model.
The raw probe's `model_or_service_operations=false` field means no model or
original-application operation. It does **not** mean no transient user service
unit operation; the qualified report explicitly records those operations.

## Budget Logic And Remaining Work

The pure budget calculator refuses missing independent formal authority,
build/pilot schema reuse, durations over four hours, mismatched/future origins,
expired budgets and inadequate recovery reserve. A later attempt with the same
origin has less runtime, not a renewed deadline. The proposed reserve is900s,
covering two300sstop/post-stop budgets with margin. This function neither
creates permission nor launches a unit; synthetic authorization test inputs
are NONMODEL, not human approval.

Production integration remains unqualified. An actual controller must bind
the human record and a persisted one-time origin, recheck the absolute deadline
after unit start, validate the original manager/JAR/cwd/config, stop only the
approved two programs, freeze the active lease and24GiB/4GiBformal spec, and
restore through the original manager after owned cleanup. Those steps have not
been combined into an authorized four-hour formal run here.

The CPU tests do not demonstrate GPU-driver failure handling, uninterruptible
tasks, user-manager/power loss, or actual Java recovery after a forced kill.
The [earlier actual ordinary repeat](orin_cogact_ordinary_repeat_20261024.md)
independently proves the normal original-manager restore contract. Neither
that result nor these mechanism tests implies all exceptional recovery works.

Twenty host checks pass over actual archives, false cleanup/cgroup/isolation
claims and budget rejection cases. Required formal VLA coverage remains4/5;
the existing [formal source package](orin_cogact_formal_source_readiness_20261024.md)
stays unactivated. Independent formal permission, production lease/restore
integration and fresh target preflight remain outstanding.

## Evidence

- [Qualified mechanism audit](../../artifacts/orin/20261023-108/lifecycle-capability-audit-001.json).
- [Actual target CPU-only report and commands](../../artifacts/orin/20261023-108/nonmodel-unit-001/report.json).
- [Final unit disappearance and Java observation](../../artifacts/orin/20261023-108/unit-final-observation-001.json).
- [Twenty host evidence/budget tests](../../artifacts/orin/20261023-108/lifecycle-evidence-tests-001.xml).
- [Pure budget calculation, not a launcher](../../artifacts/orin/20261023-108/formal_window_budget.py).

Recheck locally without model/process/service execution:

```sh
env PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=artifacts/orin/20261023-108 python3 -B -m pytest -q \
  -p no:cacheprovider artifacts/orin/20261023-108/test_lifecycle_evidence.py
```
