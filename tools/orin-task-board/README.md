# EdgeFM Orin Task Board

Local-only experiment tracking. It does not run or control model experiments,
contact Orin in the background, publish files or expose the workspace directory.

## Start

From the repository root:

```bash
npm --prefix tools/orin-task-board install --no-audit --no-fund
node tools/orin-task-board/start.mjs
```

The launcher reuses this workspace's existing server, otherwise selects an
unused loopback port starting at 4317. It does not terminate other services.
Its returned URL is authoritative. Server PID/logs are under
`artifacts/orin-task-board/`. The browser can open the local URL while the
Codex conversation is idle. The server does not restart across a machine reboot.

## Update Progress Once

The single source is `doc/edgefm_orin_task_board_state.json`. On every task
milestone, state change, new verified evidence or failure, update its affected
tasks/model gates, event, evidence references and verification time. The local
server watches this file, regenerates `doc/edgefm_orin_task_board.md`, and the
webpage fetches the same local state every 15 seconds. Browser refreshes never
query the board or start experiments.

For an explicit one-off read-only RDT process/CSV check:

```bash
node tools/orin-task-board/refresh-run.mjs
```

This verifies the actual serial, boot ID, controller PID/start ticks/argv,
frozen protocol and recorded CSV counts through SSH, preserves an immutable
observation receipt and updates both views. It is not scheduled. Observation
errors preserve the old checkpoint; no retry starts/restarts inference.
Completed collection is labeled awaiting independent audit, never formal pass.
Task/model acceptance still requires inspecting the full experiment evidence.

Recent-progress text wraps long experiment keys and hashes at 320px without
hiding content or reducing font size. An isolated browser fixture guards this
case even after the corresponding live progress event scrolls out of the feed.

Without the server, regenerate Markdown after changing the state:

```bash
node tools/orin-task-board/sync.mjs
```

Do not edit the generated Markdown as a separate data source. The web interface
offers overview/board/list, task details, model matrix, optional-task/model/status
filters, search, evidence links, actual audited CDF images and a JSON download.
It is intentionally read-only. No task drag/drop can falsely certify a model.

## Verify

```bash
npm --prefix tools/orin-task-board test
npm --prefix tools/orin-task-board run test:browser
```

The browser suite uses the already installed full Chromium channel; it does
not require the separate headless-shell binary. It covers desktop, tablet,
390px and 320px mobile layouts, overview and risk/evidence overflow/text fitting,
filtering/details, formal-versus-pilot distinctions, evidence images and state refresh. Screenshots
are saved under `artifacts/orin-task-board/screenshots/`. Test browser state
mocks never overwrite the real task state or experiment data.

Search workflow checks use isolated fixtures for summary/remaining-text matches,
including common tasks that mention a model. They do not hardcode a count of
matches in the evolving live task descriptions.
Terminal collection states are also checked: an interrupted run is not rendered
as active sampling, and collected data awaiting audit is not formal acceptance.

The browser test configuration uses the launcher's actual URL when present.
Without a launch receipt it can start a loopback test server. For a separately
started server, set `BOARD_URL` to its local URL.

Published figures are declared by `project.evidenceFigures`; they point to
byte-exact archived images, not guesses for deleted original paths. Small
historical receipts are versioned under `doc/evidence`. Declared retired roots
return an HTTP410 notice; unknown/unexpected missing evidence is not excused.

## Boundaries

- Loopback binding, local Host checks, read-only HTTP endpoints and no CORS.
- Evidence files are explicitly allowlisted, resolved inside this worktree and
  limited to document/data/image extensions. There is no arbitrary file browser.
- All remote model/run facts are timestamped snapshots, not unverified live
  claims. A local refresh is distinct from a new Orin observation.
- Five required VLA models and the fixed 15-process protocol are validated.
- Pilots, static signatures and partial collection are not formal completion.
- The model runtime, frozen engines, plugins and running collection are untouched.
