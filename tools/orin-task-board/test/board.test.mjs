import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, writeFile, readFile, rm, rename, mkdir, copyFile } from 'node:fs/promises';
import { setTimeout as delay } from 'node:timers/promises';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { request } from 'node:http';
import { loadBoard, validateBoard, renderMarkdown, syncMarkdown, evidencePath, retiredEvidence, describeLiveRun, describeCollectionStatus } from '../board.mjs';
import { createBoardServer, watchBoardState } from '../server.mjs';

async function waitFor(predicate) {
  const deadline = Date.now() + 4000;
  while (!predicate()) {
    assert.ok(Date.now() < deadline, 'state polling did not observe the update');
    await delay(20);
  }
}

test('quota-independent state polling tracks atomic replacement and closes cleanly', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'edgefm-board-poll-'));
  const state = join(directory, 'state.json'), values = [], errors = [];
  await writeFile(state, JSON.stringify({ revision: 0 }));
  const watcher = watchBoardState({ state, interval: 20,
    sync: async () => values.push(JSON.parse(await readFile(state, 'utf8')).revision),
    onError: error => errors.push(error) });
  try {
    await delay(80);
    await writeFile(join(directory, 'next.json'), JSON.stringify({ revision: 1 }));
    await rename(join(directory, 'next.json'), state);
    await waitFor(() => values.includes(1));
    await writeFile(state, JSON.stringify({ revision: 2 }));
    await waitFor(() => values.includes(2));
    watcher.close();
    await writeFile(state, JSON.stringify({ revision: 3 }));
    await delay(250);
    assert.ok(!values.includes(3));
    assert.deepEqual(errors, []);
  } finally { watcher.close(); await rm(directory, { recursive: true, force: true }); }
});

test('state polling preserves errors and recovers on the next valid update', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'edgefm-board-poll-recovery-'));
  const state = join(directory, 'state.json'), values = [], errors = [];
  await writeFile(state, JSON.stringify({ revision: 0 }));
  const watcher = watchBoardState({ state, interval: 20,
    sync: async () => values.push(JSON.parse(await readFile(state, 'utf8')).revision),
    onError: error => errors.push(error) });
  try {
    await delay(80);
    await writeFile(state, '{');
    await waitFor(() => errors.length > 0);
    assert.ok(errors[0] instanceof SyntaxError);
    await writeFile(state, JSON.stringify({ revision: 1 }));
    await waitFor(() => values.includes(1));
  } finally { watcher.close(); await rm(directory, { recursive: true, force: true }); }
});

test('required models, frozen protocol and evidence remain intact', async () => {
  const data = await loadBoard();
  assert.equal(data.models.filter(x => !x.optional).length, 5);
  const completed = data.models.filter(x => !x.optional && x.formal === 'done').length;
  assert.ok(renderMarkdown(data).includes(`| 正式 E2E / CDF | ${completed}/5 模型 |`));
  assert.match(renderMarkdown(data), /不等于五模型正式交付/);
  for (const row of data.evidence) {
    if (await evidencePath(data, row.id)) continue;
    const retired = retiredEvidence(data, row.id);
    assert.ok(retired, row.path);
    assert.ok(await evidencePath(data, retired.noticeEvidenceId), row.path);
  }
});

test('evidence retirement is explicit, bounded and cannot retire its own notice', async () => {
  const data = await loadBoard();
  assert.ok(retiredEvidence(data, 'smol-cdf'));
  assert.equal(retiredEvidence(data, 'local-bulk-cleanup-report20261026'), null);
  assert.equal(retiredEvidence(data, 'not-known'), null);
  data.evidence.push({ id: 'unrelated-missing', path: 'artifacts/unrelated/missing.json' });
  assert.equal(retiredEvidence(data, 'unrelated-missing'), null);
  for (const change of [
    value => { value.project.evidenceRetirement.roots = ['artifacts/orin/../../doc']; },
    value => { value.project.evidenceRetirement.noticeEvidenceId = 'not-known'; },
    value => { value.project.evidenceRetirement.noticeEvidenceId = 'smol-cdf'; },
    value => { value.project.evidenceRetirement.authorizedAt = 'invalid'; }
  ]) {
    const copy = structuredClone(data); change(copy);
    assert.throws(() => validateBoard(copy));
  }
});

test('published figures and retained receipts work without the local artifacts tree', async () => {
  const data = await loadBoard();
  const root = await mkdtemp(join(tmpdir(), 'edgefm-board-portable-'));
  try {
    for (const row of data.evidence) {
      if (retiredEvidence(data, row.id)) continue;
      assert.ok(!row.path.startsWith('artifacts/'), row.path);
      const source = await evidencePath(data, row.id);
      assert.ok(source, row.path);
      const target = join(root, row.path);
      await mkdir(dirname(target), { recursive: true });
      await copyFile(source, target);
      assert.ok(await evidencePath(data, row.id, root), row.path);
    }
    for (const figure of data.project.evidenceFigures) {
      assert.ok(await evidencePath(data, figure.evidenceId, root));
    }
    const changed = structuredClone(data);
    changed.project.evidenceFigures[0].evidenceId = 'smol-cdf';
    assert.throws(() => validateBoard(changed));
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('model coverage advances in an isolated fixture without completing the project', async () => {
  const data = await loadBoard();
  data.project.status = 'running';
  for (const model of data.models.filter(item => !item.optional)) {
    model.formal = ['smolvla', 'rdt'].includes(model.id) ? 'done' : 'planned';
  }
  // The historical coverage fixture must not inherit a later accepted run.
  delete data.run.auditStatus;
  assert.ok(renderMarkdown(data).includes('| 正式 E2E / CDF | 2/5 模型 |'));
  assert.match(renderMarkdown(data), /尚未完成/);
});

test('explicit optional cancellation is visible but selected model completion remains required', async () => {
  const data = await loadBoard();
  const cancelled = data.tasks.find(task => task.id === 'qwen2');
  cancelled.status = 'cancelled';
  const model = data.models.find(item => item.id === 'qwen2');
  model.deployment = model.fidelity = model.formal = 'cancelled';
  assert.doesNotThrow(() => validateBoard(data));
  assert.match(renderMarkdown(data), /用户取消，不测试/);
  assert.match(renderMarkdown(data), /用户取消/);
  for (const task of data.tasks.filter(task => task.priority === 'P0')) task.status = 'done';
  const selected = data.tasks.find(task => task.id === 'qwen08');
  selected.priority = 'P0';
  selected.status = 'running';
  data.project.status = 'done';
  assert.throws(() => validateBoard(data), /Core goal/);
  selected.status = 'done';
  data.models.find(item => item.id === 'qwen08').formal = 'planned';
  assert.throws(() => validateBoard(data), /Selected model/);
  data.models.find(item => item.id === 'qwen08').formal = 'done';
  assert.doesNotThrow(() => validateBoard(data));
});

test('live heading uses current owner/counts without inferring formal acceptance', async () => {
  const data = await loadBoard();
  delete data.run.auditStatus;
  data.run.taskId = 'rdt-formal';
  data.run.totalProcesses = 15;
  data.run.completedProcesses = 2;
  data.run.measuredInProcess = 645;
  assert.match(describeLiveRun(data), /RDT-1B 2\/15/);
  assert.match(describeLiveRun(data), /645\/1024 measured/);
  data.run.completedProcesses = 15;
  assert.match(describeLiveRun(data), /仍待独立审计/);
  data.run.taskId = 'smol-formal';
  assert.match(describeLiveRun(data), /^SmolVLA 15\/15/);
});

test('accepted collection requires explicit complete audit metadata and remains distinct from goal completion', async () => {
  const data = await loadBoard();
  data.run.auditStatus = 'passed';
  data.run.auditEvidenceId = 'accepted-fixture';
  data.run.auditSha256 = 'a'.repeat(64);
  data.run.evidenceIds.push('accepted-fixture');
  data.evidence.push({ id: 'accepted-fixture', kind: 'json', path: 'unit-fixture.json' });
  data.run.completedProcesses = data.run.totalProcesses;
  data.run.measured = data.run.plannedMeasured;
  data.run.qualityFlagsPassed = true;
  data.tasks.find(task => task.id === data.run.taskId).status = 'done';
  const model = data.tasks.find(task => task.id === data.run.taskId).model;
  data.models.find(item => item.id === model).formal = 'done';
  assert.doesNotThrow(() => validateBoard(data));
  assert.match(describeCollectionStatus(data), /已审计/);
  assert.doesNotMatch(describeLiveRun(data), /待独立审计/);
  data.run.auditLabel = '匹配功耗补充已审计';
  assert.equal(describeCollectionStatus(data), data.run.auditLabel);
  assert.match(describeLiveRun(data), /匹配功耗补充已审计/);
  assert.doesNotMatch(describeLiveRun(data), /服务恢复/);
  const malformedLabel = structuredClone(data); malformedLabel.run.auditLabel = 'bad|label';
  assert.throws(() => validateBoard(malformedLabel), /Invalid audit label/);
  const partialLabel = structuredClone(data); delete partialLabel.run.auditStatus;
  assert.notEqual(describeCollectionStatus(partialLabel), data.run.auditLabel);
  const partial = structuredClone(data); partial.run.completedProcesses--;
  assert.throws(() => validateBoard(partial), /acceptance/);
  const missing = structuredClone(data); missing.run.auditSha256 = '';
  assert.throws(() => validateBoard(missing), /acceptance/);
  const incompleteCore = structuredClone(data);
  incompleteCore.tasks.find(task => task.id === 'memory').status = 'running';
  incompleteCore.project.status = 'done';
  assert.throws(() => validateBoard(incompleteCore), /Core goal/);
});

test('operator summary follows both audited tasks without implying whole-goal completion', async () => {
  for (const status of ['running', 'blocked']) {
    const data = await loadBoard();
    data.project.status = status;
    for (const id of ['operator-ablation', 'operator-integration']) data.tasks.find(task => task.id === id).status = 'done';
    assert.match(renderMarkdown(data), /算子 \/ 集成消融 \| 范围内已完成/);
    assert.equal(data.project.status, status);
    data.tasks.find(task => task.id === 'operator-ablation').status = 'waiting';
    assert.match(renderMarkdown(data), /算子 \/ 集成消融 \| 未闭合/);
  }
});

test('schema rejects narrowed scope, invalid counters and dangling evidence', async () => {
  const data = await loadBoard();
  for (const change of [
    x => { x.models = x.models.filter(m => m.id !== 'cogact'); },
    x => { x.run.totalProcesses = 1; },
    x => { x.run.plannedMeasured = 1024; },
    x => { x.run.measured = 20000; },
    x => { x.run.callsInProcess++; },
    x => { x.tasks[0].status = 'toString'; },
    x => { x.tasks[0].evidenceIds = ['missing']; },
    x => { x.tasks[0].dependsOn = ['missing']; },
    x => { x.models.find(model => !model.optional).formal = 'planned'; x.project.status = 'done'; }
  ]) {
    const copy = structuredClone(data); change(copy);
    assert.throws(() => validateBoard(copy));
  }
});

test('one structured state renders a consistent Markdown checkpoint', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'edgefm-board-sync-'));
  try {
    const data = await loadBoard(), output = join(directory, 'board.md');
    assert.equal(await syncMarkdown(data, output), true);
    assert.equal(await syncMarkdown(data, output), false);
    assert.match(await readFile(output, 'utf8'), new RegExp(`${data.run.completedProcesses}/${data.run.totalProcesses}`));
    data.run.completedProcesses = 2;
    delete data.run.auditStatus;
    await syncMarkdown(data, output);
    assert.match(await readFile(output, 'utf8'), new RegExp(`完成进程 \\| 2/${data.run.totalProcesses}`));
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('two configurations retain five processes and fixed counts with a pinned protocol', async () => {
  const data = await loadBoard();
  delete data.run.auditStatus;
  data.evidence.push({ id: 'two-mode-fixture-spec', kind: 'json', path: 'NONMODEL-fixture.json' });
  data.run.evidenceIds.push('two-mode-fixture-spec');
  Object.assign(data.run, { totalProcesses: 10, plannedMeasured: 10240, plannedCalls: 11520,
    completedProcesses: 0, measured: 0, callsInProcess: 0, measuredInProcess: 0, warmupInProcess: 0,
    protocolConfig: { modes: ['mock-ordinary', 'mock-candidate'], processesPerMode: 5,
      warmup: 128, measured: 1024, batch: 1, specEvidenceId: 'two-mode-fixture-spec', specSha256: 'a'.repeat(64) } });
  assert.doesNotThrow(() => validateBoard(data));
  for (const change of [
    value => { value.run.protocolConfig.modes.pop(); },
    value => { value.run.protocolConfig.measured = 16; },
    value => { value.run.protocolConfig.processesPerMode = 4; },
    value => { value.run.protocolConfig.specSha256 = ''; },
    value => { value.run.plannedCalls--; }
  ]) {
    const changed = structuredClone(data); change(changed);
    assert.throws(() => validateBoard(changed));
  }
});

test('evidence is a whitelist, not an arbitrary file reader', async () => {
  const data = await loadBoard();
  assert.equal(await evidencePath(data, '../.git/config'), null);
  const copy = structuredClone(data);
  copy.evidence.push({ id: 'outside', path: '/etc/passwd.json' });
  assert.equal(await evidencePath(copy, 'outside'), null);
  copy.evidence.push({ id: 'unsupported', path: 'vlaforge/CMakeLists.txt' });
  assert.equal(await evidencePath(copy, 'unsupported'), null);
});

test('local HTTP serves state/assets, rejects writes, rebinding and unknown evidence', async () => {
  const server = createBoardServer();
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const url = `http://127.0.0.1:${server.address().port}`;
  try {
    for (const path of ['/', '/styles.css', '/app.js', '/vendor/lucide.js', '/api/board', '/evidence/local-bulk-cleanup-report20261026']) assert.equal((await fetch(url + path)).status, 200, path);
    const retired = await fetch(url + '/evidence/smol-cdf');
    assert.equal(retired.status, 410);
    assert.match(await retired.text(), /href="\/evidence\/local-bulk-cleanup-report20261026"/);
    const response = await fetch(url + '/api/board'), data = await response.json();
    assert.equal(data.schema, 'edgefm.task_board/1');
    assert.equal(response.headers.get('cache-control'), 'no-store');
    assert.equal((await fetch(url + '/api/board', { method: 'POST', body: '{}' })).status, 405);
    assert.equal((await fetch(url + '/__proto__')).status, 404);
    assert.equal((await fetch(url + '/evidence/../../.git/config')).status, 404);
    assert.equal((await fetch(url + '/evidence/not-known')).status, 404);
    const hostile = await new Promise((resolve, reject) => {
      const call = request(url, { headers: { Host: 'outside.invalid' } }, result => { result.resume(); resolve(result.statusCode); });
      call.on('error', reject); call.end();
    });
    assert.equal(hostile, 403);
  } finally { await new Promise(resolve => server.close(resolve)); }
});

test('state endpoint reads new checkpoints without rebuilding or mutating evidence', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'edgefm-board-http-'));
  const state = join(directory, 'state.json'), data = await loadBoard();
  await writeFile(state, JSON.stringify(data));
  const server = createBoardServer({ state });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const url = `http://127.0.0.1:${server.address().port}/api/board`;
  try {
    assert.equal((await (await fetch(url)).json()).run.completedProcesses, data.run.completedProcesses);
    data.run.completedProcesses = 2;
    delete data.run.auditStatus;
    await writeFile(state, JSON.stringify(data));
    assert.equal((await (await fetch(url)).json()).run.completedProcesses, 2);
    assert.equal(await readFile(state, 'utf8'), JSON.stringify(data));
    await writeFile(state, '{}');
    assert.equal((await fetch(url)).status, 503);
  } finally { await new Promise(resolve => server.close(resolve)); await rm(directory, { recursive: true, force: true }); }
});
