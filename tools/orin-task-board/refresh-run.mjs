import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile, rename } from 'node:fs/promises';
import { join } from 'node:path';
import { loadBoard, validateBoard, syncMarkdown, describeLiveRun, evidencePath, ROOT, STATE } from './board.mjs';

const script = `import csv,json,time,sys,hashlib
from pathlib import Path
p=Path(sys.argv[1]); c=json.loads(p.read_text())
def identity(item):
    q=Path('/proc',str(item['pid']))
    try:
        return {'present':True,'matches':q.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]==item['start_ticks'] and q.joinpath('cmdline').read_bytes().decode().rstrip('\\0').split('\\0')==item['argv']}
    except FileNotFoundError:
        return {'present':False,'matches':False}
cases=[]
for item in c['cases']:
    f=p.parent/item['directory']/'execute.log'
    text=f.read_text() if f.exists() else ''
    lines=text.splitlines(keepends=True)
    if lines and not lines[-1].endswith('\\n'): lines.pop()
    rows=list(csv.DictReader(lines))
    cases.append({'round':item['round'],'mode':item['mode'],'directory':item['directory'],'calls':len(rows),'measured':sum(int(r['measured']) for r in rows),'warmup':sum(r['measured']=='0' for r in rows),'flags_passed':bool(rows) and all(r['finite']==r['direct_exact']=='1' for r in rows),'exit_code':item.get('exit_code'),'child':item['child'],'identity':identity(item['child'])})
print(json.dumps({'status':c['status'],'campaign_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'serial':Path('/proc/device-tree/serial-number').read_text().strip('\\0\\n'),'elapsed_seconds':(time.monotonic_ns()-c['started_monotonic_ns'])/1e9,'controller':c['controller'],'controller_identity':identity(c['controller']),'protocol':c['protocol'],'cases':cases}))
`;
const before = await loadBoard();
const child = spawn('ssh', ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', 'orin', 'python3', '-', before.run.targetPath + 'campaign.json'], { stdio: ['pipe', 'pipe', 'pipe'] });
let stdout = '', stderr = '';
child.stdout.setEncoding('utf8'); child.stdout.on('data', chunk => { stdout += chunk; });
child.stderr.setEncoding('utf8'); child.stderr.on('data', chunk => { stderr += chunk; });
child.stdin.end(script);
const timer = setTimeout(() => child.kill('SIGTERM'), 30000);
const code = await new Promise((resolve, reject) => { child.once('error', reject); child.once('exit', resolve); }).finally(() => clearTimeout(timer));
if (code !== 0) throw new Error(`Observation failed; state retained: ${stderr}`);
const snapshot = JSON.parse(stdout), data = await loadBoard();
const audited = data.run.auditStatus === 'passed';
if (audited) {
  const path = await evidencePath(data, data.run.auditEvidenceId, ROOT);
  if (!path) throw new Error('Accepted audit evidence is unavailable; state retained');
  const bytes = await readFile(path), audit = JSON.parse(bytes);
  if (createHash('sha256').update(bytes).digest('hex') !== data.run.auditSha256
      || audit.status !== 'full-data-and-restoration-verified' || !audit.formal_sampling
      || !audit.model_data_audited || audit.campaign_sha256 !== snapshot.campaign_sha256
      || snapshot.status !== 'collected-independent-audit-pending') {
    throw new Error('Current collection differs from the accepted final audit; state retained');
  }
}
if (snapshot.serial !== data.project.serial || snapshot.boot_id !== data.run.bootId) throw new Error('Board identity changed; state retained for review');
if (snapshot.status === 'running' && !snapshot.controller_identity.matches) throw new Error('Controller identity not live/matched; no restart or inferred completion');
if (snapshot.protocol.processes_per_mode !== 5 || snapshot.protocol.measured_per_process !== 1024 || snapshot.protocol.warmup_per_process !== 128) throw new Error('Unexpected protocol');
const current = snapshot.cases.at(-1);
if (!current) throw new Error('No worker has started; preserve previous checkpoint');
const at = new Date().toISOString(), done = snapshot.cases.filter(x => x.exit_code === 0).length;
const oldDone = data.run.completedProcesses;
Object.assign(data.run, { status: snapshot.status, verifiedAt: at, completedProcesses: done,
  measured: snapshot.cases.reduce((sum, item) => sum + item.measured, 0), round: current.round + 1,
  mode: { off: '常规 TRT', 'batch-only': '共享流', required: '完整迭代 replay' }[current.mode],
  callsInProcess: current.calls, warmupInProcess: current.warmup, measuredInProcess: current.measured,
  qualityFlagsPassed: snapshot.cases.every(x => x.calls === 0 || x.flags_passed),
  controller: { pid: snapshot.controller.pid, startTicks: snapshot.controller.start_ticks, identityVerified: snapshot.controller_identity.matches },
  worker: { pid: current.child.pid, startTicks: current.child.start_ticks, identityVerified: current.identity.matches } });
data.project.verifiedAt = at;
const task = data.tasks.find(x => x.id === 'rdt-formal');
if (snapshot.status === 'running' && data.project.currentTaskId === data.run.taskId) {
  data.project.currentStage = describeLiveRun(data);
}
if (snapshot.status === 'collected-independent-audit-pending' && !audited) {
  task.status = 'waiting'; task.summary = '正式采集已结束，等待全量独立审计；尚未标为正式通过。';
  data.models.find(model => model.id === 'rdt').formal = 'waiting';
  data.project.currentStage = 'RDT 全量独立审计待执行';
} else if (snapshot.status === 'failed') {
  const safety = data.run.interruption?.kind === 'resource-safety-stop';
  task.status = 'blocked'; task.summary = safety ? '资源安全中断：11 个完整进程与末进程部分数据保留，正式协议未完成；不自动重启。' : '正式采样记录失败，保留原始证据并等待调查；未自动重启。';
  data.models.find(model => model.id === 'rdt').formal = 'blocked';
  const retainedDone = data.tasks.find(item => item.id === 'rdt-partial-evidence')?.status === 'done';
  if (['rdt-formal', 'rdt-partial-evidence', 'memory'].includes(data.project.currentTaskId)) {
    data.project.currentStage = safety ? (retainedDone ? 'RDT 资源待确认：保留输出已核验' : 'RDT 资源安全中断：归档与已有输出核验') : 'RDT 采样异常待调查';
  }
}
if (done !== oldDone) data.events.unshift({ id: `${data.run.id}-processes-${done}`, at, title: `RDT 正式采样完成 ${done}/15 个进程`, detail: `当前批次 ${data.run.id} 累计 ${data.run.measured.toLocaleString('en-US')} 条 measured；完整输出与正式结果仍待独立审计。`, evidenceIds: ['rdt-pilot', 'postprocess'] });
const directory = join(ROOT, 'artifacts/orin-task-board');
await mkdir(directory, { recursive: true });
const name = `checkpoint-${at.replace(/[^0-9]/g, '')}.json`;
await writeFile(join(directory, name), JSON.stringify({ verifiedAt: at, ...snapshot }, null, 2) + '\n');
data.run.observationPath = `artifacts/orin-task-board/${name}`;
const observation = { id: 'run-snapshot', title: '当前采样进程 / CSV 只读核验', kind: 'json', path: data.run.observationPath };
const previous = data.evidence.findIndex(item => item.id === observation.id);
if (previous < 0) data.evidence.push(observation); else data.evidence[previous] = observation;
if (!task.evidenceIds.includes(observation.id)) task.evidenceIds.push(observation.id);
if (!data.run.evidenceIds.includes(observation.id)) data.run.evidenceIds.push(observation.id);
validateBoard(data);
const temporary = `${STATE}.${process.pid}.tmp`;
await writeFile(temporary, JSON.stringify(data, null, 2) + '\n'); await rename(temporary, STATE);
await syncMarkdown(data);
console.log(JSON.stringify({ verifiedAt: at, status: snapshot.status, completedProcesses: done, measured: data.run.measured, currentProcessMeasured: current.measured, controllerMatched: snapshot.controller_identity.matches, workerMatched: current.identity.matches }));
