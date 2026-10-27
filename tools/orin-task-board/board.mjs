import { readFile, writeFile, rename, realpath, stat } from 'node:fs/promises';
import { dirname, resolve, relative, sep, extname } from 'node:path';
import { fileURLToPath } from 'node:url';

export const HERE = dirname(fileURLToPath(import.meta.url));
export const ROOT = resolve(HERE, '../..');
export const STATE = resolve(ROOT, 'doc/edgefm_orin_task_board_state.json');
export const MARKDOWN = resolve(ROOT, 'doc/edgefm_orin_task_board.md');
export const STATUS = { running: '进行中', waiting: '等待依赖', planned: '待开始', done: '已完成', blocked: '需处理', cancelled: '用户取消' };

export function validateBoard(data) {
  if (data?.schema !== 'edgefm.task_board/1') throw new Error('Invalid board schema');
  const unique = (items, label) => {
    if (!Array.isArray(items) || items.some(x => !/^[a-z0-9-]+$/.test(x.id)) || new Set(items.map(x => x.id)).size !== items.length) throw new Error(`Invalid ${label} IDs`);
    return new Set(items.map(x => x.id));
  };
  const models = unique(data.models, 'model'), tasks = unique(data.tasks, 'task'), evidence = unique(data.evidence, 'evidence');
  const retirement = data.project.evidenceRetirement;
  if (retirement && (!Number.isFinite(Date.parse(retirement.authorizedAt))
      || !evidence.has(retirement.noticeEvidenceId) || !Array.isArray(retirement.roots)
      || !retirement.roots.length || retirement.roots.some(path => typeof path !== 'string'
        || !/^artifacts\/[a-zA-Z0-9_-]+(?:\/[a-zA-Z0-9_-]+)*$/.test(path)))) {
    throw new Error('Invalid evidence retirement declaration');
  }
  if (retirement && retiredEvidence(data, retirement.noticeEvidenceId)) throw new Error('Retirement notice must remain available');
  if (data.project.evidenceFigures !== undefined && (!Array.isArray(data.project.evidenceFigures)
      || data.project.evidenceFigures.some(figure => !evidence.has(figure.evidenceId)
        || extname(data.evidence.find(row => row.id === figure.evidenceId).path).toLowerCase() !== '.png'
        || retiredEvidence(data, figure.evidenceId)
        || [figure.alt, figure.caption].some(value => typeof value !== 'string' || !value.trim() || value.length > 300)))) {
    throw new Error('Invalid published evidence figure');
  }
  const required = data.models.filter(x => !x.optional).map(x => x.id).sort();
  if (required.join(',') !== ['smolvla', 'rdt', 'pi0', 'pi05', 'cogact'].sort().join(',')) throw new Error('Five required models must remain in scope');
  if (!Number.isFinite(Date.parse(data.project.verifiedAt)) || !tasks.has(data.project.currentTaskId)) throw new Error('Invalid current checkpoint');
  if (!['running', 'done', 'paused', 'blocked'].includes(data.project.status)) throw new Error('Invalid project status');
  if (data.project.status === 'done' && (data.models.some(x => !x.optional && x.formal !== 'done') || data.tasks.some(x => x.priority === 'P0' && x.status !== 'done'))) throw new Error('Core goal is not complete');
  if (data.project.status === 'done' && data.tasks.some(task => task.priority === 'P0' && task.model !== 'common'
      && data.models.find(model => model.id === task.model)?.formal !== 'done')) throw new Error('Selected model is not formally complete');
  for (const task of data.tasks) {
    if (!Object.hasOwn(STATUS, task.status) || !['P0', 'P1'].includes(task.priority) || (!models.has(task.model) && task.model !== 'common')) throw new Error('Invalid task state');
    if (!task.dependsOn.every(id => tasks.has(id))) throw new Error('Unknown task dependency');
  }
  for (const row of [...data.tasks, ...data.models, ...data.risks, ...data.events, data.run]) {
    if (!row.evidenceIds.every(id => evidence.has(id))) throw new Error('Unknown evidence reference');
  }
  const run = data.run;
  if (run.auditLabel !== undefined && (typeof run.auditLabel !== 'string' || !run.auditLabel.trim()
      || run.auditLabel.length > 80 || /[\r\n|]/.test(run.auditLabel))) throw new Error('Invalid audit label');
  if (run.taskId !== undefined && !tasks.has(run.taskId)) throw new Error('Unknown run task');
  const protocol = run.protocolConfig ?? { modes: ['off', 'batch-only', 'required'], processesPerMode: 5, warmup: 128, measured: 1024, batch: 1 };
  if (!Array.isArray(protocol.modes) || protocol.modes.length < 2 || protocol.modes.some(mode => typeof mode !== 'string' || !mode)
      || new Set(protocol.modes).size !== protocol.modes.length || protocol.processesPerMode !== 5
      || protocol.warmup !== 128 || protocol.measured !== 1024 || protocol.batch !== 1
      || run.totalProcesses !== protocol.modes.length * 5 || run.plannedMeasured !== run.totalProcesses * 1024
      || run.plannedInProcess !== 1024 || run.plannedCalls !== run.totalProcesses * 1152) throw new Error('Frozen formal protocol cannot be reduced');
  if (run.protocolConfig && (!evidence.has(protocol.specEvidenceId) || !run.evidenceIds.includes(protocol.specEvidenceId)
      || !/^[0-9a-f]{64}$/.test(protocol.specSha256 ?? ''))) throw new Error('Formal modes require their frozen specification evidence');
  for (const key of ['completedProcesses', 'measured', 'measuredInProcess', 'callsInProcess', 'warmupInProcess']) {
    if (!Number.isSafeInteger(run[key]) || run[key] < 0) throw new Error('Invalid run count');
  }
  if (run.completedProcesses > run.totalProcesses || run.measured > run.plannedMeasured || run.measuredInProcess > run.plannedInProcess || run.warmupInProcess > 128 || run.callsInProcess > 1152 || run.callsInProcess !== run.warmupInProcess + run.measuredInProcess) throw new Error('Inconsistent progress counts');
  if (data.events.some(x => !Number.isFinite(Date.parse(x.at)))) throw new Error('Invalid event timestamp');
  if (run.auditStatus === 'passed') {
    const task = data.tasks.find(item => item.id === run.taskId);
    const model = data.models.find(item => item.id === task?.model);
    if (run.completedProcesses !== run.totalProcesses || run.measured !== run.plannedMeasured
        || !run.qualityFlagsPassed || task?.status !== 'done' || model?.formal !== 'done'
        || !evidence.has(run.auditEvidenceId) || !run.evidenceIds.includes(run.auditEvidenceId)
        || !/^[0-9a-f]{64}$/.test(run.auditSha256 ?? '')) {
      throw new Error('Formal run acceptance requires complete counts, model/task and audit evidence');
    }
  }
  return data;
}

export async function loadBoard(path = STATE) {
  return validateBoard(JSON.parse(await readFile(path, 'utf8')));
}

export function describeLiveRun(data) {
  const run = data.run, task = data.tasks.find(item => item.id === run.taskId);
  const model = data.models.find(item => item.id === task?.model);
  if (run.auditStatus === 'passed') return `${model?.name ?? run.label} ${run.auditLabel ?? '正式数据与服务恢复已审计'}；${run.completedProcesses}/${run.totalProcesses} 个完整进程`;
  const measured = Number(run.measuredInProcess).toLocaleString('en-US');
  return `${model?.name ?? run.label} ${run.completedProcesses}/${run.totalProcesses} 个完整进程；第 ${run.round} 轮${run.mode}，当前 ${measured}/${run.plannedInProcess} measured；正式结果仍待独立审计`;
}

export function describeCollectionStatus(data) {
  if (data.run.auditStatus === 'passed') return data.run.auditLabel ?? '正式数据与服务恢复已审计';
  return data.run.status === 'running' ? '运行中' : data.run.status === 'failed'
    ? '已中断，未正式完成' : '采集已结束，待独立审计';
}

const md = value => String(value ?? '').replaceAll('|', '\\|').replaceAll('\n', '<br>');
const n = value => Number(value).toLocaleString('en-US');
export function renderMarkdown(data) {
  validateBoard(data);
  const required = data.models.filter(x => !x.optional), run = data.run;
  const names = Object.fromEntries(data.models.map(x => [x.id, x.name]));
  const checkpoint = new Date(data.project.verifiedAt).toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai', hour12: false });
  const deployment = { passed: '已验证', planned: '未开始', cancelled: '用户取消，不测试' };
  const fidelity = { 'formal-passed': '正式数据已审计', 'pilot-passed': 'pilot 通过，正式待审计', planned: '未开始', cancelled: '用户取消，不测试' };
  const lines = [
    '# EdgeFM Orin 实验任务看板', '',
    `最后核验：${checkpoint}，Asia/Shanghai。此为已核验快照，不是实时台架遥测。`, '',
    `**总状态：${{ running: '进行中，尚未完成', paused: '已暂停，范围不变', blocked: '需处理，尚未完成', done: '已完成必需验收' }[data.project.status]}。当前位置：${data.project.currentStage}。**`, '',
    '网页版：<http://127.0.0.1:4317>。若端口已占用，以启动器实际返回的地址为准。', '',
    '本文件与网页共用 [结构化状态](edgefm_orin_task_board_state.json)，由 `node tools/orin-task-board/sync.mjs` 生成。',
    '后续每次有进展，更新同一状态文件；运行中的本地服务会同步生成本文件并刷新网页。', '',
    '## 一眼看进度', '',
    '| 交付项 | 已验证进度 | 说明 |', '|---|---|---|',
    `| 完整链路真机 pilot | ${required.filter(x => x.deployment === 'passed').length}/5 模型 | 不等于五模型正式交付 |`,
    `| 正式 E2E / CDF | ${required.filter(x => x.formal === 'done').length}/5 模型 | 不是按工作量计算的整体完成百分比 |`,
    `| 算子 / 集成消融 | ${['operator-ablation', 'operator-integration'].every(id => data.tasks.find(task => task.id === id)?.status === 'done') ? '范围内已完成' : '未闭合'} | 保留负收益、数值合同和计时边界；不等于全模型加速 |`,
    '| 资源 / 功耗 | 部分完成 | 已有热稳态及匹配功耗证据；物理峰值归因和其他功耗模式仍有边界 |', '',
    '## 当前所在位置', '', `已正式验收：${required.filter(x => x.formal === 'done').map(x => md(x.name)).join('、') || '暂无'} -> **${md(data.project.currentStage)}** -> 公共消融及最终交付 -> 可选 Qwen3.5。`, '',
    md(data.project.scope), '', md(data.project.acceptance), '',
    '## 模型主线', '', '| 模型 | 完整部署 | 动作核验 | 正式采样 / CDF | 下一步 |', '|---|---|---|---|---|',
    ...data.models.map(x => `| ${md(x.name)}${x.optional ? '（可选）' : ''} | ${deployment[x.deployment] ?? x.deployment}${x.engines ? `，${x.engines} engines / ${x.steps} 步` : ''} | ${fidelity[x.fidelity] ?? x.fidelity} | ${STATUS[x.formal]} | ${md(x.next)} |`), '',
    'CogACT 保留实际约 7.63B 公开依赖配置身份，不改称已测名义 3B。未开始不等于不支持。', '',
    '## 当前采样快照', '', '| 项目 | 核验事实 |', '|---|---|',
    `| 采集状态 | ${describeCollectionStatus(data)} |`,
    `| 台架 | ${data.project.device}，序列号 \`${data.project.serial}\` |`,
    `| 软件 / 功耗 | ${md(data.project.sdk)}；${md(data.project.powerMode)} |`,
    `| 协议 | ${md(run.protocol)} |`,
    `| 完成进程 | ${run.completedProcesses}/${run.totalProcesses} |`,
    `| 当前进程 | 第 ${run.round} 轮 ${md(run.mode)}；${n(run.callsInProcess)} 次调用，${n(run.warmupInProcess)} 次预热，${n(run.measuredInProcess)}/${n(run.plannedInProcess)} 条测量 |`,
    `| 总 measured | ${n(run.measured)}/${n(run.plannedMeasured)} |`,
    `| 已记录质量 | ${md(run.qualityNote)} |`,
    `| 估时 | ${md(run.estimate)} |`,
    `| 控制器 | PID \`${run.controller.pid}\` / start ticks \`${run.controller.startTicks}\` |`,
    `| worker | PID \`${run.worker.pid}\` / start ticks \`${run.worker.startTicks}\` |`,
    `| Boot ID | \`${run.bootId}\` |`,
    `| 数据目录 | \`${run.targetPath}\` |`, '',
    '板端 civil clock 不正确，使用 monotonic time。SSH 观察超时不代表任务失败；恢复时核对实际 PID/start ticks/argv 与日志，不自动重启。', '',
    '## 任务队列', '', '| ID | 任务 | 状态 | 已有证据 / 剩余动作 | 依赖 |', '|---|---|---|---|---|',
    ...data.tasks.map(x => `| ${x.id} | ${md(x.title)}${x.optional ? '（可选）' : ''} | ${STATUS[x.status]} | ${md(x.summary)}${x.remaining.length ? '<br>待完成：' + md(x.remaining.join('；')) : ''} | ${x.dependsOn.join(', ') || '无'} |`), '',
    '## 验收与风险', '', ...data.risks.map(x => `- **${md(x.title)}**：${md(x.detail)}`),
    ...data.project.constraints.map(x => '- ' + md(x)), '',
    '剩余模型工期取决于导出、算子和容量实测；RDT 纯采样时间不能外推为整项目工期。全五模型及必需交付未闭合前，Goal 不标完成。', '',
    '## 证据入口', '', ...data.evidence.filter(x => x.id !== 'task-board').map(x => {
      const retired = retiredEvidence(data, x.id);
      const path = retired ? data.evidence.find(item => item.id === retired.noticeEvidenceId).path : x.path;
      return `- [${md(x.title)}${retired ? '（原件已退役）' : ''}](${relative(dirname(MARKDOWN), resolve(ROOT, path)).split(sep).join('/')})`;
    }), '',
    '## 同步规则', '',
    '1. 每次阶段进展、状态变化、新证据、失败或阻塞，更新同一 JSON 状态，并同步本看板。',
    '2. 每次 Goal 恢复先读看板和详细状态，核验实际进程后再填进度，不根据旧 PID 或意图推断。',
    '3. 同步核验时间、当前位置、相应任务、下一道门槛、计数与证据入口。',
    '4. pilot、编译成功、静态清单和运行中数据不得写成正式通过；负收益和失败保留。',
    '5. 用户无需再次提醒。未运行网页服务时，更新状态后运行 `node tools/orin-task-board/sync.mjs`。', '',
    '## 最近变更', '', '| 时间 | 进展 | 说明 |', '|---|---|---|',
    ...data.events.map(x => `| ${md(new Date(x.at).toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai', hour12: false }))} | ${md(x.title)} | ${md(x.detail)} |`), ''
  ];
  return lines.join('\n');
}

export async function syncMarkdown(data, output = MARKDOWN) {
  data ??= await loadBoard();
  const text = renderMarkdown(data);
  const existing = await readFile(output, 'utf8').catch(() => null);
  if (existing === text) return false;
  const temporary = `${output}.${process.pid}.tmp`;
  await writeFile(temporary, text);
  await rename(temporary, output);
  return true;
}

export async function evidencePath(data, id, root = ROOT) {
  const entry = data.evidence.find(x => x.id === id);
  if (!entry || !['.md', '.json', '.csv', '.png', '.pdf', '.xml'].includes(extname(entry.path).toLowerCase())) return null;
  const base = await realpath(root), path = await realpath(resolve(root, entry.path)).catch(() => null);
  if (!path || !path.startsWith(base + sep) || !(await stat(path)).isFile()) return null;
  return path;
}

export function retiredEvidence(data, id) {
  const entry = data.evidence.find(item => item.id === id);
  const retirement = data.project.evidenceRetirement;
  return entry && retirement?.roots.some(root => entry.path === root || entry.path.startsWith(root + '/'))
    ? retirement : null;
}
