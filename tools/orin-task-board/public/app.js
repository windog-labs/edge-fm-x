const $ = selector => document.querySelector(selector);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const icon = name => `<i data-lucide="${name}" aria-hidden="true"></i>`;
const number = value => Number(value).toLocaleString('en-US');
const labels = { running: '进行中', waiting: '等待依赖', planned: '待开始', done: '已完成', blocked: '需处理', cancelled: '用户取消' };
const state = { data: null, view: ['matrix', 'evidence'].includes(location.hash.slice(1)) ? location.hash.slice(1) : 'overview', layout: 'board', status: 'all', model: 'all', query: '', optional: true, task: null, error: false };
let lastPayload = '', refreshPromise;
const time = value => new Intl.DateTimeFormat('zh-CN', { timeZone: 'Asia/Shanghai', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false }).format(new Date(value));
const statusBadge = status => `<span class="status-badge ${status}">${icon(status === 'done' ? 'check' : status === 'running' ? 'activity' : status === 'blocked' ? 'alert-triangle' : status === 'cancelled' ? 'ban' : 'clock')}${labels[status] ?? esc(status)}</span>`;
function icons() { window.lucide?.createIcons({ attrs: { 'stroke-width': 1.7 } }); }
function evidenceLink(id) {
  const item = state.data.evidence.find(x => x.id === id);
  return item ? `<a class="evidence-link" href="/evidence/${encodeURIComponent(id)}" target="_blank" rel="noopener">${icon(item.kind === 'image' ? 'image' : 'file-text')}<span>${esc(item.title)}</span>${icon('arrow-up-right')}</a>` : '';
}
function modelName(id) { return state.data.models.find(x => x.id === id)?.name ?? '公共实验'; }
function filteredTasks() {
  return state.data.tasks.filter(task => (state.optional || !task.optional) && (state.status === 'all' || task.status === state.status)
    && (state.model === 'all' || task.model === state.model) && `${task.title} ${task.summary} ${task.stage} ${modelName(task.model)} ${task.remaining.join(' ')}`.toLowerCase().includes(state.query.toLowerCase()));
}
function filteredModels() {
  return state.data.models.filter(x => (state.optional || !x.optional) && (state.model === 'all' || x.id === state.model) && `${x.name} ${x.profile}`.toLowerCase().includes(state.query.toLowerCase()));
}
function card(task) {
  return `<button class="task-card ${task.status === 'running' ? 'active-task' : ''}" data-task="${esc(task.id)}" aria-label="查看任务：${esc(task.title)}">
    <span class="card-meta"><span class="task-code">${esc(task.stage)}</span><span class="priority ${task.priority.toLowerCase()}">${task.priority}</span>${task.optional ? '<span class="optional-badge">可选</span>' : ''}</span>
    <span class="card-title">${esc(task.title)}</span><span class="card-summary">${esc(task.summary)}</span>
    <span class="card-footer"><span class="model-tag">${icon(task.model === 'common' ? 'layers' : 'box')}${esc(modelName(task.model))}</span>${task.status === 'cancelled' ? statusBadge(task.status) : ''}<span class="evidence-count" title="${task.evidenceIds.length} 份证据">${icon('paperclip')}${task.evidenceIds.length}</span></span>
  </button>`;
}
function overview() {
  const tasks = filteredTasks();
  if (state.layout === 'list') return `<div class="table-scroll"><table class="task-table"><thead><tr><th>任务</th><th>阶段</th><th>模型</th><th>状态</th><th>下一道门槛</th></tr></thead><tbody>${tasks.map(task => `<tr><td><button class="text-link" data-task="${esc(task.id)}">${esc(task.title)}</button>${task.optional ? '<span class="optional-badge">可选</span>' : ''}</td><td>${esc(task.stage)}</td><td>${esc(modelName(task.model))}</td><td>${statusBadge(task.status)}</td><td class="next-cell">${esc(task.remaining[0] ?? (task.status === 'cancelled' ? '用户取消，不测试；不计通过' : '已完成限定范围验收'))}</td></tr>`).join('')}</tbody></table>${!tasks.length ? '<div class="empty-state">没有匹配的任务</div>' : ''}</div>`;
  const columns = ['running', 'waiting', 'planned', 'done'];
  return `<div class="kanban">${columns.map(status => {
    const items = tasks.filter(task => task.status === status || (status === 'waiting' && task.status === 'blocked') || (status === 'done' && task.status === 'cancelled'));
    const title = status === 'done' && tasks.some(task => task.status === 'cancelled') ? '已结束' : labels[status];
    return `<section class="kanban-column"><div class="column-heading"><span class="column-name"><span class="status-dot ${status}"></span>${title}</span><span class="column-count">${items.length}</span></div><div class="column-body">${items.length ? items.map(card).join('') : '<div class="empty-column">暂无任务</div>'}</div></section>`;
  }).join('')}</div>`;
}
function matrix() {
  const models = filteredModels();
  const fidelity = { 'formal-passed': ['done', '正式通过'], 'pilot-passed': ['running', 'pilot 通过'], planned: ['planned', '未开始'], cancelled: ['cancelled', '用户取消，不测试'] };
  return `<div class="section-heading"><div><h2>模型验收矩阵</h2><p>同一 Orin、同精度的完整部署对照</p></div><span class="subtle">5 个必需 VLA · ${state.data.models.filter(model => model.optional && model.formal !== 'cancelled').length} 个 VLM 收尾</span></div>
    <div class="table-scroll"><table class="model-table"><thead><tr><th>模型 / 配置</th><th>完整链</th><th>动作核验</th><th>正式 E2E / CDF</th><th>常规 / Replay 均值</th><th>下一步</th></tr></thead><tbody>${models.map(model => {
      const [qualityStatus, qualityText] = fidelity[model.fidelity];
      return `<tr class="${model.id === state.data.project.currentModel ? 'current-model' : ''}"><td><strong>${esc(model.name)}</strong>${model.optional ? '<span class="optional-badge">可选</span>' : ''}<small>${esc(model.profile)}</small></td><td>${model.engines ? `<strong>${model.engines} engines</strong><small>全部 ${model.steps} 步</small>` : '<span class="subtle">待评估</span>'}</td><td><span class="status-badge ${qualityStatus}">${esc(qualityText)}</span></td><td>${statusBadge(model.formal)}</td><td class="numeric">${model.metrics ? `${model.metrics.baselineMs.toFixed(3)} / ${model.metrics.replayMs.toFixed(3)}<small>ms · ${model.metrics.speedup.toFixed(3)}x</small>` : '<span class="subtle">尚无正式结果</span>'}</td><td class="next-cell">${esc(model.next)}<div class="cell-evidence">${model.evidenceIds.length ? evidenceLink(model.evidenceIds[0]) : ''}</div></td></tr>`;
    }).join('')}</tbody></table></div>
    <div class="boundary-note">${icon('shield-check')}<span>${esc(state.data.project.acceptance)}</span></div>`;
}
function evidence() {
  const d = state.data;
  const items = d.evidence.filter(x => `${x.title} ${x.kind} ${x.path}`.toLowerCase().includes(state.query.toLowerCase()));
  return `<div class="section-heading"><div><h2>证据与验收边界</h2><p>原始记录、正式结果和仍需闭合的主张</p></div></div>
    <div class="risk-list">${d.risks.map(risk => `<div class="risk-row"><span class="risk-icon ${risk.level}">${icon(risk.level === 'warning' ? 'alert-triangle' : 'info')}</span><div><h3>${esc(risk.title)} <span class="risk-value">${esc(risk.value)}</span></h3><p>${esc(risk.detail)}</p></div></div>`).join('')}</div>
    <div class="section-heading evidence-section"><h2>已交付曲线</h2><span class="subtle">论文交接包原始图像</span></div>
    ${(d.project.evidenceFigures ?? []).map((figure, index) => `<figure class="cdf-figure ${index ? 'processes-figure' : ''}"><a href="/evidence/${encodeURIComponent(figure.evidenceId)}" target="_blank" rel="noopener"><img src="/evidence/${encodeURIComponent(figure.evidenceId)}" alt="${esc(figure.alt)}" loading="lazy"></a><figcaption>${esc(figure.caption)}</figcaption></figure>`).join('')}
    <div class="section-heading evidence-section"><h2>报告与原始证据</h2><span class="subtle">${items.length} 项</span></div>
    <div class="evidence-grid">${items.map(item => `<div class="evidence-item"><span class="file-kind">${esc(item.kind.toUpperCase())}</span>${evidenceLink(item.id)}<code>${esc(item.path)}</code></div>`).join('')}</div>`;
}
function renderBody() {
  $('#work-content').innerHTML = state.view === 'matrix' ? matrix() : state.view === 'evidence' ? evidence() : overview();
  const count = filteredTasks().length;
  $('#task-count').textContent = state.view === 'overview' ? `${count} 项任务` : state.view === 'matrix' ? `${filteredModels().length} 个模型` : '审计与原始记录';
  icons();
}
function render() {
  const d = state.data, required = d.models.filter(x => !x.optional), run = d.run;
  const focus = document.activeElement?.id, selection = focus === 'search-input' ? $('#search-input')?.selectionStart : null;
  const done = required.filter(x => x.formal === 'done').length;
  const pending = d.tasks.filter(x => x.priority === 'P0' && x.status !== 'done').length;
  $('#app').innerHTML = `<div class="app-frame"><aside class="sidebar"><div class="brand"><span class="brand-mark">${icon('circuit-board')}</span><div><strong>EdgeFM</strong><span>研究实验 / ICRA 2027</span></div></div>
    <div class="workspace-label">ORIN WORKSPACE</div><nav aria-label="主要视图">${[['overview', 'layout-dashboard', '任务总览'], ['matrix', 'table-2', '模型矩阵'], ['evidence', 'files', '证据与限制']].map(([view, glyph, title]) => `<button class="nav-item ${view === state.view ? 'selected' : ''}" data-view="${view}" aria-current="${view === state.view ? 'page' : 'false'}">${icon(glyph)}<span>${title}</span>${view === 'overview' ? `<span class="nav-count">${d.tasks.length}</span>` : ''}</button>`).join('')}</nav>
    <div class="sidebar-device"><span class="device-icon">${icon('cpu')}</span><strong>${esc(d.project.device)}</strong><span>TRT 8.5.2.2 · CUDA 11.4</span><span>MAXN · 单台架顺序执行</span><div class="sidebar-scope">5 VLA <span>+</span> 2 可选 VLM</div></div></aside>
    <main class="workspace"><header class="page-header"><div><div class="breadcrumb">EdgeFM / 部署实验</div><h1>Orin 实验看板 <span class="project-status">${icon(d.project.status === 'done' ? 'check' : 'activity')}${esc(d.project.status === 'paused' ? '已暂停' : labels[d.project.status])}</span></h1></div><div class="header-actions"><span class="verification-time">${icon('clock')}核验 ${time(d.project.verifiedAt)}</span><a class="icon-button" href="/evidence/task-board" target="_blank" rel="noopener" aria-label="打开 Markdown 看板" data-tooltip="Markdown 看板">${icon('file-text')}</a><a class="icon-button" href="/api/board" download="edgefm-orin-board.json" aria-label="下载状态 JSON" data-tooltip="下载状态">${icon('download')}</a><button class="icon-button" data-refresh aria-label="刷新本地状态" data-tooltip="刷新本地状态">${icon('refresh-cw')}</button></div></header>
    <div class="error-banner ${state.error ? '' : 'hidden'}" role="alert">${icon('alert-triangle')}状态数据暂不可用，保留最近一次核验结果。</div>
    <section class="metrics-band" aria-label="总体验收进度"><div class="metric"><span>正式 E2E / CDF</span><strong>${done}<em>/ 5</em></strong><small>模型覆盖，不是工时百分比</small></div><div class="metric"><span>完整链真机 pilot</span><strong>${required.filter(x => x.deployment === 'passed').length}<em>/ 5</em></strong><small>SmolVLA、RDT 已验证</small></div><div class="metric"><span>RDT 正式进程</span><strong>${run.completedProcesses}<em>/ ${run.totalProcesses}</em></strong><small>每模式 5 个独立进程</small></div><div class="metric"><span>待闭合核心任务</span><strong>${pending}<em>项</em></strong><small>五模型及公共消融仍未完成</small></div></section>
    <section class="run-strip" aria-label="当前运行"><div class="run-description"><span class="eyebrow">当前所在位置 <span>第二个必需模型</span></span><h2>${esc(run.label)}</h2><p>第 ${run.round} 轮 · ${esc(run.mode)} · 已预热 ${run.warmupInProcess} 次</p></div><div class="run-progress"><div><span>当前进程 measured</span><strong>${number(run.measuredInProcess)} <span>/ ${number(run.plannedInProcess)}</span></strong></div><progress value="${run.measuredInProcess}" max="${run.plannedInProcess}" aria-label="当前进程正式测量进度"></progress><small>整组 ${number(run.measured)} / ${number(run.plannedMeasured)} 条测量 · ${run.qualityFlagsPassed ? '已记录输出核验正常' : '输出待检查'}</small></div><button class="run-detail" data-task="${esc(d.project.currentTaskId)}" aria-label="查看当前任务详情">当前任务 ${icon('arrow-up-right')}</button></section>
    <section class="model-pipeline" aria-label="必需模型执行顺序">${required.map((model, index) => `<button class="model-step ${model.formal === 'done' ? 'done' : model.id === d.project.currentModel ? 'current' : ''}" data-pipeline="${model.id}" aria-label="查看 ${esc(model.name)} 模型状态"><span class="step-index">${model.formal === 'done' ? icon('check') : index + 1}</span><span><strong>${esc(model.name)}</strong><small>${model.formal === 'done' ? '正式已审计' : model.id === d.project.currentModel ? '正式采样中' : '待支持评估'}</small></span>${index < required.length - 1 ? icon('chevron-right') : ''}</button>`).join('')}</section>
    <section class="task-workspace"><div class="workspace-toolbar"><div class="section-title"><h2>${state.view === 'overview' ? '任务队列' : state.view === 'matrix' ? '模型验收' : '实验交付'}</h2><span id="task-count"></span></div><div class="workspace-controls"><label class="search-field">${icon('search')}<input id="search-input" type="search" placeholder="搜索任务或模型" value="${esc(state.query)}" aria-label="搜索任务或模型"></label><select id="model-filter" aria-label="按模型筛选"><option value="all">全部模型</option><option value="common" ${state.model === 'common' ? 'selected' : ''}>公共实验</option>${d.models.map(x => `<option value="${x.id}" ${state.model === x.id ? 'selected' : ''}>${esc(x.name)}</option>`).join('')}</select>${state.view === 'overview' ? `<div class="view-switch" role="group" aria-label="任务布局"><button class="icon-button ${state.layout === 'board' ? 'selected' : ''}" data-layout="board" aria-label="看板视图" data-tooltip="看板视图">${icon('columns-3')}</button><button class="icon-button ${state.layout === 'list' ? 'selected' : ''}" data-layout="list" aria-label="列表视图" data-tooltip="列表视图">${icon('list')}</button></div>` : ''}</div></div>
    <div class="filter-row"><div class="status-filters" role="group" aria-label="按状态筛选">${[['all', '全部'], ['running', '进行中'], ['waiting', '等待依赖'], ['planned', '待开始'], ['done', '已完成']].map(([value, label]) => `<button class="filter-button ${state.status === value ? 'selected' : ''}" data-filter="${value}" aria-pressed="${state.status === value}">${label}</button>`).join('')}</div><label class="optional-toggle"><input id="optional-toggle" type="checkbox" ${state.optional ? 'checked' : ''}>包含可选项</label></div>
    <div id="work-content"></div></section>
    <section class="activity-section"><div class="section-heading"><h2>最近进展</h2><a class="text-link" href="/evidence/goal-status" target="_blank" rel="noopener">详细记录 ${icon('arrow-up-right')}</a></div><div class="activity-list">${d.events.slice(0, 5).map(event => `<div class="activity-row"><time>${time(event.at)}</time><div><strong>${esc(event.title)}</strong><p>${esc(event.detail)}</p></div></div>`).join('')}</div></section>
    <footer class="page-footer"><span>目标仍未完成 · H20 差异单列补充</span><button class="text-link" data-risks>验收口径与风险 ${icon('arrow-up-right')}</button></footer></main></div><dialog id="task-dialog" class="task-dialog"></dialog>`;
  renderBody();
  $('.sidebar-scope').innerHTML = `5 VLA <span>+</span> ${d.models.filter(model => model.optional && model.formal !== 'cancelled').length} VLM`;
  const projectLabels = { running: '进行中', paused: '已暂停', blocked: '需处理', done: '已完成' };
  $('.project-status').innerHTML = `${icon(d.project.status === 'done' ? 'circle-check' : d.project.status === 'paused' ? 'pause' : 'activity')}${projectLabels[d.project.status] ?? '进行中'}`;
  const metricLabels = document.querySelectorAll('.metric > span');
  const runModel = d.tasks.find(task => task.id === run.taskId)?.model ?? d.project.currentModel;
  metricLabels[2].textContent = `${modelName(runModel)} 正式进程`;
  document.querySelectorAll('.metric small')[1].textContent = required.filter(x => x.deployment === 'passed').map(x => x.name).join('、') + ' 已验证';
  metricLabels[3].textContent = required.every(model => model.formal === 'done') ? '待闭合收尾任务' : '待闭合核心任务';
  document.querySelectorAll('.metric small')[3].textContent = pending
    ? required.every(model => model.formal === 'done') ? '最后补充模型待收尾' : '五模型及公共消融仍未完成'
    : '核心任务已闭合';
  const index = required.findIndex(x => x.id === d.project.currentModel);
  $('.run-description .eyebrow > span').textContent = index >= 0 ? `第 ${index + 1} 个必需模型` : '公共实验';
  document.querySelectorAll('.model-step small').forEach((element, i) => {
    element.textContent = { done: '正式已审计', running: '正式采样中', waiting: '待独立审计', blocked: '需处理', planned: '待支持评估' }[required[i].formal] ?? '待核验';
  });
  if (run.status !== 'running') {
    const accepted = run.auditStatus === 'passed';
    $('.run-description p').textContent = `${accepted ? run.auditLabel ?? '正式数据与服务恢复已审计' : run.status === 'failed' ? '采集已停止' : '采集已结束'} · 末次第 ${run.round} 轮 · ${run.mode}`;
    if (accepted) $('.run-description .eyebrow').textContent = run.auditLabel ? '末次已审计采集' : '末次正式采集';
    $('.run-progress > div > span').textContent = '末进程已保留 measured';
    $('.run-progress small').textContent = `保留 ${number(run.measured)} / ${number(run.plannedMeasured)} 条记录 · ${accepted ? run.auditLabel ?? '正式审计通过' : run.status === 'failed' ? '非完整正式数据，待核验' : '等待独立审计'}`;
  }
  $('.page-footer > span').textContent = d.project.status === 'done' ? '必需验收已完成 · H20 差异单列补充' : '目标仍未完成 · H20 差异单列补充';
  if (state.view !== 'overview') $('.status-filters')?.remove();
  if (state.view === 'matrix') $('#model-filter option[value="common"]')?.remove();
  if (state.view === 'evidence') {
    $('#model-filter')?.remove();
    $('.filter-row')?.remove();
    $('#search-input').placeholder = '搜索报告或证据';
    $('#search-input').setAttribute('aria-label', '搜索报告或证据');
  }
  $('#search-input').addEventListener('input', event => { state.query = event.target.value; renderBody(); });
  $('#model-filter')?.addEventListener('change', event => { state.model = event.target.value; renderBody(); });
  $('#optional-toggle')?.addEventListener('change', event => { state.optional = event.target.checked; renderBody(); });
  if (focus === 'search-input') { $('#search-input').focus(); $('#search-input').setSelectionRange(selection, selection); }
  if (state.task) openTask(state.task);
  icons();
}
function modal(title, body, eyebrow = '') {
  const dialog = $('#task-dialog');
  dialog.innerHTML = `<div class="dialog-header"><div><span class="eyebrow">${esc(eyebrow)}</span><h2>${esc(title)}</h2></div><button class="icon-button" data-close aria-label="关闭详情" data-tooltip="关闭">${icon('x')}</button></div><div class="dialog-body">${body}</div>`;
  if (!dialog.open) dialog.showModal();
  dialog.addEventListener('close', () => { state.task = null; }, { once: true });
  dialog.addEventListener('click', event => { if (event.target === dialog) { const box = dialog.getBoundingClientRect(); if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) dialog.close(); } });
  icons();
}
function openTask(id) {
  const task = state.data.tasks.find(x => x.id === id);
  if (!task) return;
  state.task = id;
  const list = (title, rows) => rows.length ? `<section class="detail-section"><h3>${title}</h3><ul>${rows.map(row => `<li>${esc(row)}</li>`).join('')}</ul></section>` : '';
  modal(task.title, `<div class="detail-status">${statusBadge(task.status)}<span class="priority ${task.priority.toLowerCase()}">${task.priority}</span><span class="model-tag">${esc(modelName(task.model))}</span></div><p class="detail-summary">${esc(task.summary)}</p>${list('已核验', task.verified)}${list('待完成', task.remaining)}${list('验收产物', task.deliverables)}${task.estimate ? `<section class="detail-section"><h3>依赖与耗时</h3><p>${esc(task.estimate)}</p>${task.dependsOn.length ? `<div class="dependency-list">${task.dependsOn.map(dependency => `<button class="text-link" data-task="${dependency}">${esc(state.data.tasks.find(x => x.id === dependency)?.title ?? dependency)} ${icon('arrow-up-right')}</button>`).join('')}</div>` : ''}</section>` : ''}<section class="detail-section"><h3>证据</h3>${task.evidenceIds.length ? task.evidenceIds.map(evidenceLink).join('') : '<p class="subtle">尚无 Orin 验证证据</p>'}</section>`, `${task.stage} · ${task.id}`);
  if (id === (state.data.run.taskId ?? state.data.project.currentTaskId)) {
    const run = state.data.run;
    const section = document.createElement('section');
    section.className = 'detail-section run-metadata';
    section.innerHTML = `<h3>${run.status === 'running' ? '当前运行快照' : '末次采集快照'}</h3><dl><div><dt>最近核验</dt><dd>${time(run.verifiedAt)}</dd></div><div><dt>已完成进程</dt><dd>${run.completedProcesses} / ${run.totalProcesses}</dd></div><div><dt>本进程测量</dt><dd>${number(run.measuredInProcess)} / ${number(run.plannedInProcess)}</dd></div><div><dt>累计测量</dt><dd>${number(run.measured)} / ${number(run.plannedMeasured)}</dd></div><div><dt>控制器 PID / start ticks</dt><dd>${run.controller.pid} / ${esc(run.controller.startTicks)}</dd></div><div><dt>worker PID / start ticks</dt><dd>${run.worker.pid} / ${esc(run.worker.startTicks)}</dd></div></dl><p>${esc(run.qualityNote)}</p><code>${esc(run.targetPath)}</code>`;
    $('.dialog-body').insertBefore(section, $('.dialog-body .detail-section'));
  }
}
async function refresh(force = false) {
  if (refreshPromise) return refreshPromise;
  refreshPromise = (async () => {
    try {
      const response = await fetch('/api/board', { cache: 'no-store' });
      if (!response.ok) throw new Error('Unavailable');
      const data = await response.json(); delete data.servedAt;
      const payload = JSON.stringify(data), changed = payload !== lastPayload;
      state.error = false;
      if (changed || force || !state.data) { state.data = data; lastPayload = payload; render(); }
      else $('.error-banner')?.classList.add('hidden');
    } catch {
      state.error = true;
      if (state.data) $('.error-banner')?.classList.remove('hidden');
      else $('#app').innerHTML = `<div class="loading-state">实验状态暂不可用<button class="icon-button" data-refresh aria-label="重试">${icon('refresh-cw')}</button></div>`;
      icons();
    }
  })();
  try { await refreshPromise; } finally { refreshPromise = null; }
}
document.addEventListener('click', event => {
  const target = event.target.closest('button');
  if (!target) return;
  if (target.hasAttribute('data-refresh')) { refresh(true); return; }
  if (!state.data) return;
  if (target.dataset.view) { state.view = target.dataset.view; state.query = ''; state.model = 'all'; state.status = 'all'; history.replaceState(null, '', '#' + state.view); render(); }
  if (target.dataset.layout) { state.layout = target.dataset.layout; render(); }
  if (target.dataset.filter) { state.status = target.dataset.filter; render(); }
  if (target.dataset.task) openTask(target.dataset.task);
  if (target.dataset.pipeline) { state.view = 'matrix'; state.model = target.dataset.pipeline; state.query = ''; history.replaceState(null, '', '#matrix'); render(); }
  if (target.hasAttribute('data-close')) $('#task-dialog').close();
  if (target.hasAttribute('data-risks')) modal('验收口径与风险', `<p class="detail-summary">${esc(state.data.project.acceptance)}</p>${state.data.risks.map(x => `<section class="detail-section"><h3>${esc(x.title)}</h3><p>${esc(x.detail)}</p></section>`).join('')}`, '五模型完整交付');
});
refresh();
setInterval(() => { if (!document.hidden) refresh(); }, 15000);
document.addEventListener('visibilitychange', () => { if (!document.hidden) refresh(); });
