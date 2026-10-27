import { test, expect } from '@playwright/test';
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { ROOT } from '../../board.mjs';

const screenshots = join(ROOT, 'artifacts/orin-task-board/screenshots');
test.beforeEach(async ({ page }) => { await page.goto('/'); await expect(page.getByRole('heading', { name: /Orin 实验看板/ })).toBeVisible(); });

test('board, search, filters, detail and list workflow', async ({ page }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const data = await (await page.request.get('/api/board')).json();
  await expect(page.locator('.task-card')).toHaveCount(data.tasks.length);
  await page.getByRole('button', { name: '查看任务：RDT 三模式正式采样' }).click();
  await expect(page.locator('dialog')).toBeVisible();
  await expect(page.locator('dialog').getByText('112/112 个原生输入张量逐字节一致', { exact: true })).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.locator('dialog')).not.toBeVisible();
  const searchData = structuredClone(data);
  const marker = 'board-search-contract-probe';
  searchData.tasks.find(task => task.id === 'rdt-formal').summary += ` ${marker}`;
  const commonTask = searchData.tasks.find(task => task.id === 'memory');
  commonTask.summary = 'RDT resource observation fixture';
  commonTask.remaining.push(marker);
  await page.route('**/api/board', route => route.fulfill({ json: searchData }));
  await page.getByRole('button', { name: '刷新本地状态', exact: true }).click();
  await page.getByRole('searchbox').fill('RDT');
  for (const task of data.tasks.filter(task => task.model === 'rdt')) {
    await expect(page.locator(`.task-card[data-task="${task.id}"]`)).toBeVisible();
  }
  await expect(page.locator('.task-card[data-task="memory"]')).toBeVisible();
  await page.getByRole('searchbox').fill(marker);
  await expect(page.locator('.task-card')).toHaveCount(2);
  await expect(page.locator('.task-card[data-task="rdt-formal"]')).toBeVisible();
  await expect(page.locator('.task-card[data-task="memory"]')).toBeVisible();
  await page.getByRole('searchbox').fill('不存在的任务');
  await expect(page.locator('.task-card')).toHaveCount(0);
  await page.getByRole('searchbox').fill('');
  await page.getByRole('button', { name: '已完成', exact: true }).click();
  await expect(page.locator('.task-card')).toHaveCount(data.tasks.filter(x => x.status === 'done').length);
  await page.getByRole('button', { name: '全部', exact: true }).click();
  await page.getByRole('checkbox', { name: '包含可选项' }).uncheck();
  await expect(page.locator('.task-card')).toHaveCount(data.tasks.filter(x => !x.optional).length);
  await page.getByRole('combobox', { name: '按模型筛选' }).selectOption('pi05');
  await expect(page.locator('.task-card')).toHaveCount(1);
  await page.getByRole('combobox', { name: '按模型筛选' }).selectOption('all');
  await page.getByRole('button', { name: '列表视图', exact: true }).click();
  await expect(page.locator('.task-table tbody tr')).toHaveCount(data.tasks.filter(x => !x.optional).length);
  expect(errors).toEqual([]);
});

test('matrix and evidence preserve formal/pilot distinctions and actual images', async ({ page }) => {
  const fixture = structuredClone(await (await page.request.get('/api/board')).json());
  fixture.project.currentModel = 'rdt';
  fixture.project.status = 'running';
  const pilot = fixture.models.find(model => model.id === 'rdt');
  Object.assign(pilot, { fidelity: 'pilot-passed', formal: 'running' });
  delete pilot.metrics;
  delete fixture.run.auditStatus;
  await page.route('**/api/board', route => route.fulfill({ json: fixture }));
  await page.getByRole('button', { name: '刷新本地状态', exact: true }).click();
  await page.getByRole('button', { name: '模型矩阵', exact: true }).click();
  await expect(page.locator('.model-table tbody tr')).toHaveCount(7);
  await expect(page.locator('.current-model')).toContainText('pilot 通过');
  await expect(page.locator('.current-model')).toContainText('尚无正式结果');
  await expect(page.getByRole('button', { name: '已完成', exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: '证据与限制', exact: true }).click();
  await expect(page.getByRole('heading', { name: /^抖动目标未达成/ })).toBeVisible();
  await expect(page.getByRole('combobox')).toHaveCount(0);
  const image = page.getByAltText(/三模式完整 E2E 时延/);
  await image.scrollIntoViewIfNeeded();
  await expect.poll(() => image.evaluate(element => element.complete && element.naturalWidth > 0)).toBe(true);
  await page.getByRole('searchbox').fill('2630');
  await expect(page.locator('.evidence-item')).toHaveCount(1);
});

test('refresh uses new local state while retaining the selected task view', async ({ page }) => {
  const original = await (await page.request.get('/api/board')).json();
  await page.getByRole('button', { name: '列表视图', exact: true }).click();
  const data = structuredClone(original);
  data.run.measuredInProcess = Math.min(data.run.measuredInProcess + 1, 1024);
  await page.route('**/api/board', route => route.fulfill({ json: data }));
  await page.getByRole('button', { name: '刷新本地状态', exact: true }).click();
  await expect(page.locator('.run-progress strong')).toHaveText(`${data.run.measuredInProcess.toLocaleString('en-US')} / 1,024`);
  await expect(page.locator('.task-table')).toBeVisible();
});

test('cancelled optional model stays visible without claiming formal completion', async ({ page }) => {
  const data = structuredClone(await (await page.request.get('/api/board')).json());
  data.tasks.find(task => task.id === 'qwen2').status = 'cancelled';
  const model = data.models.find(item => item.id === 'qwen2');
  model.deployment = model.fidelity = model.formal = 'cancelled';
  await page.route('**/api/board', route => route.fulfill({ json: data }));
  await page.getByRole('button', { name: '刷新本地状态', exact: true }).click();
  const card = page.locator('.task-card[data-task="qwen2"]');
  await expect(card).toBeVisible();
  await expect(card.locator('.status-badge')).toHaveText('用户取消');
  await page.getByRole('button', { name: '模型矩阵', exact: true }).click();
  const row = page.locator('.model-table tbody tr').filter({ hasText: 'Qwen3.5 2B' });
  await expect(row).toContainText('用户取消，不测试');
  await expect(row).not.toContainText('正式通过');
  await page.locator('[data-view="overview"]').click();
  await page.getByRole('button', { name: '列表视图', exact: true }).click();
  const cancelledRow = page.locator('.task-table tbody tr').filter({ hasText: 'Qwen3.5 2B' });
  await expect(cancelledRow).toContainText('用户取消，不测试；不计通过');
});

test('terminal collection stays distinct from active sampling and formal acceptance', async ({ page }) => {
  const data = await (await page.request.get('/api/board')).json();
  delete data.run.auditLabel;
  data.run.taskId = 'rdt-formal';
  delete data.run.auditStatus;
  data.run.status = 'failed';
  data.models.find(model => model.id === 'rdt').formal = 'blocked';
  await page.route('**/api/board', route => route.fulfill({ json: data }));
  await page.getByRole('button', { name: '刷新本地状态', exact: true }).click();
  await expect(page.locator('.run-description p')).toContainText('采集已停止');
  await expect(page.locator('.run-progress small')).toContainText('非完整正式数据');
  await expect(page.locator('[data-pipeline="rdt"] small')).toHaveText('需处理');
  await page.getByRole('button', { name: '查看任务：同平台算子调优消融' }).click();
  await expect(page.locator('dialog .run-metadata')).toHaveCount(0);
  await page.getByRole('button', { name: '关闭详情', exact: true }).click();
  await page.getByRole('button', { name: '查看任务：RDT 三模式正式采样' }).click();
  await expect(page.locator('dialog .run-metadata')).toContainText('末次采集快照');
  await page.getByRole('button', { name: '关闭详情', exact: true }).click();
  data.run.status = 'collected-independent-audit-pending';
  data.models.find(model => model.id === 'rdt').formal = 'waiting';
  await page.getByRole('button', { name: '刷新本地状态', exact: true }).click();
  await expect(page.locator('.run-description p')).toContainText('采集已结束');
  await expect(page.locator('.run-progress small')).toContainText('等待独立审计');
  await expect(page.locator('[data-pipeline="rdt"] small')).toHaveText('待独立审计');
  data.run.auditStatus = 'passed';
  data.models.find(model => model.id === 'rdt').formal = 'done';
  data.project.currentModel = 'pi0';
  await page.getByRole('button', { name: '刷新本地状态', exact: true }).click();
  await expect(page.locator('.run-description p')).toContainText('正式数据与服务恢复已审计');
  await expect(page.locator('.run-progress small')).toContainText('正式审计通过');
  await expect(page.locator('.metric > span').nth(2)).toHaveText('RDT-1B 正式进程');
  await expect(page.locator('[data-pipeline="rdt"] small')).toHaveText('正式已审计');
});

test('live metrics use the active task model and declared configuration count', async ({ page }) => {
  const data = await (await page.request.get('/api/board')).json();
  const task = data.tasks.find(item => item.id === data.run.taskId);
  const model = data.models.find(item => item.id === task.model);
  await expect(page.locator('.metric > span').nth(2)).toHaveText(`${model.name} 正式进程`);
  await expect(page.locator('.metric strong').nth(2)).toContainText(`/ ${data.run.totalProcesses}`);
  await expect(page.locator('.run-description h2')).toHaveText(data.run.label);
  if (data.run.protocolConfig) expect(data.run.totalProcesses).toBe(data.run.protocolConfig.modes.length * 5);
});

test('small-mobile: long experiment keys wrap without hiding evidence', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 780 });
  const data = await (await page.request.get('/api/board')).json();
  const key = 'same_effective_precision_main_table_qualified=false';
  data.events[0].detail = `${key} ${'0123456789abcdef'.repeat(4)}`;
  await page.route('**/api/board', route => route.fulfill({ json: data }));
  await page.reload();
  const detail = page.locator('.activity-row p').first();
  await expect(detail).toContainText(key);
  await detail.scrollIntoViewIfNeeded();
  const overflow = await detail.evaluate(element => ({ scroll: element.scrollWidth, width: element.clientWidth }));
  expect(overflow.scroll).toBeLessThanOrEqual(overflow.width + 1);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await mkdir(screenshots, { recursive: true });
  await page.screenshot({ path: join(screenshots, 'small-mobile-long-experiment-key.png') });
});

for (const [name, width, height] of [['desktop', 1440, 1000], ['tablet', 768, 1024], ['mobile', 390, 844], ['small-mobile', 320, 780]]) {
  test(`${name}: nonblank layout, fitting titles, no page overflow`, async ({ page }) => {
    await page.setViewportSize({ width, height });
    await page.reload();
    await expect(page.locator('.task-card').first()).toBeVisible();
    const pageOverflow = await page.locator('body *').evaluateAll(elements => elements.filter(element => {
      const rect = element.getBoundingClientRect();
      return rect.width > 0 && (rect.right > window.innerWidth + 1 || element.scrollWidth > element.clientWidth + 1);
    }).slice(0, 12).map(element => ({ tag: element.tagName, className: element.className, width: element.clientWidth, scrollWidth: element.scrollWidth, text: element.textContent.slice(0, 160) })));
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), JSON.stringify(pageOverflow)).toBe(true);
    const overflow = await page.locator('.card-title, h1, .run-description h2, .nav-item, .metric strong').evaluateAll(elements => elements.filter(element => element.scrollWidth > element.clientWidth + 1).map(element => element.textContent));
    expect(overflow).toEqual([]);
    expect(await page.locator('svg.lucide').count()).toBeGreaterThan(20);
    await mkdir(screenshots, { recursive: true });
    await page.screenshot({ path: join(screenshots, `${name}.png`) });
    await page.getByRole('button', { name: '查看任务：RDT 三模式正式采样' }).click();
    const dialog = page.locator('dialog');
    await expect(dialog).toBeVisible();
    const box = await dialog.boundingBox();
    expect(box.width).toBeLessThanOrEqual(width);
    expect(box.x).toBeGreaterThanOrEqual(0);
    await page.getByRole('button', { name: '关闭详情', exact: true }).click();
    await page.getByRole('button', { name: '证据与限制', exact: true }).click();
    await page.locator('.risk-row').first().scrollIntoViewIfNeeded();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    const evidenceOverflow = await page.locator('.risk-row > div, .risk-row h3, .risk-row p, .evidence-item code').evaluateAll(elements => elements.filter(element => element.clientWidth > 0 && element.scrollWidth > element.clientWidth + 1).map(element => element.textContent));
    expect(evidenceOverflow).toEqual([]);
    await page.screenshot({ path: join(screenshots, `${name}-risks.png`) });
  });
}
