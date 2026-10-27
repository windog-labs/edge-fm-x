import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { watchFile, unwatchFile } from 'node:fs';
import { join, extname } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { HERE, ROOT, STATE, loadBoard, syncMarkdown, evidencePath, retiredEvidence } from './board.mjs';

export const APP_ID = createHash('sha256').update(ROOT).digest('hex').slice(0, 16);
const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.json': 'application/json; charset=utf-8', '.md': 'text/plain; charset=utf-8', '.xml': 'text/plain; charset=utf-8', '.csv': 'text/plain; charset=utf-8', '.png': 'image/png', '.pdf': 'application/pdf' };

export function createBoardServer({ state = STATE, root = ROOT } = {}) {
  return createServer(async (request, response) => {
    response.setHeader('Cache-Control', 'no-store');
    response.setHeader('X-Content-Type-Options', 'nosniff');
    response.setHeader('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'");
    if (!/^(127\.0\.0\.1|localhost|\[::1\])(?::\d+)?$/.test(request.headers.host ?? '')) { response.writeHead(403).end('Local access only'); return; }
    if (!['GET', 'HEAD'].includes(request.method)) { response.writeHead(405).end('Read-only'); return; }
    const url = new URL(request.url, 'http://localhost');
    const send = (payload, type, code = 200) => {
      response.writeHead(code, { 'Content-Type': type });
      response.end(request.method === 'HEAD' ? undefined : payload);
    };
    try {
      if (url.pathname === '/health') { send(JSON.stringify({ app: 'edgefm-task-board', id: APP_ID, pid: process.pid }), TYPES['.json']); return; }
      if (url.pathname === '/api/board') {
        const data = await loadBoard(state);
        send(JSON.stringify({ ...data, servedAt: new Date().toISOString() }), TYPES['.json']);
        return;
      }
      if (url.pathname.startsWith('/evidence/')) {
        const id = decodeURIComponent(url.pathname.slice('/evidence/'.length));
        const data = await loadBoard(state), path = await evidencePath(data, id, root);
        const retired = retiredEvidence(data, id);
        if (!path && retired) {
          send(`<!doctype html><meta charset="utf-8"><title>Evidence retired</title><h1>原始证据已退役</h1><p>该原机文件已按授权清理，历史实验结论未改变。</p><p><a href="/evidence/${encodeURIComponent(retired.noticeEvidenceId)}">查看清理记录与论文交接入口</a></p>`, TYPES['.html'], 410);
          return;
        }
        if (!path) { send('Evidence not available', TYPES['.md'], 404); return; }
        send(await readFile(path), TYPES[extname(path).toLowerCase()] ?? 'application/octet-stream');
        return;
      }
      const routes = { '/': 'index.html', '/index.html': 'index.html', '/app.js': 'app.js', '/styles.css': 'styles.css', '/vendor/lucide.js': '../node_modules/lucide/dist/umd/lucide.js' };
      const file = Object.hasOwn(routes, url.pathname) ? routes[url.pathname] : null;
      if (!file) { send('Not found', TYPES['.md'], 404); return; }
      const path = join(HERE, 'public', file);
      send(await readFile(path), TYPES[extname(path)]);
    } catch (error) {
      console.error(error.message);
      if (!response.headersSent) send(JSON.stringify({ error: '看板数据暂不可用，请重试；实验状态未被修改。' }), TYPES['.json'], 503);
      else response.end();
    }
  });
}

export function watchBoardState({ state = STATE, sync = syncMarkdown, interval = 1000, onError = error => console.error('State sync:', error.message) } = {}) {
  let pending, closed = false;
  const changed = (current, previous) => {
    if (closed || (current.mtimeMs === previous.mtimeMs && current.ctimeMs === previous.ctimeMs
      && current.ino === previous.ino && current.size === previous.size)) return;
    clearTimeout(pending);
    pending = setTimeout(() => {
      if (!closed) Promise.resolve().then(() => sync()).catch(onError);
    }, 150);
  };
  // Stat polling survives atomic replacement and does not consume inotify quota.
  watchFile(state, { interval }, changed);
  return { close() { closed = true; unwatchFile(state, changed); clearTimeout(pending); } };
}

async function main() {
  await syncMarkdown();
  const server = createBoardServer();
  const port = Number(process.env.PORT ?? 4317);
  const watcher = watchBoardState();
  const close = () => { watcher.close(); server.close(); };
  server.on('error', error => { console.error(error.message); close(); process.exitCode = 1; });
  server.listen(port, '127.0.0.1', () => console.log(`Local: http://127.0.0.1:${port}`));
  process.on('SIGTERM', close);
  process.on('SIGINT', close);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await main();
