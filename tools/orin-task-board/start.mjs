import { spawn } from 'node:child_process';
import { open, mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';
import { HERE, ROOT } from './board.mjs';
import { APP_ID } from './server.mjs';

const output = join(ROOT, 'artifacts/orin-task-board');
await mkdir(output, { recursive: true });
for (let port = 4317; port < 4327; port++) {
  const url = `http://127.0.0.1:${port}`;
  try {
    const health = await (await fetch(url + '/health', { signal: AbortSignal.timeout(500) })).json();
    if (health.app === 'edgefm-task-board' && health.id === APP_ID) { console.log(JSON.stringify({ url, pid: health.pid, reused: true })); process.exit(0); }
    continue;
  } catch {}
  const log = await open(join(output, `server-${port}.log`), 'a');
  const child = spawn(process.execPath, [join(HERE, 'server.mjs')], { cwd: HERE, env: { ...process.env, PORT: String(port) }, detached: true, stdio: ['ignore', log.fd, log.fd] });
  child.unref();
  await log.close();
  for (let count = 0; count < 30; count++) {
    await sleep(200);
    try {
      const health = await (await fetch(url + '/health', { signal: AbortSignal.timeout(500) })).json();
      if (health.app !== 'edgefm-task-board' || health.id !== APP_ID) break;
      const record = { url, pid: health.pid, reused: false, root: ROOT, startedAt: new Date().toISOString() };
      await writeFile(join(output, 'launch.json'), JSON.stringify(record, null, 2) + '\n');
      console.log(JSON.stringify(record));
      process.exit(0);
    } catch {}
  }
}
throw new Error('No task-board port available; existing services were left untouched');
