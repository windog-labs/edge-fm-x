import { defineConfig } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { ROOT } from './board.mjs';

let launch = { url: `http://127.0.0.1:${process.env.PORT ?? 4317}` };
try {
  launch = JSON.parse(readFileSync(resolve(ROOT, 'artifacts/orin-task-board/launch.json'), 'utf8'));
} catch (error) { if (error.code !== 'ENOENT') throw error; }
const baseURL = process.env.BOARD_URL ?? launch.url;
export default defineConfig({
  testDir: './test/browser',
  workers: 1,
  timeout: 30000,
  use: { baseURL, channel: 'chromium', headless: true, launchOptions: { args: ['--no-sandbox'] }, trace: 'retain-on-failure' },
  webServer: process.env.BOARD_URL ? undefined : {
    command: 'node server.mjs', cwd: resolve(ROOT, 'tools/orin-task-board'),
    url: baseURL, reuseExistingServer: true, env: { PORT: new URL(baseURL).port || '4317' }
  },
  reporter: [['list'], ['junit', { outputFile: resolve(ROOT, 'artifacts/orin-task-board/browser-tests.xml') }]],
  outputDir: './test-results'
});
