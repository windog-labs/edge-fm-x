import { syncMarkdown, MARKDOWN } from './board.mjs';
console.log(JSON.stringify({ updated: await syncMarkdown(), path: MARKDOWN }));
