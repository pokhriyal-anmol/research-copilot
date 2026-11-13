import assert from "node:assert/strict";
import { test } from "node:test";

import { wake } from "./api.ts";

function fakeFetch(results: Array<number | "throw">): typeof fetch {
  let i = 0;
  return (async () => {
    const r = results[Math.min(i++, results.length - 1)];
    if (r === "throw") throw new TypeError("network");
    return new Response(null, { status: r });
  }) as typeof fetch;
}

test("keeps asking while the server boots, then reports it awake", async () => {
  assert.equal(await wake({ fetcher: fakeFetch(["throw", 502, 200]), retryMs: 1 }), true);
});

test("gives up at the deadline when the server never answers", async () => {
  assert.equal(await wake({ fetcher: fakeFetch([503]), retryMs: 5, deadlineMs: 30 }), false);
});
