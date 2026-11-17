import assert from "node:assert/strict";
import { test } from "node:test";

import { AskError, ask } from "./api.ts";

function respond(status: number, body: string) {
  globalThis.fetch = (async () => new Response(body, { status })) as typeof fetch;
}

async function messageFor(status: number, body: string): Promise<string> {
  respond(status, body);
  try {
    await ask("what?");
  } catch (error) {
    assert.ok(error instanceof AskError);
    return error.message;
  }
  throw new Error("ask did not throw");
}

test("a server error with no JSON body does not blame the question", async () => {
  assert.equal(await messageFor(500, "Internal Server Error"), "Something went wrong. Please retry.");
});

test("a validation error asks the user to check the question", async () => {
  assert.equal(await messageFor(422, JSON.stringify({ detail: [{ msg: "too short" }] })), "Please check your question.");
});

test("a string detail from the server is shown as is", async () => {
  assert.equal(await messageFor(503, JSON.stringify({ detail: "The database is unavailable." })), "The database is unavailable.");
});
