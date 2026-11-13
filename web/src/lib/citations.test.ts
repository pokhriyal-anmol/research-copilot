import assert from "node:assert/strict";
import { test } from "node:test";

import { blocks, segments } from "./citations.ts";

test("splits text around single and grouped markers", () => {
  assert.deepEqual(segments("BM25 is strong [1]. Dense helps [2, 3]."), [
    { text: "BM25 is strong " },
    { cite: [1] },
    { text: ". Dense helps " },
    { cite: [2, 3] },
    { text: "." },
  ]);
});

test("leaves text without markers and non-numeric brackets alone", () => {
  assert.deepEqual(segments("No citations [here]."), [{ text: "No citations [here]." }]);
});

test("splits an answer into paragraphs and list items without markdown", () => {
  assert.deepEqual(blocks("Ties matter:\n\n*   **Within** top $k$ [1].\n* Stable sort [4].\n\nSo report it [5]."), [
    { kind: "p", text: "Ties matter:" },
    { kind: "li", text: "Within top k [1]." },
    { kind: "li", text: "Stable sort [4]." },
    { kind: "p", text: "So report it [5]." },
  ]);
});

test("keeps a one-line answer as one paragraph", () => {
  assert.deepEqual(blocks("BM25 is strong [1]."), [{ kind: "p", text: "BM25 is strong [1]." }]);
});
