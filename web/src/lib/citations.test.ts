import assert from "node:assert/strict";
import { test } from "node:test";

import { segments } from "./citations.ts";

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
