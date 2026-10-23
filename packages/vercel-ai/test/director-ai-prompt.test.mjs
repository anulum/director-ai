// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-AI — public prompt extraction contracts

import assert from "node:assert/strict";
import test from "node:test";
import { extractDirectorAiPrompt } from "../dist/index.js";

test("extractDirectorAiPrompt supports AI SDK prompt arrays and text blocks", () => {
  assert.equal(extractDirectorAiPrompt({ prompt: [
    { role: "system", content: "System" },
    { role: "user", content: [{ type: "text", text: "Latest question" }] },
  ] }), "Latest question");
});

/** @type {[string, unknown, string][]} Public extraction inputs and exact expected text. */
const cases = [
  ["null parameters", null, ""],
  ["primitive parameters", 7, ""],
  ["missing prompt", {}, ""],
  ["direct prompt", { prompt: "Question?" }, "Question?"],
  ["latest user message", { prompt: [
    { role: "user", content: "Earlier" }, null,
    { role: "user", content: "Latest" }, { role: "assistant", content: "Reply" },
  ] }, "Latest"],
  ["absent user content", { prompt: [{ role: "user" }] }, ""],
  ["fallback messages", { prompt: [{ content: "System" }, "Text", 3] }, "System Text 3"],
  ["empty array", { prompt: [] }, ""],
  ["nested text", { prompt: [{ role: "user", content: ["A", [{ text: "B" }], null] }] }, "A B"],
  ["object text", { prompt: { text: "Body" } }, "Body"],
  ["object content", { prompt: { content: "Body" } }, "Body"],
  ["object fallback", { prompt: { kind: "structured" } }, '{"kind":"structured"}'],
  ["primitive prompt", { prompt: false }, "false"],
];
for (const [name, input, expected] of cases) {
  test("public prompt extraction handles " + name, () => {
    assert.equal(extractDirectorAiPrompt(input), expected);
  });
}
