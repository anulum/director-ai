// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-AI — loopback review transport fixture

import assert from "node:assert/strict";
import { once } from "node:events";
import { createServer } from "node:http";

/**
 * @typedef {object} ReviewRequest
 * @property {string | undefined} url Received review path.
 * @property {string | undefined} method Received HTTP method.
 * @property {import("node:http").IncomingHttpHeaders} headers Actual request headers.
 * @property {unknown} body Parsed original request body.
 */

/**
 * Bind a real loopback review endpoint and close its connections after the test.
 * Decisions are protocol inputs, without qualifying an AI inference result.
 * @param {import("node:test").TestContext} context Owning test lifecycle.
 * @param {unknown} decision JSON decision sent to the middleware.
 * @param {object} [options] HTTP response and request-observation controls.
 * @param {number} [options.status] Response status, defaulting to 200.
 * @param {string} [options.raw] Original response bytes instead of JSON encoding.
 * @param {boolean} [options.disconnect] Close the actual connection without a response.
 * @param {() => void} [options.onRequest] Observer run before sending a decision.
 * @returns {Promise<{endpoint: string, requests: ReviewRequest[]}>} Bound endpoint and received requests.
 */
export async function reviewServer(context, decision, options = {}) {
  /** @type {ReviewRequest[]} Original requests received over HTTP. */
  const requests = [];
  const server = createServer(async (request, response) => {
    request.setEncoding("utf8");
    let body = "";
    for await (const part of request) body += part;
    requests.push({
      url: request.url,
      method: request.method,
      headers: request.headers,
      body: JSON.parse(body),
    });
    options.onRequest?.();
    if (options.disconnect) {
      response.destroy();
      return;
    }
    response.writeHead(options.status ?? 200, { "content-type": "application/json" });
    response.end(options.raw ?? JSON.stringify(decision));
  });
  context.after(async () => {
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
  });
  await once(server.listen(0, "127.0.0.1"), "listening");
  const address = server.address();
  assert.ok(address && typeof address === "object");
  return { endpoint: `http://127.0.0.1:${address.port}`, requests };
}
