// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-AI Studio — production browser journeys.

import { expect, test, type Page } from "@playwright/test";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";
import { parseManifest } from "../src/contract/manifest";

/** Hold one manifest request while the browser leaves its consuming view. */
async function holdManifest(page: Page, url: string) {
  let markRequested: (() => void) | undefined;
  let release: (() => void) | undefined;
  const requested = new Promise<void>((resolve) => { markRequested = resolve; });
  const pending = new Promise<void>((resolve) => { release = resolve; });
  await page.route(url, async () => {
    markRequested?.();
    await pending;
  }, { times: 1 });
  return { requested, release };
}

test("renders capabilities from the published manifest", async ({ page }) => {
  await page.goto("/");
  expect(await page.evaluate(() => globalThis.crossOriginIsolated)).toBe(true);
  await expect(page.getByRole("heading", { name: "AI-Safety / Streaming Academy" })).toBeVisible();
  const response = page.waitForResponse((candidate) =>
    candidate.url().endsWith("/studio_manifest.json") && candidate.ok()
  );
  await page.getByRole("link", { name: "Capabilities" }).click();
  const manifest = parseManifest(await (await response).json());
  await expect(page.getByRole("heading", { name: "Capabilities" })).toBeVisible();
  await expect(page.getByText(manifest.content_digest, { exact: true })).toBeVisible();
  await expect(page.getByRole("table").getByRole("row")).toHaveCount(manifest.verbs.length + 1);
});

test("leaving Capabilities cancels the outstanding manifest load", async ({ page }) => {
  const url = "**/studio_manifest.json";
  const held = await holdManifest(page, url);
  try {
    await page.goto("/capabilities");
    await held.requested;
    await expect(page.getByRole("status")).toHaveText("Loading capabilities…");
    const [failed] = await Promise.all([
      page.waitForEvent("requestfailed", {
        predicate: (request) => request.url().endsWith("/studio_manifest.json"),
        timeout: 2500,
      }),
      page.getByRole("link", { name: "Home" }).click(),
    ]);
    expect(failed.failure()?.errorText).toBe("net::ERR_ABORTED");
    await page.getByRole("link", { name: "Capabilities" }).click();
    await expect(page.getByRole("table")).toBeVisible();
    await expect(page.getByRole("alert")).toHaveCount(0);
  } finally {
    held.release?.();
  }
});

test("Learn presents the authored lesson before loading the WASM demo", async ({ page }) => {
  const wasmRequests: string[] = [];
  const scripts: Array<Promise<{
    /** Address reported when a binding script loads early. */
    url: string;
    /** Served JavaScript checked for the generated binding export. */
    source: string;
  }>> = [];
  page.on("request", (request) => {
    if (request.url().endsWith(".wasm")) wasmRequests.push(request.url());
  });
  page.on("response", (response) => {
    if (response.ok() && response.request().resourceType() === "script") {
      scripts.push(response.text().then((source) => ({ url: response.url(), source })));
    }
  });
  /** Find the generated binding regardless of which chunk contains it. */
  const bindingRequests = async () => (await Promise.all(scripts))
    .filter(({ source }) => source.includes("__wbg_wasmstreamingkernel_free"))
    .map(({ url }) => url);

  await page.goto("/");
  await page.getByRole("link", { name: "Learn" }).click();
  await expect(page.getByRole("heading", { name: "Streaming coherence and contradiction halts" })).toBeVisible();
  await expect(page.getByText(/The Python contradiction interlock checks completed claims/)).toBeVisible();
  expect(wasmRequests).toHaveLength(0);
  expect(await bindingRequests()).toHaveLength(0);

  await page.getByRole("link", { name: "Run the WASM sample" }).click();
  await expect(page.getByRole("button", { name: "Run the sample" })).toBeEnabled();
  expect(wasmRequests).toHaveLength(1);
  expect(await bindingRequests()).toHaveLength(1);
});

test("keeps planned bookmarks reachable with a way back to the supported Studio routes", async ({ page }) => {
  const planned = [
    ["/pipeline", "Pipeline"],
    ["/benchmarks", "Benchmarks"],
    ["/evidence", "Science & Evidence"],
    ["/redteam", "Red-Team"],
    ["/architecture", "Architecture"],
    ["/lab", "3D Lab"],
  ] as const;
  for (const [path, title] of planned) {
    await page.goto(path);
    await expect(page.getByRole("heading", { name: title, exact: true })).toBeVisible();
    await expect(page.getByText("This section is planned for a future Studio release.")).toBeVisible();
    await expect(page.getByRole("main").getByRole("link", { name: "Open Studio home" })).toBeVisible();
  }
  await expect(page.getByRole("navigation").getByRole("link")).toHaveText([
    "Home", "Learn", "Play", "Capabilities",
  ]);
  await page.getByRole("link", { name: "Open Studio home" }).click();
  await expect(page.getByRole("heading", { name: "AI-Safety / Streaming Academy" })).toBeVisible();
});

test("offers recovery for an unknown Studio bookmark", async ({ page }) => {
  await page.goto("/old-workshop?lesson=2#notes");
  const bookmark = page.url();
  const unavailable = page.getByRole("heading", { name: "Studio page not found" });
  await expect(unavailable).toBeVisible();
  await page.getByRole("link", { name: "Open Studio home" }).click();
  await expect(page.getByRole("heading", { name: "AI-Safety / Streaming Academy" })).toBeVisible();
  await page.goBack();
  await expect(page).toHaveURL(bookmark);
  await expect(unavailable).toBeVisible();
});

test("detects a missing Studio mount and starts from the restored document", async ({ page }) => {
  await page.route("**/", async (route) => {
    const response = await route.fetch();
    const html = await response.text();
    const mount = '<div id="root"></div>';
    expect(html).toContain(mount);
    await route.fulfill({ response, body: html.replace(mount, "") });
  }, { times: 1 });
  const startupError = page.waitForEvent("pageerror");
  await page.goto("/");
  expect((await startupError).message).toBe("missing #root mount node");
  await expect(page.getByRole("main")).toHaveCount(0);

  await page.reload();
  await expect(page.getByRole("heading", { name: "AI-Safety / Streaming Academy" })).toBeVisible();
});

test("keeps navigation available when a lazy view fails to load", async ({ page }) => {
  await page.route("**/assets/Play-*.js", (route) => route.fulfill({ status: 503 }), { times: 1 });
  await page.goto("/");
  const unavailable = page.waitForResponse((response) =>
    /\/assets\/Play-[^/]+\.js$/.test(new URL(response.url()).pathname) && response.status() === 503
  );
  await page.getByRole("link", { name: "Play" }).click();
  await unavailable;

  await expect(page.getByRole("alert")).toHaveText("This Studio page could not load.");
  await page.getByRole("link", { name: "Home" }).click();
  await expect(page.getByRole("heading", { name: "AI-Safety / Streaming Academy" })).toBeVisible();
  await expect(page.getByRole("alert")).toHaveCount(0);

  await page.getByRole("link", { name: "Play" }).click();
  await page.getByRole("button", { name: "Reload page" }).click();
  await expect(page).toHaveURL(/\/play$/);
  const run = page.getByRole("button", { name: "Run the sample" });
  await expect(run).toBeEnabled();
  await run.click();
  await expect(page.getByRole("status")).toContainText("hard_limit");
});

test("reports an unavailable manifest and displays the future-era boundary", async ({ page, request }) => {
  const served = await request.get("/studio_manifest.json");
  expect(served.ok()).toBe(true);
  const document: unknown = await served.json();
  if (typeof document !== "object" || document === null || !("schema_a" in document)) {
    throw new Error("Expected a generated schema-A envelope");
  }
  const schema = document.schema_a;
  if (typeof schema !== "object" || schema === null) throw new Error("Expected an object for schema-A");
  const future = { ...document, schema_a: { ...schema, contract_era: "v-next" } };

  let release: (() => void) | undefined;
  const pending = new Promise<void>((resolve) => { release = resolve; });
  await page.route("**/studio_manifest.json", async (route) => {
    await pending;
    await route.fulfill({ status: 503 });
  }, { times: 1 });
  try {
    await page.goto("/capabilities");
    await expect(page.getByRole("status")).toHaveText("Loading capabilities…");
  } finally {
    release?.();
  }
  await expect(page.getByRole("alert")).toHaveText("The Studio capability manifest could not load.");
  await expect(page.getByRole("table")).toHaveCount(0);

  await page.route("**/studio_manifest.json", (route) => route.fulfill({ json: future }));
  await page.reload();
  await expect(page.getByRole("status")).toContainText("Manifest era v-next");
  await expect(page.getByRole("columnheader", { name: "Declared safety tier" })).toBeVisible();
  await expect(page.getByRole("table").getByRole("row"))
    .toHaveCount(parseManifest(future).verbs.length + 1);
});

test("reports a truncated capability manifest and recovers on reload", async ({ page }) => {
  await page.route("**/studio_manifest.json", (route) => route.fulfill({
    status: 200,
    contentType: "application/json",
    body: '{"schema_a":',
  }), { times: 1 });
  await page.goto("/capabilities");
  await expect(page.getByRole("alert")).toHaveText("The Studio capability manifest could not load.");
  await expect(page.getByRole("table")).toHaveCount(0);

  await page.reload();
  await expect(page.getByRole("table")).toBeVisible();
  await expect(page.getByRole("alert")).toHaveCount(0);
});

test("the generated WASM asset halts and each run starts a fresh trace", async ({ page }) => {
  const wasm = page.waitForResponse((response) => response.url().endsWith(".wasm") && response.ok());
  await page.goto("/play");
  expect((await wasm).headers()["content-type"]).toContain("application/wasm");
  const run = page.getByRole("button", { name: "Run the sample" });
  const trace = page.getByRole("list", { name: "Token trace" }).getByRole("listitem");
  await run.click();
  await expect(trace).toHaveCount(6);
  await expect(trace.last()).toContainText('" Germany" score=0.12');
  await expect(page.getByRole("status")).toContainText("hard_limit");
  const firstTrace = await trace.allTextContents();

  await run.click();
  await expect(trace).toHaveText(firstTrace);
  await page.getByRole("link", { name: "Home" }).click();
  await page.getByRole("link", { name: "Play" }).click();
  await expect(trace).toHaveCount(0);
  await run.click();
  await expect(trace).toHaveText(firstTrace);
});

test("keeps the in-flight WASM load when the visitor leaves and returns to Play", async ({ page }) => {
  const asset = "**/*.wasm";
  const requests: string[] = [];
  let markRequested!: () => void;
  let release!: () => void;
  const requested = new Promise<void>((resolve) => { markRequested = resolve; });
  const pending = new Promise<void>((resolve) => { release = resolve; });
  await page.route(asset, async (route) => {
    requests.push(route.request().url());
    markRequested();
    await pending;
    await route.continue();
  });

  const loading = page.getByRole("button", { name: "Loading kernel…" });
  try {
    await page.goto("/play");
    await requested;
    await expect(loading).toBeDisabled();
    await page.getByRole("link", { name: "Home" }).click();
    await expect(page.getByRole("heading", { name: "AI-Safety / Streaming Academy" })).toBeVisible();
    await expect(loading).toHaveCount(0);
    await page.getByRole("link", { name: "Play" }).click();
    await expect(loading).toBeDisabled();
  } finally {
    release();
  }

  await page.getByRole("button", { name: "Run the sample" }).click();
  await expect(page.getByRole("status")).toContainText("hard_limit");
  expect(requests).toHaveLength(1);
});

test("retries a failed asset request and runs the real kernel", async ({ page }) => {
  await page.route("**/*.wasm", (route) => route.fulfill({ status: 503 }), { times: 1 });
  await page.goto("/play");
  await expect(page.getByRole("alert")).toHaveText("The WASM kernel could not load.");
  await page.getByRole("button", { name: "Retry WASM" }).click();
  await page.getByRole("button", { name: "Run the sample" }).click();
  await expect(page.getByRole("status")).toContainText("hard_limit");
  await expect(page.getByRole("alert")).toHaveCount(0);
});

test("recovers when a successful asset response contains a truncated WASM module", async ({ page }) => {
  await page.route("**/*.wasm", (route) => route.fulfill({
    status: 200,
    contentType: "application/wasm",
    body: Buffer.from([0x00, 0x61, 0x73, 0x6d]),
  }), { times: 1 });
  await page.goto("/play");
  await expect(page.getByRole("alert")).toHaveText("The WASM kernel could not load.");
  await expect(page.getByRole("list", { name: "Token trace" })).toHaveCount(0);

  await page.getByRole("button", { name: "Retry WASM" }).click();
  await page.getByRole("button", { name: "Run the sample" }).click();
  await expect(page.getByRole("status")).toContainText("hard_limit");
  await expect(page.getByRole("alert")).toHaveCount(0);
});

test("reloads the working kernel after an incompatible WASM session result", async ({ page }) => {
  const wasmRequests: string[] = [];
  page.on("request", (request) => {
    if (request.url().endsWith(".wasm")) wasmRequests.push(request.url());
  });
  await page.route("**/*.wasm", async (route) => {
    const response = await route.fetch();
    expect(response.ok()).toBe(true);
    const bytes = await response.body();
    // Keep the native module executable while simulating a serializer field rename.
    const field = Buffer.from("halt_reason");
    const offset = bytes.indexOf(field);
    expect(offset).toBeGreaterThan(-1);
    expect(bytes.lastIndexOf(field)).toBe(offset);
    Buffer.from("halt_schema").copy(bytes, offset);
    expect(WebAssembly.validate(Uint8Array.from(bytes))).toBe(true);
    await route.fulfill({ response, body: bytes });
  }, { times: 1 });

  await page.goto("/play?source=bookmark#sample");
  const originalUrl = page.url();
  const run = page.getByRole("button", { name: "Run the sample" });
  const trace = page.getByRole("list", { name: "Token trace" });
  await run.click();
  await expect(page.getByRole("alert")).toHaveText("The WASM sample could not run.");
  await expect(trace).toHaveCount(0);

  const navigation = page.waitForRequest((request) =>
    request.isNavigationRequest() && new URL(request.url()).pathname === "/play",
  );
  await page.getByRole("button", { name: "Reload page" }).click();
  await navigation;
  await expect(page).toHaveURL(originalUrl);
  await run.click();
  await expect(page.getByRole("status")).toContainText("hard_limit");
  await expect(page.getByRole("alert")).toHaveCount(0);
  expect(wasmRequests).toHaveLength(2);
});

/** Run a browser journey against a real federation host on a separate origin. */
async function withFederationHost(
  portalUrl: URL,
  check: (hostUrl: URL) => Promise<void>,
): Promise<void> {
  const host = await createServer({
    configFile: false,
    root: fileURLToPath(new URL("./federated-host", import.meta.url)),
    server: { host: "127.0.0.1", port: 0, strictPort: true },
  });
  try {
    await host.listen();
    const address = host.httpServer?.address();
    if (!address || typeof address === "string") throw new Error("The federation host has no TCP port");
    const hostUrl = new URL(`http://127.0.0.1:${address.port}/`);
    hostUrl.searchParams.set("portal", portalUrl.href);
    await check(hostUrl);
  } finally {
    await host.close();
  }
}

test("the federated panel reads the producer manifest from another origin", async ({ page, baseURL }) => {
  if (!baseURL) throw new Error("The Studio preview URL is required");
  const portalUrl = new URL("/", baseURL);
  await withFederationHost(portalUrl, async (hostUrl) => {
    const manifestUrls: string[] = [];
    page.on("request", (request) => {
      if (request.url().endsWith("/studio_manifest.json")) manifestUrls.push(request.url());
    });
    await page.goto(hostUrl.href);
    await expect(page.getByRole("link", { name: "Open the full portal" })).toHaveAttribute("href", portalUrl.href);
    await expect(page.getByText("verbs", { exact: true })).toBeVisible();
    const producerManifest = new URL("studio_manifest.json", portalUrl).href;
    expect(manifestUrls).toEqual([producerManifest]);

    await page.route(producerManifest, (route) => route.fulfill({
      status: 503,
      headers: { "access-control-allow-origin": hostUrl.origin },
    }));
    await page.reload();
    await expect(page.getByText("Summary unavailable.", { exact: true })).toBeVisible();
    expect(manifestUrls).toEqual([producerManifest, producerManifest]);

    await page.unroute(producerManifest);
    const held = await holdManifest(page, producerManifest);
    try {
      await page.reload();
      await held.requested;
      await expect(page.getByText("Loading summary…", { exact: true })).toBeVisible();
      const [failed] = await Promise.all([
        page.waitForEvent("requestfailed", {
          predicate: (request) => request.url() === producerManifest,
          timeout: 2500,
        }),
        page.getByRole("button", { name: "Remove panel" }).click(),
      ]);
      expect(failed.failure()?.errorText).toBe("net::ERR_ABORTED");
      await expect(page.getByText("Summary unavailable.", { exact: true })).toHaveCount(0);
      expect(manifestUrls).toEqual([producerManifest, producerManifest, producerManifest]);
    } finally {
      held.release?.();
    }
  });
});

test("the federated panel waits for a fresh manifest when returning to an earlier portal", async ({ page, baseURL }) => {
  if (!baseURL) throw new Error("The Studio preview URL is required");
  const portalUrl = new URL("/", baseURL);
  const originalManifest = new URL("studio_manifest.json", portalUrl).href;
  await withFederationHost(portalUrl, async (hostUrl) => {
    await page.goto(hostUrl.href);
    const verbs = page.getByText("verbs", { exact: true });
    await expect(verbs).toBeVisible();

    const alternatePortal = new URL("/alternate/", hostUrl);
    const alternateManifest = new URL("studio_manifest.json", alternatePortal).href;
    const alternate = await holdManifest(page, alternateManifest);
    const input = page.getByLabel("Portal URL");
    const change = page.getByRole("button", { name: "Change portal" });
    const link = page.getByRole("link", { name: "Open the full portal" });
    let markOriginalRequested!: () => void;
    let releaseOriginal!: () => void;
    const originalRequested = new Promise<void>((resolve) => { markOriginalRequested = resolve; });
    const pendingOriginal = new Promise<void>((resolve) => { releaseOriginal = resolve; });
    await page.route(originalManifest, async (route) => {
      markOriginalRequested();
      await pendingOriginal;
      await route.continue();
    }, { times: 1 });

    try {
      await input.fill(alternatePortal.href);
      await change.click();
      await alternate.requested;
      await expect(link).toHaveAttribute("href", alternatePortal.href);
      await expect(verbs).toHaveCount(0);

      await input.fill(portalUrl.href);
      const [failed] = await Promise.all([
        page.waitForEvent("requestfailed", {
          predicate: (request) => request.url() === alternateManifest,
          timeout: 2500,
        }),
        change.click(),
      ]);
      expect(failed.failure()?.errorText).toBe("net::ERR_ABORTED");
      await originalRequested;
      await expect(link).toHaveAttribute("href", portalUrl.href);
      await expect(verbs).toHaveCount(0);
      await expect(page.getByText("Loading summary…", { exact: true })).toBeVisible();
    } finally {
      alternate.release?.();
      releaseOriginal();
    }
    await expect(verbs).toBeVisible();
    await expect(page.getByText("Summary unavailable.", { exact: true })).toHaveCount(0);
  });
});
