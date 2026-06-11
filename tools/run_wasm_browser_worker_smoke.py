#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Run a real headless-browser Web Worker smoke for backfire-wasm."""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import shutil
import socket
import subprocess  # nosec B404
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

SCHEMA_VERSION = "director-ai.wasm-browser-worker-smoke.v1"
DEFAULT_PACKAGE_DIR = Path("backfire-kernel/crates/backfire-wasm/pkg")
CHROME_CANDIDATES = (
    "google-chrome",
    "chromium",
    "chromium-browser",
)
WORKER_RESULT_EXPRESSION = """
(() => {
  const element = document.getElementById("result");
  return element ? element.textContent : "";
})()
""".strip()


@dataclass(frozen=True)
class WasmBrowserWorkerSmokeReport:
    """Browser Web Worker smoke report for generated WASM package."""

    schema_version: str
    passed: bool
    runtime: str
    package_dir: str
    chrome_version: str
    worker_result: dict[str, Any]
    blockers: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the smoke report."""

        return {
            "schema_version": self.schema_version,
            "passed": self.passed,
            "runtime": self.runtime,
            "package_dir": self.package_dir,
            "chrome_version": self.chrome_version,
            "worker_result": dict(self.worker_result),
            "blockers": [dict(blocker) for blocker in self.blockers],
        }

    def to_markdown(self) -> str:
        """Return a compact operator-readable report."""

        blockers = [
            f"- {blocker['code']} — {blocker['message']}" for blocker in self.blockers
        ]
        if not blockers:
            blockers = ["- none"]
        return "\n".join(
            [
                "# WASM Browser Worker Smoke",
                "",
                f"passed: {str(self.passed).lower()}",
                f"runtime: {self.runtime}",
                f"chrome_version: {self.chrome_version}",
                f"package_dir: {self.package_dir}",
                "",
                "## Worker Result",
                "",
                "```json",
                json.dumps(self.worker_result, indent=2, sort_keys=True),
                "```",
                "",
                "## Blockers",
                "",
                *blockers,
                "",
            ]
        )


def run_wasm_browser_worker_smoke(
    package_dir: str | Path,
    *,
    chrome_path: str | Path | None = None,
    timeout_seconds: int = 30,
) -> WasmBrowserWorkerSmokeReport:
    """Run headless Chrome against a browser module Worker smoke page."""

    package_path = Path(package_dir).resolve()
    blockers: list[dict[str, str]] = []
    if not package_path.is_dir():
        blockers.append(_blocker("package_dir_missing", "WASM package dir is missing"))
        return _report(
            package_path=package_path,
            chrome_version="",
            worker_result={},
            blockers=blockers,
        )
    chrome = Path(chrome_path) if chrome_path is not None else _find_chrome()
    if chrome is None:
        blockers.append(_blocker("chrome_missing", "No Chrome/Chromium binary found"))
        return _report(
            package_path=package_path,
            chrome_version="",
            worker_result={},
            blockers=blockers,
        )
    chrome_version = _chrome_version(chrome)
    with tempfile.TemporaryDirectory(prefix="director-wasm-worker-") as tmp:
        web_root = Path(tmp)
        (web_root / "pkg").symlink_to(package_path, target_is_directory=True)
        _write_smoke_assets(web_root)
        with _serve_directory(web_root) as base_url:
            worker_result = _run_chrome_worker_result(
                chrome,
                f"{base_url}/smoke.html",
                timeout_seconds=timeout_seconds,
            )
    if not worker_result.get("passed"):
        blockers.append(
            _blocker("worker_smoke_failed", "Browser Web Worker smoke did not pass")
        )
    return _report(
        package_path=package_path,
        chrome_version=chrome_version,
        worker_result=worker_result,
        blockers=blockers,
    )


def _write_smoke_assets(web_root: Path) -> None:
    (web_root / "smoke.html").write_text(
        """
<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>backfire-wasm smoke</title>
<pre id="result">pending</pre>
<script type="module">
const result = document.getElementById("result");
function runWorkerSmoke() {
  return new Promise((resolve) => {
    const worker = new Worker("./smoke-worker.js", { type: "module" });
    const timeout = setTimeout(() => {
      worker.terminate();
      resolve({ passed: false, error: "timeout" });
    }, 12000);
    worker.onmessage = (event) => {
      clearTimeout(timeout);
      worker.terminate();
      resolve(event.data);
    };
    worker.onerror = (event) => {
      clearTimeout(timeout);
      worker.terminate();
      resolve({
        passed: false,
        error: event.message,
      });
    };
  });
}
result.textContent = JSON.stringify(await runWorkerSmoke());
</script>
</html>
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (web_root / "smoke-worker.js").write_text(
        """
import init, { WasmStreamingKernel } from "./pkg/backfire_wasm.js";

const config = JSON.stringify({
  coherence_threshold: 0.6,
  hard_limit: 0.5,
  soft_limit: 0.7,
  w_logic: 0.6,
  w_fact: 0.4,
  window_size: 3,
  window_threshold: 0.55,
  trend_window: 3,
  trend_threshold: 0.2,
  history_window: 5,
  deadline_ms: 50,
  logit_entropy_limit: 1.2,
});

try {
  const wasmUrl = new URL("./pkg/backfire_wasm_bg.wasm", self.location.href);
  const wasmResponse = await fetch(wasmUrl);
  if (!wasmResponse.ok) {
    throw new Error(`wasm fetch failed: ${wasmResponse.status}`);
  }
  await init(await wasmResponse.arrayBuffer());
  const kernel = new WasmStreamingKernel(config);
  const first = kernel.process_token("safe-token", 0.9);
  const second = kernel.process_token("halt-token", 0.1);
  const third = kernel.process_token("post-halt-token", 0.99);
  const passed = (
    kernel.is_active() === false &&
    first.halted === false &&
    second.halted === true &&
    third.halted === true &&
    String(second.halt_reason || "").startsWith("hard_limit")
  );
  self.postMessage({
    passed,
    first_halted: first.halted,
    second_halted: second.halted,
    third_halted: third.halted,
    halt_reason: second.halt_reason || "",
    active_after_halt: kernel.is_active(),
    token_count: third.tokens.length,
  });
} catch (error) {
  self.postMessage({
    passed: false,
    error: String(error),
    stack: error && error.stack ? String(error.stack) : "",
  });
}
""".strip()
        + "\n",
        encoding="utf-8",
    )


@contextlib.contextmanager
def _serve_directory(directory: Path):
    handler = lambda *args, **kwargs: SimpleHTTPRequestHandler(  # noqa: E731
        *args,
        directory=str(directory),
        **kwargs,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _run_chrome_worker_result(
    chrome: Path,
    url: str,
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="director-chrome-profile-") as profile:
        profile_dir = Path(profile)
        process = subprocess.Popen(  # nosec B603
            [
                str(chrome),
                "--headless=new",
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--remote-debugging-port=0",
                f"--user-data-dir={profile_dir}",
                "about:blank",
            ],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            port = _wait_for_devtools_port(
                profile_dir,
                process,
                timeout_seconds=timeout_seconds,
            )
            worker_result = _poll_worker_result(
                port=port,
                url=url,
                timeout_seconds=timeout_seconds,
            )
        except (OSError, TimeoutError, ValueError, subprocess.SubprocessError) as exc:
            worker_result = {
                "passed": False,
                "error": str(exc),
            }
        finally:
            _terminate_process(process)
        return worker_result


def _wait_for_devtools_port(
    profile_dir: Path,
    process: subprocess.Popen[str],
    *,
    timeout_seconds: int,
) -> int:
    active_port = profile_dir / "DevToolsActivePort"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stderr = process.stderr.read() if process.stderr is not None else ""
            raise subprocess.SubprocessError(
                f"Chrome exited before DevTools opened: {stderr.strip()}"
            )
        if active_port.exists():
            lines = active_port.read_text(encoding="utf-8").splitlines()
            if lines:
                return int(lines[0])
        time.sleep(0.05)
    raise TimeoutError("Chrome DevTools port did not become available")


def _poll_worker_result(
    *,
    port: int,
    url: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    target = _create_devtools_target(port)
    websocket_url = str(target["webSocketDebuggerUrl"])
    deadline = time.monotonic() + timeout_seconds
    with _DevToolsWebSocket.connect(
        websocket_url, timeout_seconds=timeout_seconds
    ) as cdp:
        cdp.call("Runtime.enable")
        cdp.call("Page.enable")
        cdp.call("Page.navigate", {"url": url})
        while time.monotonic() < deadline:
            response = cdp.call(
                "Runtime.evaluate",
                {
                    "expression": WORKER_RESULT_EXPRESSION,
                    "returnByValue": True,
                },
            )
            result = response.get("result", {}).get("result", {})
            value = str(result.get("value", ""))
            if value and value != "pending":
                return _parse_worker_result_text(value)
            time.sleep(0.1)
    return {
        "passed": False,
        "error": "browser worker result timeout",
    }


def _create_devtools_target(port: int) -> dict[str, Any]:
    endpoint = f"http://127.0.0.1:{port}/json/new?about:blank"
    request = urllib.request.Request(endpoint, method="PUT")
    with urllib.request.urlopen(request, timeout=5) as response:  # nosec B310
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict) or "webSocketDebuggerUrl" not in payload:
        raise ValueError(
            "Chrome DevTools target response did not include a WebSocket URL"
        )
    return payload


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


class _DevToolsWebSocket:
    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._next_id = 1

    @classmethod
    def connect(
        cls,
        websocket_url: str,
        *,
        timeout_seconds: int,
    ) -> _DevToolsWebSocket:
        parsed = urllib.parse.urlparse(websocket_url)
        if parsed.scheme != "ws" or parsed.hostname is None:
            raise ValueError("Chrome DevTools URL must be a local ws:// endpoint")
        if parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError("Chrome DevTools URL must stay on localhost")
        port = parsed.port
        if port is None:
            raise ValueError("Chrome DevTools URL is missing a port")
        path = parsed.path
        if parsed.query:
            path = f"{path}?{parsed.query}"
        sock = socket.create_connection(
            (parsed.hostname, port), timeout=timeout_seconds
        )
        sock.settimeout(timeout_seconds)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = "\r\n".join(
            [
                f"GET {path} HTTP/1.1",
                f"Host: {parsed.hostname}:{port}",
                "Upgrade: websocket",
                "Connection: Upgrade",
                "Sec-WebSocket-Version: 13",
                f"Sec-WebSocket-Key: {key}",
                "",
                "",
            ]
        )
        sock.sendall(request.encode("ascii"))
        response = _read_http_response(sock)
        if b" 101 " not in response.split(b"\r\n", 1)[0]:
            sock.close()
            raise ConnectionError("Chrome DevTools WebSocket handshake failed")
        return cls(sock)

    def __enter__(self) -> _DevToolsWebSocket:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._sock.close()

    def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        command_id = self._next_id
        self._next_id += 1
        message: dict[str, Any] = {
            "id": command_id,
            "method": method,
        }
        if params is not None:
            message["params"] = params
        self._send_json(message)
        while True:
            decoded = json.loads(self._recv_text())
            if not isinstance(decoded, dict):
                continue
            payload = cast(dict[str, Any], decoded)
            if payload.get("id") == command_id:
                if "error" in payload:
                    raise RuntimeError(str(payload["error"]))
                return payload

    def _send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._send_frame(opcode=0x1, payload=body)

    def _send_frame(self, *, opcode: int, payload: bytes) -> None:
        header = bytearray([0x80 | opcode])
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        elif length <= 0xFFFF:
            header.extend([0x80 | 126, (length >> 8) & 0xFF, length & 0xFF])
        else:
            header.append(0x80 | 127)
            header.extend(length.to_bytes(8, "big"))
        mask = os.urandom(4)
        masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        self._sock.sendall(bytes(header) + mask + masked)

    def _recv_text(self) -> str:
        fragments: list[bytes] = []
        while True:
            frame = _recv_frame(self._sock)
            if frame.opcode == 0x8:
                raise ConnectionError("Chrome DevTools WebSocket closed")
            if frame.opcode == 0x9:
                self._send_frame(opcode=0xA, payload=frame.payload)
                continue
            if frame.opcode in {0x1, 0x0}:
                fragments.append(frame.payload)
                if frame.final:
                    return b"".join(fragments).decode("utf-8")


@dataclass(frozen=True)
class _WebSocketFrame:
    final: bool
    opcode: int
    payload: bytes


def _read_http_response(sock: socket.socket) -> bytes:
    data = bytearray()
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data.extend(chunk)
    return bytes(data)


def _recv_frame(sock: socket.socket) -> _WebSocketFrame:
    head = _read_exact(sock, 2)
    first, second = head
    final = bool(first & 0x80)
    opcode = first & 0x0F
    masked = bool(second & 0x80)
    length = second & 0x7F
    if length == 126:
        length = int.from_bytes(_read_exact(sock, 2), "big")
    elif length == 127:
        length = int.from_bytes(_read_exact(sock, 8), "big")
    mask = _read_exact(sock, 4) if masked else b""
    payload = _read_exact(sock, length) if length else b""
    if masked:
        payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    return _WebSocketFrame(final=final, opcode=opcode, payload=payload)


def _read_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("unexpected EOF while reading WebSocket frame")
        data.extend(chunk)
    return bytes(data)


def _parse_worker_result_text(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {
            "passed": False,
            "error": f"result JSON invalid: {exc.msg}",
            "raw": raw,
        }
    return payload if isinstance(payload, dict) else {"passed": False}


def _find_chrome() -> Path | None:
    for candidate in CHROME_CANDIDATES:
        found = shutil.which(candidate)
        if found:
            return Path(found)
    return None


def _chrome_version(chrome: Path) -> str:
    try:
        completed = subprocess.run(  # nosec B603
            [str(chrome), "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip()


def _report(
    *,
    package_path: Path,
    chrome_version: str,
    worker_result: dict[str, Any],
    blockers: list[dict[str, str]],
) -> WasmBrowserWorkerSmokeReport:
    return WasmBrowserWorkerSmokeReport(
        schema_version=SCHEMA_VERSION,
        passed=not blockers,
        runtime="headless-chrome-web-worker",
        package_dir=package_path.as_posix(),
        chrome_version=chrome_version,
        worker_result=worker_result,
        blockers=tuple(blockers),
    )


def _blocker(code: str, message: str) -> dict[str, str]:
    return {
        "code": code,
        "severity": "error",
        "message": message,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the browser worker smoke from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--chrome-path", type=Path, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON report")
    args = parser.parse_args(argv)

    report = run_wasm_browser_worker_smoke(
        args.package_dir,
        chrome_path=args.chrome_path,
        timeout_seconds=args.timeout_seconds,
    )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(report.to_markdown())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
