#!/usr/bin/env python3
"""
End-to-end smoke test for the HTTP API.

Covers the behaviours that the library test suite cannot see: the container
serving at all, the event loop staying free during a slow request, the download
deadline firing, and the input handling of the prediction endpoints.

Usage:
    python3 tests/api_smoke.py --base-url http://127.0.0.1:8000

The target must run with OPENNSFW2_DOWNLOAD_DEADLINE_SECONDS set to a small
value, otherwise the deadline check takes the full default budget.

The stall test needs the target to dial back to this script. Pass the host name
that the target resolves this machine by, which is `host.docker.internal` for a
container and `127.0.0.1` for a local process.
"""
import argparse
import base64
import json
import socketserver
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler
from typing import Any, Dict, Tuple

# Smallest valid PNG, so this script needs no image library.
PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
)

ARGS: argparse.Namespace
failures = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        failures.append(name)


def get(path: str, timeout: float = 30) -> Tuple[Any, str]:
    try:
        response = urllib.request.urlopen(ARGS.base_url + path, timeout=timeout)
        return response.status, response.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def post(path: str, payload: Dict[str, Any], timeout: float = 60) -> Tuple[Any, str]:
    request = urllib.request.Request(
        ARGS.base_url + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}
    )
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
        return response.status, response.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


class SlowDripHandler(BaseHTTPRequestHandler):
    """Serves an endless trickle of bytes, to stall a download."""

    def do_GET(self) -> None:  # pylint: disable=invalid-name
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.end_headers()
        try:
            while True:
                self.wfile.write(b"\x00" * 16)
                self.wfile.flush()
                time.sleep(1.0)
        except Exception:  # pylint: disable=broad-except
            pass

    def log_message(self, *args: Any) -> None:
        pass


def start_slow_drip_server() -> int:
    server = socketserver.TCPServer(("0.0.0.0", 0), SlowDripHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return int(server.server_address[1])


def test_health() -> None:
    status, body = get("/health/")
    check("health endpoint responds", status == 200 and "healthy" in body, body[:80])

    status, body = get("/health/model")
    check(
        "model reports loaded",
        status == 200 and json.loads(body).get("model_loaded") is True,
        body[:80]
    )

    status, _ = get("/docs")
    check("docs render", status == 200)


def test_image_inputs() -> None:
    variants = {
        "plain base64": PNG_BASE64,
        # Output of the `base64` CLI is wrapped, and browsers produce data URIs.
        "base64 wrapped at 76 columns": "\n".join(textwrap.wrap(PNG_BASE64, 76)),
        "data URI prefix": f"data:image/png;base64,{PNG_BASE64}",
    }
    for name, data in variants.items():
        status, body = post("/predict/image", {"input": {"type": "base64", "data": data}})
        ok = status == 200 and "nsfw_probability" in body
        check(f"predict image accepts {name}", ok, f"{status} {body[:70]}")

    status, body = post(
        "/predict/images",
        {"inputs": [{"type": "base64", "data": PNG_BASE64}] * 3}
    )
    ok = status == 200 and len(json.loads(body).get("results", [])) == 3
    check("predict images returns one result per input", ok, f"{status} {body[:70]}")


def test_rejects_bad_input() -> None:
    cases = {
        "junk bytes as image": ("/predict/image", base64.b64encode(b"junk").decode()),
        "undecodable video": ("/predict/video", base64.b64encode(b"junk").decode()),
    }
    for name, (path, data) in cases.items():
        status, body = post(path, {"input": {"type": "base64", "data": data}})
        check(f"rejects {name} with 400", status == 400, f"{status} {body[:70]}")

    status, body = post("/predict/images", {"inputs": []})
    check("rejects empty input list", status == 422, f"{status} {body[:70]}")

    status, body = post("/predict/image", {"input": {"type": "url", "data": "not-a-url"}})
    check("rejects malformed URL", status == 400, f"{status} {body[:70]}")


class StalledRequest:
    """Runs one prediction request in the background and records how it ended."""

    def __init__(self, url: str, timeout: float) -> None:
        self.url = url
        self.timeout = timeout
        self.status: Any = None
        self.body = ""
        self.elapsed = 0.0
        self.finished = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        start = time.monotonic()
        try:
            self.status, self.body = post(
                "/predict/image",
                {"input": {"type": "url", "data": self.url}},
                timeout=self.timeout
            )
        except Exception as e:  # pylint: disable=broad-except
            # A client-side timeout means the server never answered at all.
            self.status, self.body = "no response", repr(e)
        self.elapsed = time.monotonic() - start
        self.finished.set()

    def start(self) -> None:
        self.thread.start()


def test_event_loop_stays_free() -> None:
    """
    A blocking request must not stall the server.

    The handlers have to run off the event loop, or one slow download freezes
    every other request, including the container health check. The download must
    also give up at the deadline instead of hanging forever.
    """
    deadline = ARGS.deadline_seconds
    port = start_slow_drip_server()
    stalled = StalledRequest(
        f"http://{ARGS.callback_host}:{port}/stalled.png", timeout=deadline * 4
    )
    stalled.start()
    time.sleep(3)  # Let the request reach the download.

    # A request that failed to connect proves nothing, so confirm it is still
    # running before the health probe is treated as meaningful.
    if stalled.finished.is_set():
        check(
            "stalled request actually stalls", False,
            f"returned after {stalled.elapsed:.1f}s, "
            f"cannot reach {stalled.url}: {stalled.body[:70]}"
        )
        check("health stays responsive during a stalled request", False, "not tested")
        check("stalled download hits the deadline", False, "not tested")
        return
    check("stalled request actually stalls", True, f"still running via {stalled.url}")

    start = time.monotonic()
    try:
        status, _ = get("/health/", timeout=10)
        elapsed = time.monotonic() - start
        check(
            "health stays responsive during a stalled request",
            status == 200 and elapsed < 5,
            f"{elapsed:.2f}s"
        )
    except Exception as e:  # pylint: disable=broad-except
        check("health stays responsive during a stalled request", False, repr(e))

    if not stalled.finished.wait(timeout=deadline * 4):
        check("stalled download hits the deadline", False, "request never returned")
        return
    # The elapsed floor is what separates a real deadline from a failure to
    # connect, which would also answer 400.
    check(
        "stalled download hits the deadline",
        stalled.status == 400 and "deadline" in stalled.body
        and deadline * 0.5 < stalled.elapsed < deadline * 3,
        f"{stalled.status} after {stalled.elapsed:.1f}s "
        f"(deadline {deadline}s) {stalled.body[:70]}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000",
                        help="Base URL of the API under test.")
    parser.add_argument("--deadline-seconds", type=int, default=10,
                        help="Value of OPENNSFW2_DOWNLOAD_DEADLINE_SECONDS on the target.")
    parser.add_argument("--callback-host", default="127.0.0.1",
                        help="Host name by which the target reaches this machine.")
    return parser.parse_args()


def main() -> int:
    print(f"Testing {ARGS.base_url}\n")
    test_health()
    test_image_inputs()
    test_rejects_bad_input()
    test_event_loop_stays_free()

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    ARGS = parse_args()
    sys.exit(main())
