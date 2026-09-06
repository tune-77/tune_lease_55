"""Run the deployed readiness probe against a local HTTP server."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest


@pytest.mark.parametrize("raw_key", ["test-key", "test-key\n", " \ttest-key\r\n", "", " \t\r\n", None])
def test_probe_normalizes_key_before_sending_http(raw_key):
    expected_key = (raw_key or "").strip()
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            supplied = self.headers.get("X-API-Key")
            received.append((self.path, supplied))
            valid = self.path == "/api/health/auth" and (supplied or "") == expected_key
            self.send_response(200 if valid else 401)
            self.end_headers()

        def log_message(self, *_args):
            pass

    source = (Path(__file__).resolve().parents[1] / "scripts/start_cloud_run.sh").read_text()
    marker = 'python - "$FASTAPI_HOST" "$FASTAPI_PORT" "$READY_TIMEOUT_SECONDS" <<\'PY\'\n'
    probe = source.split(marker, 1)[1].split("\nPY\n", 1)[0]
    env = dict(os.environ)
    env.pop("API_ACCESS_KEY", None)
    env["no_proxy"] = "127.0.0.1"
    if raw_key is not None:
        env["API_ACCESS_KEY"] = raw_key
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            result = subprocess.run(
                [sys.executable, "-", "127.0.0.1", str(server.server_port), "1"],
                input=probe, env=env, text=True, capture_output=True, timeout=5,
            )
        finally:
            server.shutdown()
            worker.join(timeout=2)
    assert result.returncode == 0, result.stderr
    assert received == [("/api/health/auth", expected_key or None)]
