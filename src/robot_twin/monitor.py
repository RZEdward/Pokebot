"""Serve a loopback-only live training dashboard for a robot-twin run."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from robot_twin.env import ROOT

PAGE = Path(__file__).with_name("static") / "monitor.html"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--port", type=int, default=8770)
    a = ap.parse_args()
    run = a.run_dir.resolve()
    if not run.is_relative_to(ROOT):
        raise SystemExit("For safety, run directory must be inside this repository.")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                payload = PAGE.read_bytes()
                kind = "text/html; charset=utf-8"
            elif path == "/api/state":
                f = run / "telemetry.json"
                payload = (
                    f.read_bytes() if f.exists() else json.dumps({"status": "waiting"}).encode()
                )
                kind = "application/json"
            elif path == "/latest.png":
                f = run / "latest.png"
                if not f.exists():
                    self.send_error(404)
                    return
                payload = f.read_bytes()
                kind = "image/png"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"Robot twin monitor: http://127.0.0.1:{a.port} (run: {run})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
