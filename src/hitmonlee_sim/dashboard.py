"""Serve the local-only live training monitor."""

from __future__ import annotations

import argparse
import ipaddress
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

STATIC_PAGE = Path(__file__).with_name("static") / "monitor.html"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65_535:
        parser.error("--port must be between 1 and 65535.")
    if args.host.casefold() != "localhost":
        try:
            address = ipaddress.ip_address(args.host)
        except ValueError:
            parser.error("--host must be localhost or an IPv4 loopback address.")
        if address.version != 4 or not address.is_loopback:
            parser.error("--host must be localhost or an IPv4 loopback address.")
    return args


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.resolve()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/state":
                telemetry_path = run_dir / "telemetry.json"
                if telemetry_path.is_file():
                    try:
                        payload = telemetry_path.read_bytes()
                    except OSError:
                        payload = b"{}"
                else:
                    payload = json.dumps({"status": "waiting", "run_dir": str(run_dir)}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if path in ("/", "/index.html"):
                payload = STATIC_PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_error(404)

        def log_message(self, format: str, *args: object) -> None:
            print(f"[monitor] {self.address_string()} - {format % args}")

    url = f"http://{args.host}:{args.port}"
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving local training monitor for {run_dir} at {url}")
    print("This server binds to localhost by default; press Ctrl+C to stop.")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("Stopping monitor.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
