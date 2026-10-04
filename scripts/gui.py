"""A local web page for trying speculative decoding.

    python -m scripts.gui            # then open http://127.0.0.1:7860

Standard library only. The models load in a background thread while the page is already
up; generation requests are served one at a time, because two at once would not fit next
to the 1.7B on an 8 GB machine. Results stream back as one JSON line per method, so the
page shows plain decoding before the speculative runs finish.
"""
import argparse
import json
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from scripts.demo import Demo

PAGE = Path(__file__).with_name("gui.html")
LIMITS = {"max_new": (1, 256), "k": (1, 8)}


class State:
    demo: Demo | None = None
    error: str | None = None
    lock = threading.Lock()


def load() -> None:
    try:
        State.demo = Demo()
    except Exception:  # surfaced on the page instead of a dead server
        State.error = traceback.format_exc()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        pass

    def send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/":
            self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/status":
            demo = State.demo
            body = {"ready": demo is not None, "error": State.error, "models": demo.names if demo else None}
            self.send(200, json.dumps(body).encode(), "application/json")
        else:
            self.send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        if self.path != "/generate":
            return self.send(404, b"not found", "text/plain")
        if State.demo is None:
            return self.send(503, b"models are still loading", "text/plain")
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = str(request.get("prompt", "")).strip()
        if not prompt:
            return self.send(400, b"empty prompt", "text/plain")
        params = {key: min(max(int(request.get(key, lo)), lo), hi) for key, (lo, hi) in LIMITS.items()}
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        with State.lock:
            try:
                for result in State.demo.run(prompt, params["max_new"], params["k"], bool(request.get("raw"))):
                    self.wfile.write((json.dumps(result) + "\n").encode())
                    self.wfile.flush()
            except Exception as e:
                self.wfile.write((json.dumps({"error": str(e)}) + "\n").encode())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    threading.Thread(target=load, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Speculative decoding playground: {url}  (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
