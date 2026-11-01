"""The dashboard's HTTP server: the page, the live cameras as MJPEG, robotd's state, and runs (upload a video with its
task, follow it live as server-sent events, fetch its files). Standard library only.

    GET  /                               the page
    GET  /api/state                      robotd and the cameras, now
    GET  /api/stream/<camera>            the camera as MJPEG: overhead (grid, grippers, objects), left_wrist, right_wrist
    GET  /api/jobs                       recent runs
    POST /api/jobs?name=<file>&task=...  start a run; the body is the video
    GET  /api/jobs/<id>                  a run and all its events
    GET  /api/jobs/<id>/events?since=N   its events after N as they happen (text/event-stream)
    GET  /api/jobs/<id>/files/<path>     a file of the run: the upload, the segmented preview, what the brain saw
    POST /api/jobs/<id>/stop             stop it (and the arms)
"""
import json
import mimetypes
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
TYPES = {".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".html": "text/html",
         ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm", ".jpg": "image/jpeg",
         ".svg": "image/svg+xml", ".png": "image/png"}


def inside(root, rel):
    """The path of rel under root, or None if it would leave root."""
    path = os.path.realpath(os.path.join(root, rel))
    return path if path.startswith(os.path.realpath(root) + os.sep) and os.path.isfile(path) else None


class Handler(BaseHTTPRequestHandler):
    server_version = "understudy-ui"
    live = None  # set by serve()
    runner = None

    def log_message(self, fmt, *args):  # quiet: the page polls
        pass

    # ---- helpers ---------------------------------------------------------------------------------------------------
    def send_json(self, obj, code=200):
        body = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path):
        ext = os.path.splitext(path)[1].lower()
        ctype = TYPES.get(ext) or mimetypes.guess_type(path)[0] or "application/octet-stream"
        size = os.path.getsize(path)
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        m = re.match(r"bytes=(\d*)-(\d*)", rng or "")
        if m and (m.group(1) or m.group(2)):  # videos need range requests to seek
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:
                start = max(0, size - int(m.group(2)))
            end = min(end, size - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 16, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    # ---- GET ---------------------------------------------------------------------------------------------------------
    def do_GET(self):
        url = urlparse(self.path)
        path, q = url.path, parse_qs(url.query)
        try:
            if path in ("/", "/index.html"):
                return self.send_file(os.path.join(STATIC, "index.html"))
            if path.startswith("/static/"):
                f = inside(STATIC, unquote(path[len("/static/"):]))
                return self.send_file(f) if f else self.send_error(404)
            if path == "/api/state":
                state = self.live.snapshot()
                state["busy"] = self.runner.busy()
                state["job"] = self.runner.current.summary() if self.runner.current else None
                return self.send_json(state)
            m = re.fullmatch(r"/api/stream/(\w+)", path)
            if m:
                return self.stream(m.group(1))
            if path == "/api/jobs":
                return self.send_json(self.runner.history())
            m = re.fullmatch(r"/api/jobs/([\w-]+)(/events|/files/(.+))?", path)
            if m:
                job = self.runner.get(m.group(1))
                if job is None:
                    return self.send_error(404, "no such run")
                if m.group(2) == "/events":
                    since = int((q.get("since") or ["-1"])[0])
                    resumed = self.headers.get("Last-Event-ID")  # the browser reconnecting mid-run
                    if resumed and resumed.isdigit():
                        since = max(since, int(resumed))
                    return self.events(job, since)
                if m.group(3):
                    f = inside(job.dir, unquote(m.group(3)))
                    return self.send_file(f) if f else self.send_error(404)
                return self.send_json(dict(job.summary(), events=job.events))
            self.send_error(404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def stream(self, cam):
        if cam not in self.live.frames:
            return self.send_error(404, "no such camera")
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        seq = -1
        while self.live.running:
            seq, data = self.live.wait_frame(cam, seq)
            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                             + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
            self.wfile.flush()

    def events(self, job, since):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b"retry: 1000\n\n")  # reconnect after a second, not the default three
        last = since
        while True:
            new = job.events[last + 1:] or job.wait(last)
            for e in new:
                self.wfile.write(f"id: {e['i']}\ndata: {json.dumps(e, default=str)}\n\n".encode())
                last = e["i"]
            if not new:
                self.wfile.write(b": still here\n\n")  # keeps proxies and the browser from giving up
            self.wfile.flush()
            if job.status != "running" and last >= len(job.events) - 1:
                return

    # ---- POST --------------------------------------------------------------------------------------------------------
    def do_POST(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        try:
            if url.path == "/api/jobs":
                size = int(self.headers.get("Content-Length") or 0)
                if size <= 0:
                    return self.send_json({"error": "no video in the request"}, 400)
                data = self.rfile.read(size)
                try:
                    job = self.runner.start(data, (q.get("name") or ["video.mp4"])[0], (q.get("task") or [""])[0])
                except (ValueError, RuntimeError) as e:
                    return self.send_json({"error": str(e)}, 409 if isinstance(e, RuntimeError) else 400)
                return self.send_json(job.summary(), 201)
            m = re.fullmatch(r"/api/jobs/([\w-]+)/stop", url.path)
            if m:
                job = self.runner.current
                if job is None or job.id != m.group(1):
                    return self.send_json({"error": "that run isn't going"}, 409)
                return self.send_json({"stopped": self.runner.stop()})
            self.send_error(404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


def serve(live, runner, host="127.0.0.1", port=8000):
    Handler.live, Handler.runner = live, runner
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    return httpd
