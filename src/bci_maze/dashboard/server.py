"""Loopback-only HTTP interface for the local EEG dashboard."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
from urllib.parse import urlsplit, parse_qs
import webbrowser

from .controller import ConflictError, DashboardController
from .experiments import list_reports, report_path


STATIC_ROOT = Path(__file__).parent / "static"


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], controller: DashboardController):
        super().__init__(address, DashboardHandler)
        self.controller = controller


class DashboardHandler(BaseHTTPRequestHandler):
    server: DashboardServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:
        pass

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(10)

    def handle(self) -> None:
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError):
            # Closing or reloading a dashboard tab can cancel an in-flight poll.
            self.close_connection = True

    def _send(self, code: int, data: bytes, mime: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code: int, data: dict) -> None:
        self._send(code, json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8")

    def _local_request(self, *, write: bool = False) -> bool:
        host = self.headers.get("Host", "")
        port = self.server.server_port
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if port == 80:
            allowed.update({"127.0.0.1", "localhost"})
        if host.lower() not in allowed:
            self.close_connection = True
            self._json(403, {"error": "控制台仅接受本机地址请求。"})
            return False
        if write:
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{host}":
                self.close_connection = True
                self._json(403, {"error": "控制台仅接受同源操作请求。"})
                return False
            if self.headers.get("Sec-Fetch-Site", "same-origin") not in {"same-origin", "none"}:
                self.close_connection = True
                self._json(403, {"error": "请从本机控制台页面提交操作。"})
                return False
        return True

    def do_GET(self) -> None:
        if not self._local_request():
            return
        path = urlsplit(self.path).path
        try:
            if path == "/api/state":
                self._json(200, self.server.controller.snapshot())
                return
            if path == "/api/config":
                self._json(200, self.server.controller.config())
                return
            if path == "/api/experiments":
                self._json(200, dict(reports=list_reports(self.server.controller.root),
                                     recordings=self.server.controller.config()["recordings"]))
                return
            if path in {"/api/experiment/report", "/api/experiment/asset"}:
                query = parse_qs(urlsplit(self.path).query)
                try:
                    candidate = report_path(self.server.controller.root, query.get("path", [""])[0],
                                            asset=path.endswith("/asset"))
                except ValueError as error:
                    self._json(400, {"error": str(error)})
                    return
                if path.endswith("/asset"):
                    self._send(200, candidate.read_bytes(), "image/png")
                else:
                    self._json(200, json.loads(candidate.read_text(encoding="utf-8")))
                return
            static_name = "index.html" if path == "/" else path.lstrip("/")
            candidate = (STATIC_ROOT / static_name).resolve()
            if not candidate.is_relative_to(STATIC_ROOT.resolve()) or not candidate.is_file():
                self._json(404, {"error": "未找到此页面或资源。"})
                return
            mime = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            if candidate.suffix in {".html", ".css", ".js", ".svg"}:
                mime += "; charset=utf-8"
            self._send(200, candidate.read_bytes(), mime)
        except Exception as error:
            self._json(500, {"error": f"{type(error).__name__}: {error}"})

    def do_POST(self) -> None:
        if not self._local_request(write=True):
            return
        if self.headers.get_content_type() != "application/json":
            self.close_connection = True
            self._json(415, {"error": "操作请求须使用 application/json。"})
            return
        try:
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("请求须使用 Content-Length。")
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > 65536:
                raise ValueError("请求内容大小须为 1–65536 字节。")
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("请求内容必须是 JSON 对象。")
        except (ValueError, UnicodeError) as error:
            self.close_connection = True
            self._json(400, {"error": str(error)})
            return
        controller = self.server.controller
        actions = {
            "/api/monitor/start": controller.start_monitor,
            "/api/job/start": controller.start_job,
            "/api/unity": controller.unity_request,
        }
        path = urlsplit(self.path).path
        try:
            if path == "/api/monitor/stop":
                if payload:
                    raise ValueError("停止请求应为空对象。")
                result = controller.stop_monitor()
            elif path == "/api/job/stop":
                if payload:
                    raise ValueError("停止请求应为空对象。")
                result = controller.stop_job()
            elif path in actions:
                result = actions[path](payload)
            else:
                self._json(404, {"error": "未知的控制台操作。"})
                return
            self._json(200, result)
        except ConflictError as error:
            self._json(409, {"error": str(error)})
        except (ValueError, FileNotFoundError) as error:
            self._json(400, {"error": str(error)})
        except Exception as error:
            self._json(500, {"error": f"{type(error).__name__}: {error}"})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MindMaze 本地实验控制台")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port 须为 1–65535")
    controller = DashboardController()
    try:
        server = DashboardServer(("127.0.0.1", args.port), controller)
    except OSError as error:
        print(f"控制台启动失败：{error}。可使用 --port 指定其他端口。", flush=True)
        return 2
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"MindMaze 控制台：{url}\n按 Ctrl+C 关闭控制台；运行中的任务将收到保存并停止请求。", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("正在停止监测和运行任务…", flush=True)
    finally:
        server.server_close()
        while True:
            try:
                if controller.close(timeout=10):
                    break
                print("任务仍在退出，等待保存记录并释放设备。可在采集提示窗口按 Esc 结束采集。", flush=True)
            except KeyboardInterrupt:
                print("正在完成资源清理，请等待任务退出。", flush=True)
    return 0
