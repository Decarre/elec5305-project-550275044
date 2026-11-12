"""Run the local recording demo with the project's frozen attention model."""

import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock, Timer
import webbrowser

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODELS = ROOT / "output/demo_slakh160"
os.environ.setdefault("MPLBACKEND", "Agg")

import torch
from instrument_localization.inference import predict_audio

SAMPLES = {name: MODELS / (name + ".wav")
           for name in ("success", "failure")}
ASSETS = {"/": HERE / "index.html", "/app.js": HERE / "app.js", "/cases.json": MODELS / "cases.json"}
ASSETS.update({"/audio/" + name: path for name, path in SAMPLES.items()})
PREDICTION_LOCK = Lock()
MAX_UPLOAD = 25 * 1024 * 1024


class DemoHandler(BaseHTTPRequestHandler):
    def respond(self, status, body, mime):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status, data):
        self.respond(status, json.dumps(data).encode(), "application/json")

    def do_GET(self):
        path = ASSETS.get(self.path)
        if path is None or not path.is_file():
            self.send_error(404)
            return
        mime = "audio/wav" if path.suffix == ".wav" else mimetypes.guess_type(path.name)[0]
        self.respond(200, path.read_bytes(), mime)

    def do_POST(self):
        local_host = "127.0.0.1:{}".format(self.server.server_port)
        if (self.headers.get("Host") != local_host
                or self.headers.get("Origin") not in (None, "http://" + local_host)):
            self.send_error(403)
            return
        prefix = "/predict/"
        source = self.path[len(prefix):]
        if not self.path.startswith(prefix) or source not in (*SAMPLES, "upload"):
            self.send_error(404)
            return
        try:
            with TemporaryDirectory(prefix="elec5305-demo-") as directory:
                audio = SAMPLES.get(source)
                if audio is None:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 12 <= size <= MAX_UPLOAD:
                        self.send_json(400, {"error": "Choose a WAV file up to 25 MB."})
                        return
                    data = self.rfile.read(size)
                    if data[:4] not in (b"RIFF", b"RF64", b"RIFX") or data[8:12] != b"WAVE":
                        self.send_json(400, {"error": "The selected file is not a WAV."})
                        return
                    audio = Path(directory) / "upload.wav"
                    audio.write_bytes(data)
                with PREDICTION_LOCK:
                    result = predict_audio(audio, MODELS / "attention_seed5305.pt",
                                           MODELS / "attention_5305_calibration.json", directory)
                self.send_json(200, {key: result[key] for key in ("duration_seconds", "classes", "intervals")})
        except Exception as error:
            self.log_error("Inference failed: %s", error)
            self.send_json(422, {"error": "Prediction failed. Check that the WAV opens and try again."})


if __name__ == "__main__":
    import argparse
    from urllib.request import urlopen

    parser = argparse.ArgumentParser()
    parser.add_argument("--open", action="store_true", help="Open the demo in your browser")
    parser.add_argument("--port", type=int, default=5306)
    args = parser.parse_args()
    torch.set_num_threads(4)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), DemoHandler)
    except OSError:
        if not args.open:
            raise
        url = "http://127.0.0.1:{}/".format(args.port)
        with urlopen(url, timeout=2) as response:
            if b"<title>Instrument Activity | ELEC5305</title>" not in response.read(4096):
                raise
        webbrowser.open(url)
        raise SystemExit(0)
    with server:
        url = "http://127.0.0.1:{}/".format(server.server_port)
        print("Demo ready at " + url, flush=True)
        if args.open:
            Timer(0.2, webbrowser.open, args=(url,)).start()
        server.serve_forever()
