"""Zero-framework local demo API and web UI. Run: python server.py"""
import json
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd
from src.pipeline import ROOT, predict_history

WEB = ROOT / "web"
MODELS = {name: json.loads((ROOT / "models" / f"{name}.json").read_text()) for name in ("velocity","combined")}
DATA = pd.read_csv(ROOT / "data" / "synthetic_repetitions.csv")
METRICS = json.loads((ROOT / "metrics.json").read_text())


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def respond(self, status, payload):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        route = urlparse(self.path).path
        if route == "/api/metrics": return self.respond(200, METRICS)
        if route == "/api/demo":
            subset = DATA[(DATA.participant == "P20") & (DATA.exercise == "squat") & (DATA.set_id == "P20-squat-0")]
            return self.respond(200, {"source":"SYNTHETIC","repetitions":subset.to_dict(orient="records")})
        if route == "/api/docs":
            return self.respond(200, {"POST /api/predict":{"body":{"model":"combined | velocity","history":"1-4 completed repetition records with exercise, load_fraction, rep, velocity, velocity_loss, tremor_amplitude, tremor_band_power"},"returns":"estimated_rir, warning, status"},"GET /api/demo":"sample synthetic set","GET /api/metrics":"held-out synthetic evaluation"})
        return super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != "/api/predict": return self.respond(404,{"error":"not found"})
        try:
            length = int(self.headers.get("Content-Length","0"))
            if length > 30000: raise ValueError("request too large")
            body = json.loads(self.rfile.read(length))
            name = body.get("model","combined")
            history = body["history"]
            if name not in MODELS or not isinstance(history,list) or not 1 <= len(history) <= 4:
                raise ValueError("model must be velocity or combined; history must have 1-4 records")
            for row in history:
                if row["exercise"] not in ("squat","bench press") or not .3 <= float(row["load_fraction"]) <= 1 or int(row["rep"]) < 1:
                    raise ValueError("invalid exercise, load_fraction, or rep")
                for key in MODELS[name]["channels"]:
                    if not 0 <= float(row[key]) <= 10: raise ValueError(f"invalid {key}")
            estimate = predict_history(history,MODELS[name])
            self.respond(200,{"status":"ok","model":name,"estimated_rir":estimate,"warning":estimate<=2,
                              "message":"Near endpoint (experimental)" if estimate<=2 else "Continue observing", "source":"synthetic-trained prototype"})
        except (ValueError,TypeError,KeyError,json.JSONDecodeError) as exc:
            self.respond(400,{"status":"unavailable","error":str(exc)})


if __name__ == "__main__":
    print("Open http://127.0.0.1:8000")
    ThreadingHTTPServer(("127.0.0.1",8000), Handler).serve_forever()
