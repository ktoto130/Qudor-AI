"""Safe local Quoridor viewer. Static UI lives in index.html."""
from __future__ import annotations
import argparse, csv, json, secrets, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import torch

from quoridor_ai.core.encoding import encode_batch
from quoridor_ai.core.engine import State, ACTION_SIZE, apply_unchecked, legal_actions
from quoridor_ai.model import PolicyValueNet

ROOT = Path(__file__).resolve().parent
TOKEN = secrets.token_urlsafe(24)
LOCK = threading.RLock()
GAME = {"state": State(), "model": None, "model_path": None, "mode": "human",
        "paused": False, "thinking": False, "speed": 1.0, "history": [], "probs": [], "error": ""}
MODEL_CACHE: list[str] | None = None


def safe_models() -> list[str]:
    """Return only regular checkpoint files contained by ROOT."""
    global MODEL_CACHE
    if MODEL_CACHE is None:
        paths = []
        for p in ROOT.rglob("*.pt"):
            rp = p.resolve()
            if rp.is_file() and ROOT in rp.parents:
                paths.append(rp.relative_to(ROOT).as_posix())
        MODEL_CACHE = sorted(paths, key=lambda x: ("smoke_run/latest.pt" not in x, x))
    return MODEL_CACHE


def load_model(rel: str) -> None:
    if rel not in safe_models():
        raise ValueError("checkpoint отсутствует в разрешённом каталоге")
    path = (ROOT / Path(rel)).resolve()
    if ROOT not in path.parents:
        raise ValueError("недопустимый путь")
    # weights_only is intentional: checkpoints containing arbitrary pickle data are rejected.
    data = torch.load(path, map_location="cpu", weights_only=True)
    config = data.get("config", {}) if isinstance(data, dict) else {}
    model = PolicyValueNet(config.get("channels", data.get("channels", 96)),
                           config.get("blocks", data.get("blocks", 8)))
    model.load_state_dict(data["model"])
    model.eval()
    with LOCK:
        GAME["model"], GAME["model_path"], GAME["error"] = model, rel, ""


def wall_indices(s: State) -> tuple[list[int], list[int]]:
    return ([i for i in range(64) if (s.h >> i) & 1],
            [i for i in range(64) if (s.v >> i) & 1])


def state_payload() -> dict:
    with LOCK:
        s = GAME["state"]
        acts = legal_actions(s) if s.winner is None else []
        hc, vc = wall_indices(s)
        return {"p0": s.p0, "p1": s.p1, "hc": hc, "vc": vc,
                "walls0": s.walls0, "walls1": s.walls1, "player": s.player,
                "ply": s.ply, "winner": s.winner, "mode": GAME["mode"],
                "model": GAME["model_path"], "paused": GAME["paused"],
                "thinking": GAME["thinking"], "history": GAME["history"],
                "probs": GAME["probs"], "error": GAME["error"],
                "legal": acts,
                "legalWallsH": [a - 81 for a in acts if 81 <= a < 145],
                "legalWallsV": [a - 145 for a in acts if 145 <= a < ACTION_SIZE]}


def ai_move() -> None:
    with LOCK:
        s, model = GAME["state"], GAME["model"]
        if (GAME["thinking"] or model is None or GAME["paused"] or s.winner is not None
                or (GAME["mode"] == "human" and s.player == 0)):
            return
        GAME["thinking"] = True
        version = s.ply
        speed = GAME["speed"]
    time.sleep(0.6 / max(0.25, speed))
    with LOCK:
        # Re-check after the delay: reset, pause, mode switch, or a completed game wins.
        s, model = GAME["state"], GAME["model"]
        if (s.ply != version or model is None or GAME["paused"] or s.winner is not None
                or (GAME["mode"] == "human" and s.player == 0)):
            GAME["thinking"] = False
            return
        actions = legal_actions(s)
        if not actions:
            GAME["thinking"] = False
            return
        with torch.inference_mode():
            logits, _ = model(torch.from_numpy(encode_batch([s])).float())
        vals = logits[0, actions].float()
        order = vals.argsort(descending=True)
        action = actions[int(order[0])]
        GAME["probs"] = [{"action": int(actions[int(i)]), "score": float(vals[int(i)])}
                          for i in order[:5]]
        GAME["state"] = apply_unchecked(s, action)
        GAME["history"].append({"ply": s.ply + 1, "player": s.player, "action": action})
        GAME["thinking"] = False
        again = GAME["state"].winner is None and (GAME["mode"] == "ai" or GAME["state"].player == 1)
    if again:
        threading.Thread(target=ai_move, daemon=True).start()


def metrics() -> list[dict]:
    rows = []
    for f in sorted(ROOT.rglob("metrics.csv")):
        try:
            with f.open(encoding="utf-8", newline="") as stream:
                rows.extend(dict(row, source=f.relative_to(ROOT).as_posix()) for row in csv.DictReader(stream))
        except (OSError, UnicodeError, csv.Error):
            continue
    return rows[-300:]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def json(self, data: dict | list, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def authorized(self) -> bool:
        origin = self.headers.get("Origin")
        if origin and origin not in {"http://127.0.0.1:8765", "http://localhost:8765"}:
            return False
        token = self.headers.get("X-Qudor-Token")
        return token in {None, TOKEN}

    def do_GET(self):
        try:
            path = urlparse(self.path).path
            if path == "/":
                body = (ROOT / "index.html").read_bytes()
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
            if path == "/api/token": return self.json({"token": TOKEN})
            if path == "/api/models": return self.json(safe_models())
            if path == "/api/state": return self.json(state_payload())
            if path == "/api/metrics": return self.json(metrics())
            return self.json({"error": "not found"}, 404)
        except Exception as exc:
            return self.json({"error": str(exc)}, 500)

    def do_POST(self):
        try:
            if not self.authorized(): return self.json({"error": "forbidden"}, 403)
            length = int(self.headers.get("Content-Length", "0"))
            if length > 1_000_000: return self.json({"error": "request too large"}, 413)
            raw = self.rfile.read(length)
            body = json.loads(raw or b"{}")
            path = urlparse(self.path).path
            if path == "/api/model":
                load_model(str(body.get("path", ""))); return self.json({"ok": True})
            if path == "/api/reset":
                with LOCK: GAME.update(state=State(), history=[], probs=[], paused=False, thinking=False, error="")
                return self.json({"ok": True})
            if path == "/api/settings":
                with LOCK:
                    GAME["mode"] = body.get("mode", GAME["mode"])
                    GAME["speed"] = max(.25, min(3.0, float(body.get("speed", GAME["speed"]))))
                    if "paused" in body: GAME["paused"] = bool(body["paused"])
                    should_start = GAME["mode"] == "ai" and not GAME["paused"]
                if should_start: threading.Thread(target=ai_move, daemon=True).start()
                return self.json({"ok": True})
            if path == "/api/step":
                with LOCK: GAME["paused"] = False
                threading.Thread(target=ai_move, daemon=True).start(); return self.json({"ok": True})
            if path == "/api/action":
                action = int(body["action"])
                with LOCK:
                    s = GAME["state"]
                    if GAME["thinking"] or GAME["mode"] != "human" or s.player != 0 or action not in legal_actions(s):
                        return self.json({"error": "illegal action"}, 400)
                    GAME["state"] = apply_unchecked(s, action); GAME["history"].append({"ply": s.ply + 1, "player": s.player, "action": action})
                threading.Thread(target=ai_move, daemon=True).start(); return self.json({"ok": True})
            return self.json({"error": "not found"}, 404)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            return self.json({"error": str(exc)}, 400)
        except Exception as exc:
            return self.json({"error": str(exc)}, 500)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--host", default="127.0.0.1"); parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(); print(f"Qudor viewer: http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__": main()
