"""Optional Stage C: a local vision-language model re-scores the top few clips. Off by default.

Runs through a local Ollama server (http://127.0.0.1:11434), so nothing leaves the machine and no
Python ML stack is needed. Only the top `top_n` clips (default 6) are judged, each with ONE image -
a contact sheet of 4 frames from the event - plus the clip's verbatim transcript. Output is constrained
with a JSON schema, validated, cached per clip, and blended into the ranking with a modest weight.
If the server is missing, the model is not pulled, or the answer is malformed, the clip keeps its
Stage B score and a note says why. The model's one-line description is labelled model-generated and
is never used as a caption or quote.

Suggested models for a 12 GB GPU (pull once with `ollama pull <name>`):
  qwen2.5vl:7b   ~6 GB  best quality of the small VLMs here      (default)
  gemma3:4b      ~3 GB  lighter, faster
  text only      any text model with --judge-images 0 (e.g. qwen2.5:7b)
"""
import base64, hashlib, json, shutil, subprocess, time, urllib.error, urllib.request
from pathlib import Path

SCHEMA = {
    "type": "object",
    "properties": {
        "humour": {"type": "integer", "minimum": 0, "maximum": 10},
        "surprise": {"type": "integer", "minimum": 0, "maximum": 10},
        "intensity": {"type": "integer", "minimum": 0, "maximum": 10},
        "standalone": {"type": "integer", "minimum": 0, "maximum": 10},
        "highlight_probability": {"type": "number", "minimum": 0, "maximum": 1},
        "what_happens": {"type": "string"},
    },
    "required": ["humour", "surprise", "intensity", "standalone", "highlight_probability", "what_happens"],
}

PROMPT = """You review moments from a Grand Theft Auto livestream for a clipping editor.
Judge ONLY from the transcript and the frames below. Do not guess what is not shown or said.
The image is 4 frames in time order (left to right) from around the key moment.

Transcript of the clip (verbatim, may contain recognition errors):
\"\"\"{text}\"\"\"
{summary}
Rate 0-10: humour (is it funny), surprise (unexpected), intensity (emotional energy),
standalone (understandable without having watched the stream).
highlight_probability: 0-1, how likely an editor would post this as a short clip.
what_happens: one plain sentence, max 25 words, describing only what is visible or said.
Answer with JSON only."""


class JudgeError(Exception):
    pass


def contact_sheet(video, start, dur, width=384, n=4):
    """One JPEG with n frames side by side (single ffmpeg call). Returns bytes or None."""
    exe = shutil.which("ffmpeg") or "ffmpeg"
    fps = max(n / max(dur, 0.5), 0.05)
    r = subprocess.run([exe, "-v", "error", "-ss", f"{max(0.0, start):.2f}", "-t", f"{dur:.2f}", "-i", str(video), "-an",
                        "-vf", f"fps={fps:.4f},scale={width}:-2,tile={n}x1", "-frames:v", "1", "-q:v", "4",
                        "-f", "image2", "-c:v", "mjpeg", "-"], capture_output=True)
    return r.stdout if r.returncode == 0 and r.stdout[:2] == b"\xff\xd8" else None


def validate(obj):
    if not isinstance(obj, dict):
        raise JudgeError("answer is not a JSON object")
    out = {}
    for k in ("humour", "surprise", "intensity", "standalone"):
        v = obj.get(k)
        if not isinstance(v, (int, float)) or not 0 <= v <= 10:
            raise JudgeError(f"{k} missing or out of range: {v!r}")
        out[k] = float(v)
    p = obj.get("highlight_probability")
    if not isinstance(p, (int, float)) or not 0 <= p <= 1:
        raise JudgeError(f"highlight_probability missing or out of range: {p!r}")
    out["highlight_probability"] = float(p)
    out["what_happens"] = str(obj.get("what_happens", ""))[:200]
    return out


def judge_score(v):
    """0..1 from a validated answer."""
    dims = (v["humour"] + v["surprise"] + v["intensity"] + v["standalone"]) / 40.0
    return round(0.5 * v["highlight_probability"] + 0.5 * dims, 3)


class OllamaJudge:
    def __init__(self, model="qwen2.5vl:7b", host="http://127.0.0.1:11434", images=True, timeout=180,
                 unload_after=True, cache_dir=None):
        self.model, self.host, self.images, self.timeout = model, host.rstrip("/"), images, timeout
        self.unload_after, self.cache = unload_after, Path(cache_dir) if cache_dir else None

    def _post(self, path, body):
        req = urllib.request.Request(self.host + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            raise JudgeError(f"ollama HTTP {e.code}: {e.read().decode(errors='replace')[:200]}")
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            raise JudgeError(f"ollama not reachable at {self.host} ({e})")

    def available(self):
        """(ok, reason). Checks the server and that the model is pulled."""
        try:
            req = urllib.request.Request(self.host + "/api/tags")
            with urllib.request.urlopen(req, timeout=5) as r:
                tags = json.loads(r.read().decode())
        except (urllib.error.URLError, OSError, TimeoutError, ValueError) as e:
            return False, f"ollama not reachable at {self.host} ({e})"
        names = {m.get("name") for m in tags.get("models", [])} | {m.get("model") for m in tags.get("models", [])}
        want = self.model if ":" in self.model else self.model + ":latest"
        if want not in names and self.model not in names:
            return False, f"model {self.model} not pulled (run: ollama pull {self.model})"
        return True, ""

    def ask(self, text, summary, image=None, last=False):
        prompt = PROMPT.format(text=text[:1500] or "(no speech)",
                               summary=(f"Setup line: \"{summary['setup']}\"\nReaction line: \"{summary['reaction']}\"\n"
                                        if summary else ""))
        msg = {"role": "user", "content": prompt}
        if image and self.images:
            msg["images"] = [base64.b64encode(image).decode()]
        key = hashlib.sha1((self.model + prompt).encode() + (image or b"")).hexdigest()
        if self.cache:
            f = self.cache / f"{key}.json"
            if f.exists():
                return json.loads(f.read_text(encoding="utf-8")), 0.0, True
        t = time.perf_counter()
        resp = self._post("/api/chat", {"model": self.model, "messages": [msg], "stream": False, "format": SCHEMA,
                                        "options": {"temperature": 0},
                                        # free the VRAM after the last clip so the GPU is back for gaming/encoding
                                        "keep_alive": 0 if (last and self.unload_after) else "5m"})
        dt = time.perf_counter() - t
        content = ((resp.get("message") or {}).get("content") or "").strip()
        try:
            v = validate(json.loads(content))
        except json.JSONDecodeError:
            raise JudgeError(f"answer is not JSON: {content[:120]!r}")
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)
            (self.cache / f"{key}.json").write_text(json.dumps(v), encoding="utf-8")
        return v, dt, False


def apply(results, video, judge, top_n=6, weight=0.35, notes=None, timings=None):
    """Judge the first top_n results (already ranked), blend, re-sort in place. Never raises."""
    notes = notes if notes is not None else []
    ok, why = judge.available()
    if not ok:
        notes.append(f"VLM judge skipped: {why}")
        return results
    calls, cached, secs = 0, 0, 0.0
    todo = results[:top_n]
    for i, r in enumerate(todo):
        img = None
        if judge.images and video:
            ev = r["boundary"]
            a, b = max(r["start"], ev["ev_start"] - 1), min(r["end"], ev["ev_end"] + 1)
            img = contact_sheet(video, a, max(1.0, b - a))
        try:
            v, dt, hit = judge.ask(r.get("text", ""), (r.get("dialogue") or {}).get("summary"), img, last=i == len(todo) - 1)
        except JudgeError as e:
            r["judge"] = {"error": str(e)}
            notes.append(f"VLM judge failed on {r['start']:.0f}s: {e}")
            continue
        calls += 0 if hit else 1; cached += hit; secs += dt
        js = judge_score(v)
        before = r["scores"]["entertainment"]
        r["scores"]["entertainment"] = round((1 - weight) * before + weight * js, 3)
        r["judge"] = dict(v, score=js, model=judge.model, images=bool(img), before=before,
                          note="model-generated description; not a quote, not for captions")
        r["evidence"].append(f"local VLM ({judge.model}) rates it {js:.2f}")
    results.sort(key=lambda x: -x["scores"]["entertainment"])
    if timings is not None:
        timings.update(judge_calls=calls, judge_cached=cached, judge_s=round(secs, 2))
    return results
