"""Optional VLM judge, tested against a mock Ollama HTTP server (no model is run here)."""
import json, subprocess, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import pytest
from conftest import needs_ffmpeg
from highlight import judge as J


class Mock:
    def __init__(self, answers, models=("qwen2.5vl:7b",)):
        self.answers, self.models, self.chats = list(answers), models, []
        mock = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj):
                b = json.dumps(obj).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

            def do_GET(self):
                self._send({"models": [{"name": m, "model": m} for m in mock.models]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                mock.chats.append(body)
                ans = mock.answers.pop(0) if mock.answers else {}
                self._send({"message": {"role": "assistant", "content": ans if isinstance(ans, str) else json.dumps(ans)}})

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()


def A(h, s, i, st, p, what="streamer laughs after jumping a bridge"):
    return {"humour": h, "surprise": s, "intensity": i, "standalone": st, "highlight_probability": p, "what_happens": what}


def R(start, ent):
    return {"start": start, "end": start + 20, "text": "no way hahaha", "scores": {"entertainment": ent},
            "evidence": [], "boundary": {"ev_start": start + 8, "ev_end": start + 12},
            "dialogue": {"summary": {"setup": "watch this", "reaction": "no way"}}}


def test_judge_reranks_and_labels_output(tmp_path):
    m = Mock([A(1, 1, 1, 2, 0.05), A(9, 8, 8, 9, 0.95)])
    try:
        res = [R(0, 0.60), R(100, 0.55)]
        notes, tm = [], {}
        J.apply(res, None, J.OllamaJudge(host=m.url, images=False), top_n=6, weight=0.35, notes=notes, timings=tm)
        assert [r["start"] for r in res] == [100, 0]
        assert res[0]["judge"]["before"] == 0.55 and "not a quote" in res[0]["judge"]["note"]
        assert tm["judge_calls"] == 2 and not notes
        body = m.chats[0]
        assert body["format"] == J.SCHEMA and body["options"]["temperature"] == 0 and "images" not in body["messages"][0]
        assert body["keep_alive"] == "5m" and m.chats[-1]["keep_alive"] == 0   # VRAM freed after the last clip
        assert "no way hahaha" in body["messages"][0]["content"]
    finally:
        m.close()


def test_only_top_n_are_judged():
    m = Mock([A(5, 5, 5, 5, 0.5)] * 5)
    try:
        res = [R(i * 50, 0.9 - i * 0.1) for i in range(5)]
        J.apply(res, None, J.OllamaJudge(host=m.url, images=False), top_n=2)
        assert len(m.chats) == 2 and sum("judge" in r for r in res) == 2
    finally:
        m.close()


def test_unreachable_server_and_missing_model_leave_scores_alone():
    res = [R(0, 0.6)]
    notes = []
    J.apply(res, None, J.OllamaJudge(host="http://127.0.0.1:9", timeout=2), notes=notes)
    assert res[0]["scores"]["entertainment"] == 0.6 and "not reachable" in notes[0]
    m = Mock([], models=("llama3:8b",))
    try:
        notes = []
        J.apply(res, None, J.OllamaJudge(host=m.url), notes=notes)
        assert "not pulled" in notes[0] and not m.chats
    finally:
        m.close()


@pytest.mark.parametrize("bad", [A(11, 1, 1, 1, 0.5), A(1, 1, 1, 1, 1.5), "not json at all", {"humour": 3}])
def test_malformed_answers_are_rejected(bad):
    m = Mock([bad])
    try:
        res = [R(0, 0.6)]
        notes = []
        J.apply(res, None, J.OllamaJudge(host=m.url, images=False), notes=notes)
        assert res[0]["scores"]["entertainment"] == 0.6 and "error" in res[0]["judge"] and notes
    finally:
        m.close()


def test_cache_avoids_repeat_calls(tmp_path):
    m = Mock([A(7, 7, 7, 7, 0.7)])
    try:
        jd = J.OllamaJudge(host=m.url, images=False, cache_dir=tmp_path / "c")
        a, b = [R(0, 0.5)], [R(0, 0.5)]
        J.apply(a, None, jd); tm = {}
        J.apply(b, None, jd, timings=tm)
        assert len(m.chats) == 1 and tm["judge_cached"] == 1
        assert a[0]["scores"] == b[0]["scores"]
    finally:
        m.close()


@needs_ffmpeg
def test_contact_sheet_is_one_jpeg_with_four_frames(tmp_path):
    v = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=s=320x180:r=10", "-t", "6",
                    "-c:v", "libx264", str(v)], check=True)
    jpg = J.contact_sheet(v, 1, 4)
    assert jpg and jpg[:2] == b"\xff\xd8"
    import cv2, numpy as np
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] == 4 * 384


@needs_ffmpeg
def test_engine_with_judge_end_to_end(tmp_path):
    import synth
    from highlight import engine
    x, words, msgs, vis, _ = synth.episode(9, dur=120, n_events=2, n_decoys=1)
    synth.write_wav(tmp_path / "a.wav", x)
    synth.render_video(tmp_path / "v.mp4", 120, vis, size=(320, 180), fps=10, audio_wav=tmp_path / "a.wav")
    m = Mock([A(8, 8, 8, 8, 0.9)] * 10)
    try:
        cfg = engine.Config(top=3, cam="none", hwaccel="none", judge="ollama", judge_host=m.url, judge_top=2)
        res = engine.analyse(str(tmp_path / "v.mp4"), tmp_path / "a.wav", words, 120.0, cfg, chat_msgs=msgs, d=tmp_path)
        judged = [h for h in res["highlights"] if "judge" in h]
        assert judged and all(h["judge"]["images"] for h in judged)
        assert "images" in m.chats[0]["messages"][0] and res["timings"]["judge_calls"] == len(m.chats)
    finally:
        m.close()
