"""Stage B building blocks: boundaries, scoring, dedupe, visual primitives."""
import numpy as np
import pytest
from conftest import W, needs_ffmpeg
import synth
from highlight import audio, boundaries, candidates, chat, dedupe, scoring, text, visual


def _score_with_peak(n, peak, width=3):
    s = np.zeros(n); s[peak - width:peak + width + 1] = 1.0; s[peak] = 2.0
    return s


# ---------- boundaries
def test_boundary_starts_at_setup_sentence_and_ends_after_reaction():
    words = (W("random chatter about cars here.", 10) + W("okay watch this jump over the bridge", 40)
             + W("no way that worked", 47) + W("anyway back to the shop.", 52) + W("later stuff", 70))
    n = 90
    b = boundaries.compute(47, _score_with_peak(n, 47), words, np.full(n, -30.0), n, min_len=10, max_len=30)
    assert b.start == pytest.approx(40 - 0.25, abs=0.01)      # the setup sentence, not peak - 22 s
    assert 50 <= b.end <= 54.5                                # after the reaction, at a sentence end
    assert not b.mid_sentence_start and not b.dangling_start
    assert 10 <= b.end - b.start <= 30


def test_boundary_length_limits_and_no_words():
    n = 200
    db = np.full(n, -30.0); db[80] = -60
    b = boundaries.compute(90, _score_with_peak(n, 90), [], db, n, min_len=15, max_len=40)
    assert 15 <= b.end - b.start <= 40 and b.start <= 87 and b.end >= 94
    assert "quietest" in b.start_reason


def test_boundary_respects_video_edges():
    n = 30
    b = boundaries.compute(2, _score_with_peak(n, 3), [], np.full(n, -30.0), n, min_len=15, max_len=40)
    assert b.start >= 0 and b.end <= n and b.end - b.start >= 15


def test_dangling_opener_is_avoided_when_possible():
    words = W("so then he went", 30) + W("| | the real setup starts here now", 33) + W("what the", 41)
    n = 80
    b = boundaries.compute(41, _score_with_peak(n, 41, 1), words, np.full(n, -30.0), n, min_len=5, max_len=30)
    assert text.norm([w for w in words if w["s"] >= b.start][0]["w"]) == "the"


# ---------- scoring
def test_scoring_dimensions_reflect_the_event_type():
    x, words, msgs, _, labels = synth.episode(11, dur=200, n_events=3, n_decoys=0)
    f = audio.features(x=x.astype(np.float32)); n = len(f["db"])
    feats = {"audio": f, "text": text.features(words, n), "chat": chat.features(msgs, n), "heat": np.zeros(n)}
    st = candidates.run(f, feats["text"], feats["chat"], shortlist=6)
    by_kind = {}
    for lab in labels:
        p = min(st.shortlist, key=lambda q: abs(q - lab["peak"]))
        b = boundaries.compute(p, st.score, words, f["db"], n)
        dims, conf, ev, mods = scoring.score(st, feats, b, words, None, st.score[p], st.score.max())
        by_kind[lab["kind"]] = dims
        assert set(dims) == set(scoring.DIMENSIONS)
        assert all(0 <= v <= 1 for v in dims.values()) and 0.05 <= conf <= 0.95 and ev
    assert by_kind["laugh"]["humour"] > by_kind["shout"]["humour"]
    assert by_kind["shout"]["emotional_intensity"] > 0.5


# ---------- dedupe
def _c(i, s, e, txt="", h=None):
    return {"id": f"j#{i}", "start": s, "end": e, "text": txt, "hashes": h or []}


def test_dedupe_rules():
    long = "the quick brown fox jumps over the lazy dog while the streamer screams about the bridge jump"
    other = "completely different words describing a car chase through the city at night with police"
    ranked = [_c(1, 10, 30, long), _c(2, 20, 35), _c(3, 100, 120, long), _c(4, 200, 220, other),
              _c(5, 300, 320, "", [1, 2, 3]), _c(6, 400, 420, "", [1, 2, 0xFFFF])]
    kept, dropped = dedupe.dedupe(ranked)
    assert [c["id"] for c in kept] == ["j#1", "j#4", "j#5"]
    reasons = {c["id"]: c["duplicate_reason"] for c in dropped}
    assert "overlaps" in reasons["j#2"] and "speech" in reasons["j#3"] and "frame" in reasons["j#6"]
    kept2, dropped2 = dedupe.dedupe([_c(7, 10, 30, long)], others=[_c(9, 500, 520, long)])
    assert not kept2 and "other job" in dropped2[0]["duplicate_reason"]


# ---------- visual
def test_dhash_and_motion_regions():
    rng = np.random.default_rng(0)
    a = (rng.random((40, 64)) * 255).astype(np.uint8)
    assert visual.hamming(visual.dhash(a), visual.dhash(a)) == 0
    assert visual.hamming(visual.dhash(a), visual.dhash(255 - a)) > 40
    fr = np.stack([a, a, a]).copy()
    fr[1, 0:10, 50:64] = 255 - fr[1, 0:10, 50:64]          # change only inside the cam box
    game, cam, _ = visual.motion(fr, box=(50 / 64, 0, 14 / 64, 10 / 40))
    assert game.max() == 0 and cam.max() > 50


@needs_ffmpeg
def test_frame_sampling_and_event_motion(tmp_path):
    vis = [("game", 6.0, 2.0)]
    synth.render_video(tmp_path / "v.mp4", 12, vis, size=(320, 180), fps=10)
    fr = visual.frames(tmp_path / "v.mp4", 0, 12, fps=2, hwaccel="none")
    assert fr.shape[0] in (23, 24, 25) and fr.shape[2] == visual.WIDTH
    base = visual.baseline(tmp_path / "v.mp4", 12, [(5, 9)], n=2, win=2.0, hwaccel="none")
    out = visual.analyse(tmp_path / "v.mp4", 3, 11, 6, 8, base, hwaccel="none")
    assert out["game_motion_z"] > 3 and len(out["hashes"]) == 3
