"""Speech-aware boundaries: Silero VAD (bundled in faster-whisper) + Auto-Editor-style fallback + snapping rules."""
import builtins
import numpy as np
import pytest
from highlight import boundaries as B, speech

SR = 16000


def _b(start, end, ev_s, ev_e):
    return B.Bounds(start, end, ev_s, ev_e, "test", "test", False, False, False)


# ---------- snapping rules (hand-built segments)
def test_start_inside_utterance_moves_back_to_onset():
    segs = [(8.0, 12.0), (20.0, 24.0)]
    b = B.snap_to_speech(_b(10.0, 30.0, 20.0, 24.0), segs, 100, min_len=5, max_len=40)
    assert b.start == pytest.approx(8.0 - 0.12) and not b.mid_speech_start and "speech onset" in b.start_reason


def test_long_utterance_start_moves_forward_instead():
    segs = [(2.0, 12.0), (13.0, 15.0), (20.0, 24.0)]   # onset 8 s back: too far, skip past it
    b = B.snap_to_speech(_b(10.0, 30.0, 20.0, 24.0), segs, 100, min_len=5, max_len=40)
    assert b.start == pytest.approx(13.0 - 0.12) and "cut-off utterance" in b.start_reason


def test_end_inside_utterance_extends_to_its_end():
    segs = [(20.0, 24.0), (28.0, 31.0)]
    b = B.snap_to_speech(_b(10.0, 29.5, 20.0, 24.0), segs, 100, min_len=5, max_len=40)
    assert b.end == pytest.approx(31.12) and not b.mid_speech_end


def test_end_extension_respects_max_len_by_pulling_back():
    segs = [(20.0, 24.0), (29.0, 45.0)]
    b = B.snap_to_speech(_b(10.0, 30.0, 20.0, 24.0), segs, 100, min_len=5, max_len=21)
    assert b.end == pytest.approx(29.0 - 0.12) and b.end - b.start <= 21


def test_trailing_and_leading_silence_trimmed_but_event_kept():
    segs = [(12.0, 18.0), (20.0, 23.0)]
    b = B.snap_to_speech(_b(10.5, 30.0, 20.0, 23.0), segs, 100, min_len=5, max_len=40)
    assert b.start == pytest.approx(12.0 - 0.12) and b.end == pytest.approx(23.5)
    assert b.start <= b.ev_start and b.end >= b.ev_end


def test_never_breaks_min_length_or_event():
    segs = [(5.0, 6.0)]
    b0 = _b(0.0, 15.0, 3.0, 13.0)
    b = B.snap_to_speech(b0, segs, 100, min_len=15, max_len=40)
    assert b.end - b.start >= 15 and b.start <= 3.0 and b.end >= 13.0


def test_no_segments_is_a_no_op():
    b0 = _b(1.0, 20.0, 5.0, 8.0)
    assert B.snap_to_speech(b0, [], 100) is b0


# ---------- segment detection
def test_merge_windows():
    assert speech.merge_windows([(10, 20), (20.5, 30), (50, 60), (-5, 2)]) == [(0.0, 2), (10, 30), (50, 60)]


def test_energy_fallback_matches_auto_editor_rules():
    x = np.zeros(SR * 10, np.float32)
    x[SR * 2: SR * 4] = 0.3                       # loud 2-4 s
    x[int(SR * 4.15): int(SR * 4.3)] = 0.3       # 0.15 s gap -> closed (min cut 0.2 s)
    x[SR * 7: int(SR * 7.04)] = 0.3              # 40 ms blip -> kept as 0.44 s with margins
    segs = speech.energy_segments(x)
    assert segs[0][0] == pytest.approx(1.8, abs=0.03) and segs[0][1] == pytest.approx(4.5, abs=0.03)
    assert len(segs) == 2


def test_silero_finds_voice_like_audio_and_ignores_silence():
    import synth
    rng = np.random.default_rng(0)
    x = synth.bed(30, rng)
    synth.add(x, synth.speech(6, rng, level=0.1), 10)
    segs, used = speech.segments(x.astype(np.float32), [(0, 30)])
    assert used == "silero"
    covered = sum(max(0.0, min(e, 15.5) - max(s, 10.5)) for s, e in segs)  # speech is split at its pauses
    assert covered >= 0.7 * 5.0
    assert not any(s < 9.0 for s, e in segs) and not any(e > 17.5 for s, e in segs)


def test_falls_back_to_energy_without_faster_whisper(monkeypatch):
    real = builtins.__import__

    def fake(name, *a, **k):
        if name.startswith("faster_whisper"):
            raise ImportError("not installed")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    x = np.zeros(SR * 5, np.float32); x[SR: SR * 2] = 0.3
    segs, used = speech.segments(x, [(0, 5)])
    assert used == "energy" and segs
    with pytest.raises(ImportError):
        speech.segments(x, [(0, 5)], method="silero")


def test_engine_clip_edges_are_not_mid_utterance(tmp_path):
    import synth
    from highlight import engine
    x, words, msgs, _, _ = synth.episode(13, dur=150, n_events=2, n_decoys=1)
    synth.write_wav(tmp_path / "a.wav", x)
    res = engine.analyse(None, tmp_path / "a.wav", words, 150.0, engine.Config(visual=False), chat_msgs=msgs)
    assert res["timings"]["speech_method"] == "silero" and res["highlights"]
    flagged = [h for h in res["highlights"] if h["boundary"]["mid_speech_start"] or h["boundary"]["mid_speech_end"]]
    assert len(flagged) <= len(res["highlights"]) // 2
    off = engine.analyse(None, tmp_path / "a.wav", words, 150.0, engine.Config(visual=False, speech_snap="off"), chat_msgs=msgs)
    assert "speech_s" not in off["timings"]


def test_cli_exposes_speech_snap():
    import highlights, inspect
    assert '"--speech-snap"' in inspect.getsource(highlights.main) and "speech_snap=a.speech_snap" in inspect.getsource(highlights.main)
