"""Stage A building blocks: util, audio, transcript, chat, fusion."""
import json
import numpy as np
import pytest
from conftest import W
import synth
from highlight import audio, candidates, chat, text, util


# ---------- util
def test_smooth_has_no_edge_bias():
    a = np.full(50, -30.0)
    assert np.allclose(util.smooth(a, 30), -30.0)  # zero-padding would pull the ends toward 0


def test_robust_z_flat_and_outlier():
    assert np.all(util.robust_z(np.ones(10)) == 0)
    z = util.robust_z(np.r_[np.zeros(99), 10.0])
    assert z[-1] == z.max() and z[-1] > 5


def test_shift_earlier():
    assert list(util.shift_earlier([0, 0, 0, 5, 0], 2)) == [0, 5, 0, 0, 0]


def test_peaks_nms():
    s = np.zeros(100); s[10] = 5; s[14] = 4; s[60] = 3
    assert util.peaks(s, 5, 10) == [10, 60]


# ---------- audio
def _feats(sig, rng, pad=2):
    x = synth.bed(len(sig) / synth.SR + 2 * pad, rng)
    synth.add(x, sig, pad)
    return audio.features(x=x.astype(np.float32))


def test_laughter_scores_above_speech():
    rng = np.random.default_rng(0)
    lf = _feats(synth.laugh(3.5, rng), rng)
    sf = _feats(synth.speech(20, rng), rng)
    assert lf["laugh"].max() > 0.6
    assert np.percentile(sf["laugh"], 95) < 0.3


def test_sustained_rhythm_is_suppressed():
    l = np.r_[np.zeros(5), np.full(20, 0.8), np.zeros(5), np.full(4, 0.8)]
    out = audio.suppress_sustained(l)
    assert out[10] == pytest.approx(0.16) and out[-2] == 0.8


def test_loud_shout_has_jump_onset_and_brightness():
    rng = np.random.default_rng(1)
    x = synth.bed(60, rng)
    synth.add(x, synth.speech(25, rng, level=0.05), 5)
    synth.add(x, synth.shout(3, rng), 40)
    f = audio.features(x=x.astype(np.float32))
    assert 39 <= int(np.argmax(f["jump"])) <= 43
    assert f["onset"][39:43].max() > np.percentile(f["onset"], 90)
    assert f["bright"][40:43].mean() > np.median(f["bright"][5:30])


def test_read_wav_rejects_wrong_format(tmp_path):
    import wave
    p = tmp_path / "s.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(44100); w.writeframes(b"\0" * 400)
    with pytest.raises(ValueError):
        audio.read_wav(p)


# ---------- transcript
def test_multi_word_phrases_match():
    cats = [c for _, c in text.tag_words(W("okay no way oh my god let's go"))]
    assert cats.count("surprise") == 2 and "hype" in cats


def test_common_words_alone_do_not_fire():
    assert text.tag_words(W("what do you want to run for bro wait here")) == []


def test_repetition_and_laugh_tokens():
    cats = [c for _, c in text.tag_words(W("no no no hahahaha"))]
    assert "repeat" in cats and "laugh" in cats


def test_sentence_edges():
    w = W("we go now. | then we drive")
    assert text.sentence_starts(w) == [0, 3]
    assert text.sentence_ends(w) == [2, 5]


# ---------- chat
def test_load_all_chat_formats(tmp_path):
    yt = tmp_path / "chat.live_chat.json"
    item = {"replayChatItemAction": {"videoOffsetTimeMsec": "12500", "actions": [{"addChatItemAction": {"item": {
        "liveChatTextMessageRenderer": {"authorName": {"simpleText": "ann"}, "message": {"runs": [{"text": "KEKW"}]}}}}}]}}
    yt.write_text(json.dumps(item) + "\n", encoding="utf-8")
    assert chat.load(yt) == [(12.5, "ann", "KEKW")]
    td = tmp_path / "chat.json"
    synth.write_twitchdownloader(td, [(3.0, "bob", "LUL")])
    assert chat.load(td) == [(3.0, "bob", "LUL")]
    jl = tmp_path / "chat.twitch.jsonl"
    jl.write_text(json.dumps({"t": 1_700_000_010.0, "user": "c", "text": "W"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        chat.load(jl)
    assert chat.load(jl, wall_offset=1_700_000_000.0) == [(10.0, "c", "W")]
    cs = tmp_path / "chat.csv"
    cs.write_text("t,user,text\n4,dan,POG\n", encoding="utf-8")
    assert chat.load(cs) == [(4.0, "dan", "POG")]


def test_chat_burst_counts_users_not_spam_and_is_lag_shifted():
    base = [(float(t), f"u{t % 50}", "hi") for t in range(0, 300)]
    crowd = [(200 + i * 0.1, f"fan{i}", "KEKW") for i in range(50)]
    spam = [(100 + i * 0.05, "bot", "FOLLOW FOR FOLLOW") for i in range(100)]
    ads = [(150 + i * 0.1, f"x{i}", "join discord.gg/abc") for i in range(50)]
    f = chat.features(base + crowd + spam + ads, 300, lag=5)
    assert 193 <= int(np.argmax(f["burst"])) <= 200      # crowd at 200-205, shifted 5 s earlier
    assert f["burst"][95:105].max() < 0.5 * f["burst"].max()  # one spammer is one user
    assert f["burst"][140:160].max() < 0.5 * f["burst"].max()  # link/ad messages are dropped
    assert f["laugh"][195:200].max() > 0.5


# ---------- fusion
def test_missing_modalities_get_zero_weight():
    rng = np.random.default_rng(2)
    f = audio.features(x=synth.bed(60, rng).astype(np.float32))
    st = candidates.run(f)
    assert st.weights["chat"] == 0 and st.weights["text"] == 0 and st.weights["heat"] == 0
    assert st.weights["loud"] > 0


def test_stage_a_finds_planted_event():
    x, words, msgs, _, labels = synth.episode(5, dur=180, n_events=2, n_decoys=1)
    f = audio.features(x=x.astype(np.float32))
    st = candidates.run(f, text.features(words, len(f["db"])), chat.features(msgs, len(f["db"])), shortlist=4)
    for lab in labels:
        assert any(lab["start"] - 3 <= p <= lab["end"] + 3 for p in st.shortlist[:3]), (lab, st.shortlist)
