"""Conversation analysis on hand-written transcripts (not the synthetic generator)."""
from conftest import W
from highlight import dialogue


def test_setup_payoff_with_verbatim_summary():
    words = W("okay chat watch this. I am going to jump the bridge with the bike", 10) + W("no way hahaha", 18)
    d = dialogue.analyse(words, 9.5, 25, 18, 20)
    assert d["setup_payoff"] and "stunt" in d["game_events"] and "vehicle" in d["game_events"]
    assert d["summary"]["reaction"] == "no way hahaha"
    assert "jump the bridge" in d["summary"]["setup"]
    clip_text = " ".join(w["w"] for w in words)
    assert all(tok in clip_text.split() for tok in d["summary"]["setup"].split())  # verbatim, nothing invented


def test_plain_talk_is_not_a_payoff():
    words = W("so we drive to the shop and buy some paint for the car and then go home", 10)
    d = dialogue.analyse(words, 9.5, 20, 15, 17)
    assert not d["setup_payoff"] and not d["question_reaction"] and d["argument"] < 0.5


def test_question_then_reaction_and_stunned_silence():
    words = W("did he just drive into the lake?", 30) + W("oh my god", 35)  # 2.6 s silence before reaction
    d = dialogue.analyse(words, 29, 45, 35, 37)
    assert d["question_reaction"] and d["stunned_silence"]
    assert any("stunned" in e for e in dialogue.evidence(d))


def test_argument_markers():
    words = W("no you said left. that's not my fault. why would you do that. shut up you crashed it", 5, 0.3)
    d = dialogue.analyse(words, 4, 20, 10, 12)
    assert d["argument"] >= 0.5


def test_empty_clip():
    d = dialogue.analyse([], 0, 10, 3, 5)
    assert d["summary"] is None and d["words"] == 0


def test_summary_splits_at_first_reaction_word_without_overlap():
    words = W("wait watch this I am going to jump the bridge", 20) + W("hahaha no way", 24.5)
    d = dialogue.analyse(words, 19, 35, 24, 27)   # event region starts before the laugh
    assert d["summary"]["setup"].endswith("jump the bridge") and d["summary"]["reaction"] == "hahaha no way"
