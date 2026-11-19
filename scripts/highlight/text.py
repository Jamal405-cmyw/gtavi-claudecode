"""Stage A transcript features from word timestamps ({"w","s","e"} as written by moments.transcribe).

Phrases are matched on the normalised token *sequence*, so multi-word cues ("no way", "oh my god")
work - the original detector matched single words and could never see them (docs/AUDIT.md P0 #1).
Common words ("what", "run", "bro") only count inside phrases or repetitions, not on their own.
"""
import re
import numpy as np
from .util import smooth

_TOK = re.compile(r"[a-z0-9']+")

PHRASES = {
    "surprise": ["no way", "what the", "oh my god", "oh my gosh", "are you serious", "are you kidding",
                 "you're kidding", "how did", "what just happened", "did you see that", "did you see",
                 "wait what", "holy", "omg", "wtf", "bruh", "i can't believe", "excuse me", "hold on",
                 "what was that", "oh shit", "oh my", "yo what"],
    "hype": ["let's go", "lets go", "go go", "yes yes", "insane", "crazy", "clutch", "unbelievable",
             "massive", "sheesh", "huge", "w", "gg", "let's goo", "lets goo", "goated", "beautiful"],
    "laugh": ["haha", "hahaha", "hahahaha", "ha ha", "lol", "lmao", "lmfao", "laughs", "laughter", "laughing",
              "i'm dead", "im dead", "i'm crying", "that's hilarious", "so funny", "hehe", "dying"],
    "tension": ["oh no", "run run", "get out", "help me", "no no", "go go go", "behind you", "watch out",
                "he's coming", "they're coming", "i'm dying", "we're dead", "get in", "drive drive"],
}
PROFANE = {"shit", "fuck", "fucking", "damn", "hell", "bloody", "crap"}
DANGLING = {"and", "but", "so", "because", "cause", "it", "that", "this", "he", "she", "they", "them",
            "which", "then", "also", "or", "like"}


def norm(word):
    m = _TOK.findall(word.lower().replace("’", "'"))
    return m[0] if m else ""


def _phrase_index():
    idx = {}
    for cat, plist in PHRASES.items():
        for p in plist:
            toks = tuple(p.split())
            idx.setdefault(toks[0], []).append((toks, cat))
    for v in idx.values():
        v.sort(key=lambda t: -len(t[0]))
    return idx


_IDX = _phrase_index()


def tag_words(words):
    """[(t_start, category)] for every phrase hit, repetition and profanity."""
    toks = [norm(w.get("w", "")) for w in words]
    hits = []
    i = 0
    while i < len(toks):
        t = toks[i]
        matched = 0
        for ptoks, cat in _IDX.get(t, ()):
            if tuple(toks[i:i + len(ptoks)]) == ptoks:
                hits.append((words[i]["s"], cat)); matched = len(ptoks); break
        if not matched and t.startswith("haha"):
            hits.append((words[i]["s"], "laugh")); matched = 1
        if not matched and t in PROFANE:
            hits.append((words[i]["s"], "tension"))
        # same token 3+ times in a row ("no no no", "go go go") = excitement
        if t and i + 2 < len(toks) and toks[i + 1] == t and toks[i + 2] == t:
            hits.append((words[i]["s"], "repeat"))
        i += max(1, matched)
    return hits


def features(words, n):
    """Per-second arrays: surprise, hype, laugh, tension, repeat, rate (words/s), punct (!/?), has_words."""
    out = {k: np.zeros(n) for k in ("surprise", "hype", "laugh", "tension", "repeat", "rate", "punct")}
    for t, cat in tag_words(words):
        if 0 <= int(t) < n:
            out[cat][int(t)] += 1
    for w in words:
        s = int(w.get("s", -1))
        if 0 <= s < n:
            out["rate"][s] += 1
            if w.get("w", "").rstrip()[-1:] in "!?":
                out["punct"][s] += 1
    for k in ("surprise", "hype", "laugh", "tension", "repeat", "punct"):
        out[k] = smooth(out[k], 4) * 4  # count per 4 s, centred
    out["rate"] = smooth(out["rate"], 3)
    out["has_words"] = len(words) > 0
    return out


def sentence_starts(words, gap=0.6):
    """Indices of words that start a sentence (first word, after a pause, or after . ? !)."""
    st = []
    for i, w in enumerate(words):
        if i == 0 or w["s"] - words[i - 1]["e"] >= gap or words[i - 1]["w"].rstrip()[-1:] in ".?!":
            st.append(i)
    return st


def sentence_ends(words, gap=0.6):
    en = []
    for i, w in enumerate(words):
        last = i == len(words) - 1
        if last or words[i + 1]["s"] - w["e"] >= gap or w["w"].rstrip()[-1:] in ".?!":
            en.append(i)
    return en


def text_between(words, s, e):
    return " ".join(w["w"].strip() for w in words if s <= w["s"] < e).strip()
