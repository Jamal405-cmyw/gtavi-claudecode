"""Stage B conversation analysis of one shortlisted clip (transcript words only, no model needed).

Looks at the *structure* of what is said around the event, not just hype words:
  setup_payoff      speech that sets something up, then a reaction at the event
  question_reaction a question in the setup answered by an exclamation/laugh at the event
  stunned_silence   >= 1.5 s of silence right before the outburst
  argument          disagreement/banter markers, lots of "you", rapid short sentences
  game_events       GTA gameplay named in speech (cops, wanted level, crash, heist...)
  summary           extractive: the setup line and the reaction line, quoted verbatim

Everything is derived from the transcript; nothing is generated. The summary is for the human
reviewer and the hook writer; it must not be presented as a quote from someone other than the streamer.
"""
import re
from .text import norm, sentence_ends, sentence_starts, tag_words

DISAGREE = ["no you", "that's not", "thats not", "you're wrong", "youre wrong", "shut up", "i told you",
            "why would you", "what are you doing", "stop it", "are you stupid", "you said", "not my fault",
            "your fault", "whatever", "listen", "bro what", "no no no", "that's cap", "you're lying"]
QUESTION_START = {"what", "how", "why", "did", "is", "are", "can", "could", "would", "should", "do", "does",
                  "where", "who", "was", "were", "will", "you"}
GAME = {"police": "police", "cops": "police", "cop": "police", "wanted": "wanted level", "stars": "wanted level",
        "chase": "chase", "crash": "crash", "crashed": "crash", "explode": "explosion", "exploded": "explosion",
        "explosion": "explosion", "boom": "explosion", "wasted": "death", "dead": "death", "died": "death",
        "busted": "arrest", "arrested": "arrest", "heist": "heist", "bank": "heist", "robbery": "heist",
        "rob": "heist", "jump": "stunt", "flip": "stunt", "stunt": "stunt", "plane": "vehicle", "helicopter": "vehicle",
        "heli": "vehicle", "tank": "vehicle", "boat": "vehicle", "bike": "vehicle", "motorcycle": "vehicle",
        "glitch": "glitch", "bug": "glitch", "gator": "wildlife", "alligator": "wildlife", "shark": "wildlife",
        "lucia": "character", "jason": "character", "vice": "location", "leonida": "location"}
_PHRASES = [tuple(p.split()) for p in DISAGREE]


def _phrase_hits(toks, phrases):
    hits = 0
    for i in range(len(toks)):
        for p in phrases:
            if tuple(toks[i:i + len(p)]) == p:
                hits += 1
    return hits


def _line(words):
    return " ".join(w["w"].strip() for w in words).strip()


def analyse(words, start, end, ev_start, ev_end):
    clip = [w for w in words if start <= w["s"] < end]
    setup = [w for w in clip if w["s"] < ev_start]
    event = [w for w in clip if ev_start - 1 <= w["s"] < ev_end + 1]
    out = {"setup_payoff": False, "question_reaction": False, "stunned_silence": False, "argument": 0.0,
           "game_events": [], "summary": None, "words": len(clip)}
    if not clip:
        return out
    react_cats = {c for _, c in tag_words(event)}
    reacted = bool(react_cats & {"surprise", "hype", "laugh", "tension", "repeat"})
    out["setup_payoff"] = len(setup) >= 4 and reacted
    # question in the last sentences of the setup
    q = False
    starts = set(sentence_starts(clip))
    for i, w in enumerate(clip):
        if w["s"] >= ev_start:
            break
        if w["w"].rstrip().endswith("?") or (i in starts and norm(w["w"]) in QUESTION_START and ev_start - w["s"] < 10):
            q = True
    out["question_reaction"] = q and reacted
    # silence right before the event's first word
    first_ev = next((i for i, w in enumerate(clip) if w["s"] >= ev_start - 0.5), None)
    if first_ev is not None and first_ev > 0:
        out["stunned_silence"] = clip[first_ev]["s"] - clip[first_ev - 1]["e"] >= 1.5 and reacted
    toks = [norm(w["w"]) for w in clip]
    dur = max(1.0, clip[-1]["e"] - clip[0]["s"])
    you = sum(t in ("you", "you're", "youre", "your") for t in toks) / max(1, len(toks))
    short_sent = len(starts) / dur * 10  # sentences per 10 s
    arg = 0.5 * min(2, _phrase_hits(toks, _PHRASES)) + (0.3 if you > 0.08 else 0) + (0.2 if short_sent > 3 else 0)
    out["argument"] = round(min(1.0, arg), 2)
    seen = []
    for t in toks:
        g = GAME.get(t)
        if g and g not in seen:
            seen.append(g)
    out["game_events"] = seen
    # extractive summary, split at the first reaction word: setup = the words before it (from the last
    # sentence start, at least ~4 words), reaction = that word onward. Verbatim, no overlap.
    hits = sorted(t for t, c in tag_words(event) if c in ("surprise", "hype", "laugh", "tension", "repeat"))
    cue = hits[0] if hits else ev_start
    k = next((i for i, w in enumerate(clip) if w["s"] >= cue - 0.01), len(clip))
    st_idx = [i for i in sorted(starts) if i < k]
    a = st_idx[-1] if st_idx else max(0, k - 10)
    if k - a < 4:
        a = max(0, k - 10)
    ends = set(sentence_ends(clip))
    z = next((i + 1 for i in range(k, min(len(clip), k + 12)) if i in ends), min(len(clip), k + 12))
    s_line, r_line = _line(clip[a:k]), _line(clip[k:z])  # reaction stops at its sentence end
    if s_line or r_line:
        out["summary"] = {"setup": s_line, "reaction": r_line}
    return out


def evidence(d):
    ev = []
    if d["setup_payoff"]: ev.append("setup line followed by a reaction (setup → payoff)")
    if d["question_reaction"]: ev.append("a question in the setup gets an exclamation/laugh")
    if d["stunned_silence"]: ev.append("stunned silence right before the outburst")
    if d["argument"] >= 0.5: ev.append("argument/banter in the dialogue")
    if d["game_events"]: ev.append("gameplay mentioned: " + ", ".join(d["game_events"][:4]))
    return ev
