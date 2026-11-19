"""Stage B scoring: seven 0..1 dimensions, a confidence value and human-readable evidence.

These are transparent heuristics over measured signals. They rank moments *within one video*; they do
not predict views or virality, and the output says so.

  entertainment            overall pick score used for ranking
  humour                   laughter (audio), laugh words, laughing chat
  surprise                 sharp onsets, surprise words, "WHAT??" chat, cuts/flashes, gameplay motion
  emotional_intensity      loudness jump, shouting timbre, fast speech, facecam movement, hype words
  contextual_clarity       setup included, clean sentence start/end, no dangling opener
  standalone_watchability  length fit, little dead air, event fully inside the clip
  hook_potential           how early the clip gets going; opening line is a question/exclamation
"""
import numpy as np
from .text import tag_words
from .util import sat, window_max, window_mean

DIMENSIONS = ("entertainment", "humour", "surprise", "emotional_intensity", "contextual_clarity",
              "standalone_watchability", "hook_potential")
DISCLAIMER = ("Scores are heuristic estimates from audio, transcript, chat and sampled frames. They rank moments "
              "within this video; they do not predict views or guarantee virality.")


def _fmt(t):
    return f"{int(t // 60)}:{int(t % 60):02d}"


def score(stage, feats, b, words, vis, peak_score, top_peak, ideal=(15, 35)):
    """feats: raw per-second dicts {'audio','text','chat','heat'}; b: Bounds; vis: visual.analyse() or None."""
    ch = stage.channels
    s, e = b.ev_start - 2, b.ev_end + 2
    au, tx, ct = feats["audio"], feats.get("text"), feats.get("chat")
    m = lambda k: window_max(ch[k], s, e) if k in ch and stage.weights.get(k, 0) > 0 else 0.0
    ev = []
    # ---------- raw evidence
    laugh_a = window_max(au["laugh"], s, e)
    lex = {k: window_max(tx[k], s, e) for k in ("surprise", "hype", "laugh", "tension", "repeat")} if tx and tx.get("has_words") else {}
    chat_on = "chat" in ch and stage.weights.get("chat", 0) > 0
    burst = window_max(ct["burst"], s, e) if chat_on else 0.0
    cl = {k: window_max(ct[k], s, e) for k in ("laugh", "hype", "surprise")} if chat_on else {}
    gz = vis.get("game_motion_z", 0.0) if vis else 0.0
    cz = (vis.get("cam_motion_z") or 0.0) if vis else 0.0
    cuts = vis.get("cuts", 0) if vis else 0
    flash = vis.get("flash", 0.0) if vis else 0.0

    if laugh_a > 0.35: ev.append(f"laughter-like audio ({laugh_a:.2f}) near {_fmt(b.ev_start)}")
    if m("loud") > 1.5: ev.append(f"loudness jump {m('loud'):.1f}σ above the local level")
    if m("bright") > 1.5: ev.append(f"shouting/excited voice timbre {m('bright'):.1f}σ")
    if m("onset") > 2: ev.append(f"sudden sound onset {m('onset'):.1f}σ")
    for k, v in lex.items():
        if v >= 1: ev.append(f"{k} words in transcript")
    if chat_on and burst > 0.4:
        top = max(cl, key=cl.get) if cl else ""
        ev.append(f"chat {np.exp(burst):.1f}x busier than usual (lag-corrected)" + (f", mostly {top}" if cl.get(top, 0) > 0.25 else ""))
    if m("heat") > 1: ev.append("YouTube most-replayed peak")
    if gz > 3: ev.append(f"gameplay motion {gz:.1f}σ above this stream's norm")
    if cz > 3: ev.append(f"facecam movement {cz:.1f}σ (streamer reacting)")
    if cuts: ev.append(f"{cuts} hard visual change(s)")
    if flash > 40: ev.append("bright flash on screen")

    # ---------- dimensions
    humour = sat(1.6 * laugh_a + 0.5 * lex.get("laugh", 0) + 2.5 * cl.get("laugh", 0) * max(burst, 0))
    surprise = sat(0.15 * m("onset") + 0.35 * lex.get("surprise", 0) + 2.0 * cl.get("surprise", 0) * max(burst, 0)
                   + 0.08 * max(gz, 0) + 0.25 * min(cuts, 3) + (0.4 if flash > 40 else 0))
    intensity = sat(0.2 * m("loud") + 0.25 * m("bright") + 0.1 * m("rate") + 0.07 * max(cz, 0)
                    + 0.3 * lex.get("hype", 0) + 0.25 * lex.get("tension", 0) + 0.2 * lex.get("repeat", 0))
    setup_words = sum(1 for w in words if b.start <= w["s"] < b.ev_start)
    if words:
        clarity = 0.35 + 0.35 * min(1.0, setup_words / 8) + (0.0 if b.mid_sentence_start else 0.15) \
            + (0.0 if b.mid_sentence_end else 0.1) - (0.1 if b.dangling_start else 0.0)
    else:
        clarity = 0.35  # cannot judge context without a transcript
    clarity = float(np.clip(clarity, 0, 1))
    L = b.end - b.start
    fit = 1.0 if ideal[0] <= L <= ideal[1] else max(0.0, 1 - min(abs(L - ideal[0]), abs(L - ideal[1])) / 15)
    dead = 1.0 - window_mean(au["active"], b.start, b.end)
    contained = 1.0 if (b.start <= b.ev_start and b.end >= b.ev_end) else 0.6
    watch = float(np.clip(0.35 * fit + 0.25 * (1 - min(1.0, dead * 1.5)) + 0.25 * clarity + 0.15 * contained, 0, 1))
    first = (vis or {}).get("event_offset")
    lead = min(b.ev_start - b.start, first) if first is not None else b.ev_start - b.start
    hook = float(np.clip(1.0 - max(0.0, lead - 2.0) / 12.0, 0.15, 1.0))
    opening = [w for w in words if b.start <= w["s"] < b.start + 3]
    if opening and (any(w["w"].rstrip()[-1:] in "!?" for w in opening) or tag_words(opening)):
        hook = min(1.0, hook + 0.15); ev.append("opening line is a question/exclamation")
    crowd = sat(0.8 * max(burst, 0) + 0.3 * m("heat"))
    # half relative to this video's best moment, half absolute, so a flat video's "best" is not a 1.0
    strength = 0.5 * float(np.clip(peak_score / (top_peak + 1e-9), 0, 1)) + 0.5 * sat(peak_score / 3.0)
    peak_dims = max(humour, surprise, intensity)
    # ---------- reaction gate: loud gameplay nobody reacts to is the most common false positive
    reaction = window_max(feats["reaction"], s, e) if "reaction" in feats else None
    can_judge = bool(words) or chat_on or (vis is not None and vis.get("cam_motion_z") is not None)
    reacted = (reaction or 0) >= 0.3 or cz > 3 or m("heat") > 1
    gate = 1.0
    if reaction is not None and can_judge and not reacted:
        gate = 0.6
        ev.append("no reaction from streamer, chat or facecam (loud game audio only?) - ranked down")
    loud_secs = int(np.sum(au["jump"][max(0, int(s)):int(e) + 1] > 6))
    if loud_secs <= 1 and not reacted and m("onset") > 0:
        gate *= 0.8
        ev.append("single short sound spike (mic bump or click?)")
    ent = (0.25 * peak_dims + 0.15 * (humour + surprise + intensity) / 3 + 0.2 * strength + 0.15 * crowd
           + 0.15 * watch + 0.10 * hook) * gate
    # ---------- confidence: how many independent modalities agree, and how many we could see
    avail = ["audio"] + (["text"] if words else []) + (["chat"] if chat_on else []) \
        + (["visual"] if vis and vis.get("frames", 0) > 1 else []) + (["heat"] if stage.weights.get("heat", 0) > 0 else [])
    agree = set()
    if laugh_a > 0.35 or m("loud") > 1.5 or m("bright") > 1.5 or m("onset") > 2: agree.add("audio")
    if any(v >= 1 for v in lex.values()): agree.add("text")
    if chat_on and burst > 0.4: agree.add("chat")
    if gz > 3 or cz > 3 or cuts or flash > 40: agree.add("visual")
    if m("heat") > 1: agree.add("heat")
    conf = 0.25 + 0.45 * (len(agree) / max(1, len(avail))) + 0.06 * min(len(avail), 5)
    if not words: conf -= 0.1
    conf = float(np.clip(conf, 0.05, 0.95))
    dims = {"entertainment": ent, "humour": humour, "surprise": surprise, "emotional_intensity": intensity,
            "contextual_clarity": clarity, "standalone_watchability": watch, "hook_potential": hook}
    return {k: round(float(v), 3) for k, v in dims.items()}, round(conf, 2), ev, {"modalities_available": avail,
                                                                                     "modalities_agreeing": sorted(agree),
                                                                                     "reaction": None if reaction is None else round(reaction, 2),
                                                                                     "gate": round(gate, 2)}
