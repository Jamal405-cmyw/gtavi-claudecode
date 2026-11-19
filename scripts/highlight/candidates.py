"""Stage A: fuse cheap per-second signals into one score and pick a shortlist of peaks.

Every evidence channel is a non-negative series in roughly 0..4 ("how unusual is this second").
Missing modalities (no chat, no heatmap, no transcript) get weight 0 instead of fake values.
"""
from dataclasses import dataclass, field
import numpy as np
from .util import peaks, pos, robust_z, smooth

DEFAULT_WEIGHTS = {"loud": 1.0, "onset": 0.5, "bright": 0.6, "laugh": 1.2, "text": 1.0, "rate": 0.3,
                   "chat": 1.6, "heat": 1.5}


@dataclass
class StageA:
    n: int
    channels: dict            # name -> per-second evidence (np.ndarray)
    weights: dict             # name -> weight actually used (0 if unavailable)
    score: np.ndarray
    shortlist: list = field(default_factory=list)   # peak seconds, best first


def evidence(audio, text=None, chat=None, heat=None):
    """Per-second evidence channels from the raw feature dicts."""
    n = len(audio["db"])
    act = audio["active"]
    ch = {
        "loud": smooth(pos(robust_z(audio["jump"])), 3),
        "onset": smooth(pos(robust_z(audio["onset"])), 2),
        # timbre only counts when the second is actually loud (quiet hiss is often "bright")
        "bright": smooth(pos(robust_z(audio["bright"])) * (act > 0.5) * (audio["jump"] > 0), 3),
        "laugh": smooth(audio["laugh"] * 3.0, 3),
    }
    if text is not None and text.get("has_words"):
        lex = (1.0 * text["surprise"] + 0.8 * text["hype"] + 1.2 * text["laugh"] + 0.8 * text["tension"]
               + 0.8 * text["repeat"] + 0.3 * text["punct"])
        ch["text"] = np.clip(lex, 0, 4)
        ch["rate"] = smooth(pos(robust_z(text["rate"]), 3), 3)
    if chat is not None and chat.get("n_msgs", 0) > 0:
        boost = 1 + chat["laugh"] + chat["hype"] + chat["surprise"]  # reaction-heavy chat counts more
        ch["chat"] = np.clip(pos(chat["burst"]) * 2.0 * boost, 0, 4)
    if heat is not None and np.any(heat):
        ch["heat"] = smooth(pos(robust_z(heat)), 3)
    for k in list(ch):
        ch[k] = np.asarray(ch[k], dtype=float)[:n]
    return ch


def run(audio, text=None, chat=None, heat=None, weights=None, shortlist=15, min_sep=20):
    w = dict(DEFAULT_WEIGHTS, **(weights or {}))
    ch = evidence(audio, text, chat, heat)
    n = len(audio["db"])
    used = {k: (w.get(k, 0.0) if k in ch else 0.0) for k in DEFAULT_WEIGHTS}
    score = np.zeros(n)
    for k, v in ch.items():
        score += used[k] * v
    score = smooth(score, 3)
    return StageA(n=n, channels=ch, weights=used, score=score, shortlist=peaks(score, shortlist, min_sep))


def firing(stage, t, radius=6, thresh=1.0):
    """Channels that clearly fire within +-radius s of t (their max >= thresh)."""
    s, e = max(0, int(t - radius)), min(stage.n, int(t + radius) + 1)
    return sorted(k for k, v in stage.channels.items() if stage.weights.get(k, 0) > 0 and e > s and v[s:e].max() >= thresh)
