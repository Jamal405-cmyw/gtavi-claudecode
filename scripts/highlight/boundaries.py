"""Context-aware clip boundaries.

1. Event region: grow from the Stage A peak while the score stays above `keep` x peak score.
2. Start: a sentence start 3-15 s before the event (the setup), preferring long pauses, a ~7 s lead,
   and not opening on a dangling word ("and", "so", "it"...). Without words: the quietest second.
3. End: the first sentence end after the reaction has died down (+0.5 s tail), never mid-word.
4. Length forced into [min_len, max_len] by moving to neighbouring sentence edges first.
5. Dead air at the head is trimmed.
"""
from dataclasses import dataclass, asdict
import numpy as np
from .text import DANGLING, norm, sentence_ends, sentence_starts


@dataclass
class Bounds:
    start: float
    end: float
    ev_start: float
    ev_end: float
    start_reason: str
    end_reason: str
    mid_sentence_start: bool
    mid_sentence_end: bool
    dangling_start: bool

    def as_dict(self):
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def event_region(score, peak, keep=0.4, max_side=15):
    n = len(score)
    thr = score[peak] * keep
    a = peak
    while a > 0 and peak - a < max_side and score[a - 1] >= thr:
        a -= 1
    b = peak
    while b < n - 1 and b - peak < max_side and score[b + 1] >= thr:
        b += 1
    return float(a), float(b + 1)


def _quiet_point(db, lo, hi):
    lo, hi = max(0, int(lo)), min(len(db), int(np.ceil(hi)))
    if hi <= lo:
        return float(max(0, lo))
    return float(lo + int(np.argmin(db[lo:hi])))


def compute(peak, score, words, db, duration, min_len=15.0, max_len=40.0, lead=(3.0, 15.0), target_lead=7.0,
            tail=(1.0, 12.0), keep=0.4):
    ev_s, ev_e = event_region(score, peak, keep=keep)
    starts = sentence_starts(words) if words else []
    ends = sentence_ends(words) if words else []
    # ---- start
    best, s_reason = None, ""
    for i in starts:
        t = words[i]["s"]
        d = ev_s - t
        if not (lead[0] <= d <= lead[1]):
            continue
        gap = words[i]["s"] - words[i - 1]["e"] if i > 0 else 2.0
        dangling = norm(words[i]["w"]) in DANGLING
        sc = min(gap, 2.0) - abs(d - target_lead) * 0.15 - (0.8 if dangling else 0.0)
        if best is None or sc > best[0]:
            best = (sc, t, dangling)
    if best:
        start, dangling = max(0.0, best[1] - 0.25), best[2]
        s_reason = "sentence start before the setup"
    else:
        start = _quiet_point(db, ev_s - lead[1], ev_s - lead[0] + 1) if len(db) else max(0.0, ev_s - target_lead)
        dangling = False
        s_reason = "quietest second before the event (no sentence start in range)"
    # ---- end
    end, e_reason = None, ""
    for i in ends:
        t = words[i]["e"]
        if ev_e + tail[0] <= t + 0.5 <= ev_e + tail[1]:
            end, e_reason = t + 0.5, "first sentence end after the reaction"
            break
    if end is None:
        end = _quiet_point(db, ev_e + tail[0], ev_e + 4) + 0.5 if len(db) else ev_e + 2.0
        e_reason = "quiet point after the reaction"
    end = min(end, duration)
    # ---- length limits
    if end - start < min_len:
        later = [words[i]["e"] + 0.5 for i in ends if words[i]["e"] + 0.5 > end and words[i]["e"] + 0.5 - start <= max_len]
        if later and later[-1] - start >= min_len:
            end = next(t for t in later if t - start >= min_len); e_reason += "; extended to a later sentence end"
        else:
            earlier = [words[i]["s"] - 0.25 for i in starts if words[i]["s"] - 0.25 < start and end - (words[i]["s"] - 0.25) <= max_len]
            cand = [t for t in earlier if end - t >= min_len]
            if cand:
                start = max(cand); s_reason += "; moved back to an earlier sentence start"
            else:
                pad = min_len - (end - start)
                start = max(0.0, start - pad * 0.6); end = min(duration, end + pad * 0.4)
                if end - start < min_len:
                    start = max(0.0, end - min_len)
                    end = min(duration, max(end, start + min_len))
                s_reason += "; padded to the minimum length"
    if end - start > max_len:
        later_starts = [words[i]["s"] - 0.25 for i in starts if start < words[i]["s"] - 0.25 <= ev_s - lead[0] and end - (words[i]["s"] - 0.25) <= max_len]
        if later_starts:
            start = min(later_starts); s_reason += "; shortened to a later sentence start"
        if end - start > max_len:
            end = start + max_len; e_reason += "; cut at the maximum length"
    # ---- trim dead air at the head (no word and quiet in the first 1.5 s)
    if words:
        first = next((w for w in words if w["s"] >= start), None)
        if first and first["s"] - start > 1.5 and first["s"] - 0.25 < ev_s and end - (first["s"] - 0.25) >= min_len:
            start = first["s"] - 0.25; s_reason += "; trimmed dead air"
    start, end = max(0.0, start), min(duration, end)
    first_i = next((i for i, w in enumerate(words) if w["s"] >= start), None)
    sset, eset = set(starts), set(ends)
    mid_start = any(w["s"] < start < w["e"] for w in words) or (first_i is not None and first_i not in sset)
    last_i = max((i for i, w in enumerate(words) if w["e"] <= end), default=None)
    mid_end = any(w["s"] < end - 0.05 < w["e"] for w in words) or (
        last_i is not None and last_i not in eset and words[last_i + 1]["s"] < end + 0.6)
    return Bounds(round(start, 2), round(end, 2), ev_s, ev_e, s_reason, e_reason, bool(mid_start), bool(mid_end), bool(dangling))
