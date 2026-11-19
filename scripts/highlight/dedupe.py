"""Duplicate detection between highlight candidates (same video or across jobs).

Two candidates are duplicates when
  - they overlap in time (overlap > 50 % of the shorter one), or
  - their transcripts are near-identical (word 3-gram Jaccard >= 0.6, both >= 12 words), or
  - their frames match (>= 2 of 3 dHashes within 10 bits) and transcripts do not contradict it
    (Jaccard >= 0.2, or one side has no speech).
The higher-ranked candidate is kept; the other records `duplicate_of`.
"""
import re
from .visual import hamming

_T = re.compile(r"[a-z0-9']+")


def grams(text, n=3):
    t = _T.findall((text or "").lower())
    return {tuple(t[i:i + n]) for i in range(max(0, len(t) - n + 1))}, len(t)


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def overlap_frac(a, b):
    ov = min(a["end"], b["end"]) - max(a["start"], b["start"])
    short = min(a["end"] - a["start"], b["end"] - b["start"])
    return max(0.0, ov) / short if short > 0 else 0.0


def same(a, b, same_video=True):
    """(is_duplicate, reason)."""
    if same_video and overlap_frac(a, b) > 0.5:
        return True, "overlaps in time"
    ga, na = grams(a.get("text", ""))
    gb, nb = grams(b.get("text", ""))
    j = jaccard(ga, gb)
    if na >= 12 and nb >= 12 and j >= 0.6:
        return True, f"same speech (similarity {j:.2f})"
    ha, hb = a.get("hashes") or [], b.get("hashes") or []
    if len(ha) == 3 and len(hb) == 3:
        close = sum(1 for x, y in zip(ha, hb) if hamming(x, y) <= 10)
        if close >= 2 and (j >= 0.2 or na < 4 or nb < 4):
            return True, "same footage (frame hashes match)"
    return False, ""


def dedupe(ranked, others=()):
    """ranked: candidates best-first. others: candidates from other jobs. Returns (kept, dropped)."""
    kept, dropped = [], []
    for c in ranked:
        hit = None
        for k in kept:
            dup, why = same(c, k, same_video=True)
            if dup:
                hit = (k.get("id"), why); break
        if not hit:
            for o in others:
                dup, why = same(c, o, same_video=False)
                if dup:
                    hit = (o.get("id"), why + " (other job)"); break
        if hit:
            c["duplicate_of"], c["duplicate_reason"] = hit
            dropped.append(c)
        else:
            kept.append(c)
    return kept, dropped
