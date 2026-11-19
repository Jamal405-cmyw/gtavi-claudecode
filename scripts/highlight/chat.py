"""Chat loading (every format this repo produces or accepts) and per-second chat features.

Formats, auto-detected:
  - YouTube live-chat replay (yt-dlp `*.live_chat.json`, one JSON object per line, videoOffsetTimeMsec)
  - TwitchDownloader chat JSON ({"comments":[{"content_offset_seconds", "commenter", "message"}]})
  - JSON lines {"t": seconds, "user", "text"} - relative offsets, or wall-clock (live.py
    chat.twitch.jsonl) when `wall_offset` (unix time of video second 0) is given
  - CSV with header t,user,text

Features count *unique chatters* per second, not messages, so one spammer or bot cannot fake a spike.
Ads/raids/giveaways/links are dropped (same NOISE filter as live.py). The series is shifted earlier
by `lag` seconds because chat reacts after the moment.
"""
import csv, json, re
from pathlib import Path
import numpy as np
from .util import shift_earlier, smooth

LAUGH = re.compile(r"(?i)\b(LUL|LULW|KEKW|KEK|OMEGALUL|ICANT|LMAO+|LMFAO|LOL+|HAHA\w*|xD+)\b|[😂🤣💀]")
HYPE = re.compile(r"(?i)\b(POG\w*|PogChamp|W+|LETS ?GO+|HYPE|CLIP ?(IT|THAT)?|GG|EZ|INSANE|HOLY)\b|[🔥]")
SURPRISE = re.compile(r"(?i)\b(WHAT+|NO WAY|WTF|monkaS|monkaW|D:|OMG|HUH+|WAIT)\b|\?{2,}|[😱😮]")
NOISE = re.compile(r"(?i)giveaway|\braid(ing|ed)?\b|\bbrb\b|be right back|ad break|sponsor|subscribe|"
                   r"discord\.gg|https?://|promo code|follow me")


def _yt_text(actions):
    out = []
    for a in actions:
        r = (((a.get("addChatItemAction") or {}).get("item")) or {}).get("liveChatTextMessageRenderer")
        if not r:
            continue
        user = ((r.get("authorName") or {}).get("simpleText")) or r.get("authorExternalChannelId") or "?"
        parts = []
        for run in (r.get("message") or {}).get("runs", []):
            if "text" in run:
                parts.append(run["text"])
            elif "emoji" in run:
                e = run["emoji"]
                sc = e.get("shortcuts") or [""]
                parts.append(e.get("emojiId") if len(e.get("emojiId", "")) <= 2 else sc[0])
        out.append((user, "".join(parts).strip()))
    return out


def load(path, wall_offset=None):
    """[(t_seconds, user, text)] sorted by time."""
    p = Path(path)
    raw = p.read_text(encoding="utf-8", errors="ignore")
    msgs = []
    head = raw.lstrip()[:1]
    if head == "{" and '"comments"' in raw[:20000] and not raw.lstrip().startswith('{"replayChatItemAction"'):
        try:
            js = json.loads(raw)
            for m in js.get("comments", []):
                c = m.get("commenter") or {}
                msgs.append((float(m.get("content_offset_seconds", 0)), c.get("display_name") or c.get("_id") or "?",
                             ((m.get("message") or {}).get("body") or "")))
            return sorted(msgs)
        except json.JSONDecodeError:
            pass
    if p.suffix.lower() == ".csv":
        for r in csv.DictReader(raw.splitlines()):
            try:
                msgs.append((float(r["t"]), r.get("user") or "?", r.get("text") or ""))
            except (KeyError, ValueError):
                continue
        return sorted(msgs)
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            o = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if "replayChatItemAction" in o:
            a = o["replayChatItemAction"]
            try:
                t = int(a["videoOffsetTimeMsec"]) / 1000
            except (KeyError, ValueError):
                continue
            msgs += [(t, u, x) for u, x in _yt_text(a.get("actions", []))]
        elif "t" in o:
            t = float(o["t"])
            if t > 1e9:  # wall clock
                if wall_offset is None:
                    raise ValueError(f"{path}: wall-clock chat timestamps need --chat-offset (unix time of video second 0)")
                t -= wall_offset
            msgs.append((t, o.get("user") or "?", o.get("text") or ""))
    return sorted(msgs)


def features(msgs, n, lag=5.0):
    """Per-second arrays (already shifted earlier by lag): users, burst, laugh, hype, surprise."""
    users = [set() for _ in range(n)]
    cat = {k: np.zeros(n) for k in ("laugh", "hype", "surprise")}
    kept = 0
    for t, u, txt in msgs:
        s = int(t)
        if not (0 <= s < n) or not txt or NOISE.search(txt):
            continue
        kept += 1
        users[s].add(u)
        if LAUGH.search(txt): cat["laugh"][s] += 1
        if HYPE.search(txt): cat["hype"][s] += 1
        if SURPRISE.search(txt): cat["surprise"][s] += 1
    cnt = np.array([len(x) for x in users], dtype=float)
    rate = smooth(cnt, 5)
    base = smooth(rate, 180)                      # local baseline: viewer counts drift over a stream
    burst = np.log((rate + 0.2) / (base + 0.2))   # >0 when chat is busier than usual here
    out = {"users": shift_earlier(rate, lag), "burst": shift_earlier(burst, lag), "n_msgs": kept}
    tot = smooth(cnt, 5) + 1e-6
    for k, v in cat.items():
        out[k] = shift_earlier(np.clip(smooth(v, 5) / tot, 0, 1), lag)  # share of messages in category
    return out
