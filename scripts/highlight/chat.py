"""Chat loading (every format this repo produces or accepts) and per-second chat features.

Formats, auto-detected (plain or gzip-compressed):
  - YouTube live-chat replay (yt-dlp `*.live_chat.json`, one JSON object per line, videoOffsetTimeMsec)
  - TwitchDownloader chat JSON ({"FileInfo", "video": {"start", "end", "length"}, "comments": [...]}),
    including `--compression Gzip` output (`chat.json.gz`)
  - TwitchDownloader TXT export: `[h:mm:ss] user: message` (relative) or `[YYYY-MM-DD HH:MM:SS UTC]`
  - JSON lines {"t": seconds, "user", "text"} - relative offsets, or wall-clock (live.py
    chat.twitch.jsonl) when `wall_offset` (unix time of video second 0) is given
  - CSV with header t,user,text

TwitchDownloader writes `content_offset_seconds` relative to the start of the *VOD*, also when the
chat was trimmed (-b/-e) or belongs to a clip; the window is stored in video.start/video.end. If your
video file is that trimmed range or the clip, offsets must be shifted by -video.start ("trimmed").
`align="auto"` decides from the media duration; see align_offset().

Features count *unique chatters* per second, not messages, so one spammer or bot cannot fake a spike.
Ads/raids/giveaways/links are dropped (same NOISE filter as live.py). The series is shifted earlier
by `lag` seconds because chat reacts after the moment.
"""
import csv, gzip, io, json, re
from datetime import datetime, timezone
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


TXT_REL = re.compile(r"^\[(\d+):(\d{2}):(\d{2})\]\s+([^:]+?):\s?(.*)$")
TXT_UTC = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?) UTC\]\s+([^:]+?):\s?(.*)$")


def _read_text(p):
    raw = Path(p).read_bytes()
    if raw[:2] == b"\x1f\x8b":  # gzip (TwitchDownloader --compression Gzip)
        raw = gzip.decompress(raw)
    return raw.decode("utf-8-sig", errors="ignore")


def _td_text(m):
    msg = m.get("message") or {}
    body = msg.get("body")
    if not body:
        body = "".join((f.get("text") or "") for f in (msg.get("fragments") or []))
    return body or ""


def align_offset(video, media_duration=None, align="auto"):
    """Seconds to ADD to TwitchDownloader offsets so they match the media file.
    vod: the file starts at VOD second 0 (normal yt-dlp/TwitchDownloader VOD download) -> 0
    trimmed: the file is exactly the downloaded window/clip -> -video.start
    auto: trimmed when video.start > 0 and the file length matches end-start (and not the full VOD)."""
    if isinstance(align, (int, float)):
        return float(align)
    start = float((video or {}).get("start") or 0.0)
    if align == "vod" or start <= 0:
        return 0.0
    if align == "trimmed":
        return -start
    end = float((video or {}).get("end") or 0.0)
    length = float((video or {}).get("length") or -1)
    if media_duration and end > start:
        window = end - start
        tol = max(5.0, 0.03 * window)
        if abs(media_duration - window) <= tol and not (length > 0 and abs(media_duration - length) <= tol):
            return -start
    return 0.0


def load_with_info(path, wall_offset=None, media_duration=None, align="auto"):
    """([(t_seconds, user, text)] sorted by time, info dict with format/shift/version)."""
    p = Path(path)
    text = _read_text(p)
    info = {"format": None, "shift": 0.0}
    msgs = []
    stripped = text.lstrip()
    if stripped[:1] == "{":
        try:
            js = json.loads(stripped)
        except json.JSONDecodeError:
            js = None  # JSON lines (YouTube replay / jsonl) - handled below
        if isinstance(js, dict) and isinstance(js.get("comments"), list):
            video = js.get("video") or {}
            shift = align_offset(video, media_duration, align)
            fi = (js.get("FileInfo") or {}).get("Version") or {}
            info.update(format="twitchdownloader-json", shift=shift, video_start=video.get("start"),
                        video_end=video.get("end"), version=".".join(str(fi.get(k, 0)) for k in ("Major", "Minor", "Patch")) if fi else None)
            for m in js["comments"]:
                c = m.get("commenter") or {}
                try:
                    t = float(m.get("content_offset_seconds", 0)) + shift
                except (TypeError, ValueError):
                    continue
                msgs.append((t, c.get("display_name") or c.get("name") or c.get("_id") or "?", _td_text(m)))
            return sorted(msgs), info
    if p.suffix.lower() == ".csv" or p.name.lower().endswith(".csv.gz"):
        info["format"] = "csv"
        for r in csv.DictReader(io.StringIO(text)):
            try:
                msgs.append((float(r["t"]), r.get("user") or "?", r.get("text") or ""))
            except (KeyError, ValueError, TypeError):
                continue
        return sorted(msgs), info
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        m = TXT_REL.match(ln)
        if m:
            info["format"] = "twitchdownloader-txt"
            h, mi, se, user, body = m.groups()
            msgs.append((int(h) * 3600 + int(mi) * 60 + int(se), user.strip(), body))
            continue
        m = TXT_UTC.match(ln)
        if m:
            info["format"] = "twitchdownloader-txt-utc"
            if wall_offset is None:
                raise ValueError(f"{path}: UTC chat timestamps need --chat-offset (unix time of video second 0)")
            ts = datetime.fromisoformat(m.group(1)).replace(tzinfo=timezone.utc).timestamp()
            msgs.append((ts - wall_offset, m.group(2).strip(), m.group(3)))
            continue
        try:
            o = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if not isinstance(o, dict):
            continue
        if "replayChatItemAction" in o:
            info["format"] = "youtube-replay"
            a = o["replayChatItemAction"]
            try:
                t = int(a["videoOffsetTimeMsec"]) / 1000
            except (KeyError, ValueError):
                continue
            msgs += [(t, u, x) for u, x in _yt_text(a.get("actions", []))]
        elif "t" in o:
            info["format"] = "jsonl"
            t = float(o["t"])
            if t > 1e9:  # wall clock
                if wall_offset is None:
                    raise ValueError(f"{path}: wall-clock chat timestamps need --chat-offset (unix time of video second 0)")
                t -= wall_offset
            msgs.append((t, o.get("user") or "?", o.get("text") or ""))
    return sorted(msgs), info


def load(path, wall_offset=None, media_duration=None, align="auto"):
    """[(t_seconds, user, text)] sorted by time."""
    return load_with_info(path, wall_offset, media_duration, align)[0]


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
