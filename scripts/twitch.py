"""Twitch helpers that need no account and no API key.

top_streams(): Twitch's public web GQL endpoint with the public web Client-Id (same one yt-dlp uses).
irc_chat():    anonymous IRC read-only chat (justinfan nick, no password).
vod_chat():    VOD chat replay via TwitchDownloaderCLI (MIT, user-installed), if it is on PATH.
"""
import json, os, random, re, shutil, socket, ssl, subprocess, sys, threading, time
from pathlib import Path
from datetime import datetime, timezone

GQL = "https://gql.twitch.tv/gql"
CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass


def gql(query, variables=None):
    import requests
    r = requests.post(GQL, json={"query": query, "variables": variables or {}},
                      headers={"Client-Id": CLIENT_ID}, timeout=15)
    r.raise_for_status()
    return r.json()


_Q_GAME = """query($name:String!,$n:Int!){ game(name:$name){ displayName streams(first:$n){ edges{ node{
 title viewersCount createdAt previewImageURL(width:640,height:360) broadcaster{ login displayName } } } } } }"""
_Q_SEARCH = """query($q:String!){ searchCategories(userQuery:$q,first:5){ edges{ node{ ... on Game{ name displayName } } } } }"""


def _rows(game):
    out = []
    for e in ((game.get("streams") or {}).get("edges") or []):
        n = e.get("node") or {}
        b = n.get("broadcaster") or {}
        if not b.get("login"):
            continue
        out.append({"login": b["login"], "name": b.get("displayName") or b["login"], "title": n.get("title") or "",
                    "viewers": n.get("viewersCount") or 0, "created_at": n.get("createdAt"),
                    "preview": n.get("previewImageURL"), "url": f"https://twitch.tv/{b['login']}"})
    return out


def top_streams(game="Grand Theft Auto VI", limit=20):
    """Live streams of a Twitch category, sorted by viewers (desc). Empty list if the category is unknown/offline."""
    names = [game]
    try:
        d = gql(_Q_SEARCH, {"q": game})
        for e in ((d.get("data") or {}).get("searchCategories") or {}).get("edges") or []:
            nm = (e.get("node") or {}).get("name")
            if nm and nm not in names:
                names.append(nm)
    except Exception as ex:
        print(f"twitch category search failed: {ex}", file=sys.stderr)
    if game.lower() != "gta vi":
        names.append("Grand Theft Auto VI" if "grand theft" not in game.lower() else "GTA VI")
    for nm in names:
        try:
            g = ((gql(_Q_GAME, {"name": nm, "n": int(limit)}).get("data") or {}).get("game"))
        except Exception as ex:
            print(f"twitch gql failed: {ex}", file=sys.stderr)
            continue
        if g:
            rows = _rows(g)
            if rows:
                return sorted(rows, key=lambda r: r["viewers"], reverse=True)
    return []


def started_ts(iso):
    try:
        return datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except Exception:
        return None


def _parse_privmsg(line):
    """'@tags :user!user@user.tmi.twitch.tv PRIVMSG #chan :text' -> (user, text) or None"""
    tags = ""
    if line.startswith("@"):
        tags, _, line = line.partition(" ")
    if " PRIVMSG " not in line:
        return None
    prefix, _, rest = line.partition(" PRIVMSG ")
    user = prefix.lstrip(":").split("!")[0]
    for t in tags.lstrip("@").split(";"):
        if t.startswith("display-name=") and t[13:]:
            user = t[13:]
    _, _, text = rest.partition(" :")
    return user, text.strip()


def irc_chat(channel, out_path, stop_event=None, host="irc.chat.twitch.tv", port=6697):
    """Append {t, user, text} JSON lines to out_path until stop_event is set. Reconnects on drop.
    Uses TLS (6697); falls back to plain 6667 if TLS fails."""
    stop_event = stop_event or threading.Event()
    channel = channel.lower().lstrip("#")
    use_tls = True
    with open(out_path, "a", encoding="utf-8", buffering=1) as out:
        while not stop_event.is_set():
            sock = None
            try:
                raw = socket.create_connection((host, port if use_tls else 6667), timeout=15)
                sock = ssl.create_default_context().wrap_socket(raw, server_hostname=host) if use_tls else raw
                sock.settimeout(5)
                sock.sendall(f"NICK justinfan{random.randint(10000, 99999)}\r\n".encode())
                sock.sendall(f"JOIN #{channel}\r\n".encode())
                buf = b""
                while not stop_event.is_set():
                    try:
                        data = sock.recv(4096)
                    except socket.timeout:
                        continue
                    if not data:
                        break
                    buf += data
                    *lines, buf = buf.split(b"\r\n")
                    for ln in lines:
                        s = ln.decode("utf-8", "replace")
                        if s.startswith("PING"):
                            sock.sendall(("PONG" + s[4:] + "\r\n").encode())
                            continue
                        m = _parse_privmsg(s)
                        if m and m[1]:
                            out.write(json.dumps({"t": time.time(), "user": m[0], "text": m[1]}, ensure_ascii=False) + "\n")
            except (OSError, ssl.SSLError) as ex:
                if use_tls and not isinstance(ex, socket.timeout):
                    use_tls = False  # maybe TLS is blocked; try plain 6667 next
            finally:
                if sock:
                    try: sock.close()
                    except OSError: pass
            stop_event.wait(3)


CLI_HELP = ("Twitch VOD chat needs TwitchDownloaderCLI (https://github.com/lay295/TwitchDownloader/releases, "
            "MIT). Put TwitchDownloaderCLI(.exe) on PATH or set TWITCH_DOWNLOADER_CLI to its full path, "
            "or pass a chat file with --chat.")


def vod_id(url_or_id):
    """'https://www.twitch.tv/videos/123456789?t=1h' or 'v123456789' or '123456789' -> '123456789' (else None)."""
    s = str(url_or_id).strip()
    m = re.search(r"twitch\.tv/(?:[^/]+/)?videos?/(\d+)", s) or re.fullmatch(r"v?(\d{5,})", s)
    return m.group(1) if m else None


def find_cli():
    env = os.environ.get("TWITCH_DOWNLOADER_CLI")
    if env:
        return env if Path(env).is_file() else None
    return shutil.which("TwitchDownloaderCLI") or shutil.which("TwitchDownloaderCLI.exe")


def vod_chat(video, out, end_seconds=None, timeout=3600):
    """Download a VOD's chat replay to `out` + '.gz' (TwitchDownloader JSON, gzip) and return that path.
    Offsets in the file are seconds from the start of the VOD. `end_seconds` stops early (use the same
    limit as the video download). Raises RuntimeError with install instructions if the CLI is missing."""
    vid = vod_id(video)
    if not vid:
        raise ValueError(f"not a Twitch VOD url or id: {video}")
    exe = find_cli()
    if not exe:
        raise RuntimeError(CLI_HELP)
    out = Path(out)
    if out.suffix.lower() != ".json":
        out = out.with_suffix(".json")
    cmd = [exe, "chatdownload", "--id", vid, "-o", str(out), "--compression", "Gzip", "--collision", "Overwrite"]
    if end_seconds:
        cmd += ["-e", f"{int(end_seconds)}s"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    got = Path(str(out) + ".gz")
    if r.returncode != 0 or not got.exists():
        raise RuntimeError(f"TwitchDownloaderCLI failed ({r.returncode}): {(r.stderr or r.stdout).strip()[-400:]}")
    return got
