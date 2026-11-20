"""TwitchDownloader integration: file formats (from its ChatRoot.cs / ChatText.cs schema) and the optional CLI."""
import gzip, json, os, stat, sys
import numpy as np
import pytest
from highlight import chat
import twitch


def td_json(comments, start=0.0, end=600.0, length=600.0, version=(1, 4, 0)):
    """Minimal TwitchDownloader 1.4 chat file (fields as in TwitchObjects/ChatRoot.cs)."""
    return {"FileInfo": {"Version": dict(zip(("Major", "Minor", "Patch"), version))},
            "streamer": {"name": "Streamer", "id": 1},
            "video": {"title": "GTA stream", "id": "123456789", "start": start, "end": end, "length": length},
            "comments": [{"_id": f"c{i}", "content_offset_seconds": t, "commenter": {"display_name": u, "_id": u},
                          "message": {"body": b, "fragments": [{"text": b, "emoticon": None}]}}
                         for i, (t, u, b) in enumerate(comments)]}


def test_gzip_json_and_version(tmp_path):
    p = tmp_path / "chat.json.gz"
    p.write_bytes(gzip.compress(json.dumps(td_json([(12.5, "ann", "KEKW"), (3.0, "bob", "hi")])).encode()))
    msgs, info = chat.load_with_info(p)
    assert msgs == [(3.0, "bob", "hi"), (12.5, "ann", "KEKW")]
    assert info["format"] == "twitchdownloader-json" and info["version"] == "1.4.0" and info["shift"] == 0.0


def test_empty_body_falls_back_to_fragments(tmp_path):
    js = td_json([(5.0, "c", "")])
    js["comments"][0]["message"]["fragments"] = [{"text": "LUL "}, {"text": "what", "emoticon": None}]
    p = tmp_path / "chat.json"; p.write_text(json.dumps(js))
    assert chat.load(p) == [(5.0, "c", "LUL what")]


def test_align_offset_rules():
    full = {"start": 0, "end": 7200, "length": 7200}
    win = {"start": 3600, "end": 3900, "length": 7200}
    assert chat.align_offset(full, 7200) == 0.0
    assert chat.align_offset(win, 300) == -3600          # the media file IS the trimmed window/clip
    assert chat.align_offset(win, 7200) == 0.0           # full VOD video + trimmed chat: offsets already right
    assert chat.align_offset(win, None) == 0.0           # unknown -> keep VOD offsets
    assert chat.align_offset(win, 7200, "trimmed") == -3600 and chat.align_offset(win, 300, "vod") == 0.0
    assert chat.align_offset(win, 300, 12.5) == 12.5


def test_trimmed_chat_lands_on_the_right_second(tmp_path):
    """Regression: chat downloaded with -b 3600s for a 300 s clip used VOD offsets (3600+), i.e. off the end."""
    crowd = [(3600 + 150 + i * 0.1, f"fan{i}", "KEKW") for i in range(40)]
    base = [(3600 + t, f"u{t % 40}", "hi") for t in range(0, 300)]
    p = tmp_path / "chat.json"; p.write_text(json.dumps(td_json(crowd + base, start=3600, end=3900, length=7200)))
    msgs, info = chat.load_with_info(p, media_duration=300.0)
    assert info["shift"] == -3600
    f = chat.features(msgs, 300, lag=5)
    assert 143 <= int(np.argmax(f["burst"])) <= 150


def test_txt_relative_and_utc(tmp_path):
    rel = tmp_path / "chat.txt"
    rel.write_text("[0:00:05] ann: hello\n[1:02:03] bob: KEKW: so funny\nnot a chat line\n", encoding="utf-8")
    msgs, info = chat.load_with_info(rel)
    assert msgs == [(5, "ann", "hello"), (3723, "bob", "KEKW: so funny")] and info["format"] == "twitchdownloader-txt"
    utc = tmp_path / "chat_utc.txt"
    utc.write_text("[2026-11-20 13:00:10 UTC] ann: W\n[2026-11-20 13:00:12.500 UTC] bob: POG\n", encoding="utf-8")
    with pytest.raises(ValueError):
        chat.load(utc)
    t0 = 1795179600.0  # 2026-11-20 13:00:00 UTC
    assert chat.load(utc, wall_offset=t0) == [(10.0, "ann", "W"), (12.5, "bob", "POG")]


def test_youtube_and_jsonl_still_work_gzipped(tmp_path):
    item = {"replayChatItemAction": {"videoOffsetTimeMsec": "1500", "actions": [{"addChatItemAction": {"item": {
        "liveChatTextMessageRenderer": {"authorName": {"simpleText": "z"}, "message": {"runs": [{"text": "LOL"}]}}}}}]}}
    p = tmp_path / "chat.live_chat.json.gz"
    p.write_bytes(gzip.compress((json.dumps(item) + "\n" + json.dumps(item) + "\n").encode()))
    assert chat.load(p) == [(1.5, "z", "LOL"), (1.5, "z", "LOL")]


@pytest.mark.parametrize("src,vid", [("https://www.twitch.tv/videos/2345678901", "2345678901"),
                                     ("https://twitch.tv/somechannel/video/2345678901?t=1h2m", "2345678901"),
                                     ("v2345678901", "2345678901"), ("2345678901", "2345678901"),
                                     ("https://www.twitch.tv/somechannel", None), ("https://youtu.be/abc", None)])
def test_vod_id(src, vid):
    assert twitch.vod_id(src) == vid


def test_missing_cli_gives_install_instructions(monkeypatch):
    monkeypatch.setenv("TWITCH_DOWNLOADER_CLI", "/nonexistent/TwitchDownloaderCLI")
    with pytest.raises(RuntimeError, match="TwitchDownloaderCLI"):
        twitch.vod_chat("https://www.twitch.tv/videos/123456789", "chat.json")


@pytest.mark.skipif(os.name == "nt", reason="fake CLI is a POSIX script")
def test_cli_invocation_with_fake_binary(tmp_path, monkeypatch):
    log = tmp_path / "args.json"
    fake = tmp_path / "TwitchDownloaderCLI"
    fake.write_text(f"""#!{sys.executable}
import gzip, json, sys
a = sys.argv[1:]
json.dump(a, open({str(log)!r}, "w"))
out = a[a.index("-o") + 1] + ".gz"
data = {{"FileInfo": {{"Version": {{"Major": 1, "Minor": 4, "Patch": 0}}}}, "video": {{"start": 0, "end": 5400, "length": 9000}},
        "comments": [{{"content_offset_seconds": 42.0, "commenter": {{"display_name": "x"}}, "message": {{"body": "KEKW"}}}}]}}
open(out, "wb").write(gzip.compress(json.dumps(data).encode()))
""")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("TWITCH_DOWNLOADER_CLI", str(fake))
    got = twitch.vod_chat("https://www.twitch.tv/videos/987654321", tmp_path / "chat.json", end_seconds=5400)
    args = json.loads(log.read_text())
    assert args[:3] == ["chatdownload", "--id", "987654321"]
    assert args[args.index("--collision") + 1] == "Overwrite"     # default "Prompt" would hang unattended runs
    assert args[args.index("--compression") + 1] == "Gzip" and args[args.index("-e") + 1] == "5400s"
    assert got.name == "chat.json.gz" and chat.load(got) == [(42.0, "x", "KEKW")]
