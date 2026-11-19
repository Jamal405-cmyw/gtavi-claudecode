"""Two-stage highlight detection (successor to moments.py; writes the same candidates.json for cut.py).

Stage A scans the whole video cheaply (audio level/onset/timbre/laughter, transcript phrases, chat,
YouTube most-replayed). Stage B looks only at the shortlist: context-aware boundaries, a few sampled
frames for gameplay/facecam motion, seven-dimension scoring with confidence and evidence, and
duplicate removal. GPU is used for Whisper and frame decoding when available; CPU otherwise.

usage: python highlights.py <url-or-file> [--top 8] [--min 15] [--max 40] [--chat chat.json]
       [--chat-lag 5] [--chat-offset UNIX] [--cam auto|none|x:y:w:h] [--no-visual]
       [--whisper-model auto|small|large-v3-turbo|...] [--device auto|cuda|cpu] [--no-transcribe]
       [--max-minutes 90] [--dedupe-against jobs/*/highlights.json]
"""
import argparse, glob, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import ffprobe, job_dir, load_json
from highlight import accel, chat as CH
from highlight.engine import Config, analyse, has_video_stream, write_outputs


def find_chat(d, arg):
    if arg:
        return Path(arg)
    for pat in ("chat*.live_chat.json", "chat.json", "chat.jsonl", "chat.csv"):
        hit = sorted(d.glob(pat))
        if hit:
            return hit[0]
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--min", type=float, default=15); ap.add_argument("--max", type=float, default=40)
    ap.add_argument("--chat"); ap.add_argument("--chat-lag", type=float, default=5.0)
    ap.add_argument("--chat-offset", type=float, help="unix time of video second 0 (for live.py wall-clock chat logs)")
    ap.add_argument("--cam", default="auto"); ap.add_argument("--no-visual", action="store_true")
    ap.add_argument("--hwaccel", default="auto", help="ffmpeg decode accel for frame sampling: auto|cuda|none")
    ap.add_argument("--whisper-model", default="auto"); ap.add_argument("--device", default="auto")
    ap.add_argument("--no-transcribe", action="store_true", help="skip Whisper if no transcript.json is cached")
    ap.add_argument("--max-minutes", type=int, default=90)
    ap.add_argument("--dedupe-against", nargs="*", default=[], help="other highlights.json files (globs ok)")
    a = ap.parse_args(argv)

    import moments  # reuse download, WAV extraction and the cached Whisper transcription
    d = job_dir(a.src)
    vid, info = moments.fetch(a.src, d, a.max_minutes)
    info = load_json(d / "info.json") if (d / "info.json").exists() else info
    probe = ffprobe(vid)
    dur = float(probe["format"]["duration"])
    if (info.get("duration") or 0) > a.max_minutes * 60:
        print(f"note: only the first {a.max_minutes} min were downloaded and analysed (--max-minutes)")
    wav, _ = moments.audio_rms(vid, d)
    words = []
    if (d / "transcript.json").exists() or not a.no_transcribe:
        model, device, _ = accel.whisper_choice(a.whisper_model, a.device)
        if not (d / "transcript.json").exists():
            print(f"transcribing with faster-whisper {model} on {device}")
        words = moments.transcribe(wav, d, model=model, device=device)
    cpath = find_chat(d, a.chat)
    msgs = CH.load(cpath, wall_offset=a.chat_offset) if cpath else None
    others = []
    for pat in a.dedupe_against:
        for f in glob.glob(pat):
            if Path(f).resolve() != (d / "highlights.json").resolve():
                others += json.loads(Path(f).read_text(encoding="utf-8")).get("highlights", [])
    cfg = Config(top=a.top, min_len=a.min, max_len=a.max, chat_lag=a.chat_lag, chat_offset=a.chat_offset,
                 visual=not a.no_visual, cam=a.cam, hwaccel=a.hwaccel)
    res = analyse(vid, wav, words, dur, cfg, chat_msgs=msgs, info=info, d=d,
                  has_video=has_video_stream(probe), others=others)
    res["chat_file"] = str(cpath) if cpath else None
    write_outputs(d, res, vid, info)
    t = res["timings"]
    print(f"job: {d}\n{len(res['highlights'])} highlights ({len(res['duplicates'])} duplicates removed, "
          f"{res['shortlisted']} shortlisted) in {t['total_s']}s "
          f"[stage A {t['stage_a_s']}s, stage B {t['stage_b_s']}s, {t['vis_frames']} frames]")
    for x in res["notes"]:
        print("note:", x)
    print(f"-> {d / 'highlights.md'}")
    return d


if __name__ == "__main__":
    main()
