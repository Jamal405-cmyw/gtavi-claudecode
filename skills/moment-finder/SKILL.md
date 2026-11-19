---
name: moment-finder
description: Find the best clip moments in a long video, stream VOD or local file. Use when the user gives a YouTube URL or video file and wants highlights, "find the moments", or candidates to cut.
---

# Moment finder

Run:

`python "${CLAUDE_PLUGIN_ROOT}/scripts/highlights.py" <url-or-file> --top 8`

Add `--chat <file>` when you have a chat log (TwitchDownloader JSON, yt-dlp `*.live_chat.json`, JSONL or CSV; YouTube chat replay is picked up automatically). Add `--cam x:y:w:h` if the webcam position is known; otherwise it is detected.

It writes `jobs/<id>/highlights.md` (ranked, with scores and evidence) and `candidates.json`/`candidates.md` for `clip-cutter`. Signals: transcript (faster-whisper word timestamps; large-v3-turbo on an NVIDIA GPU, small on CPU), loudness, sudden sounds, shouting, laughter, chat bursts (lag-corrected), YouTube "most replayed", and, for the shortlist only, gameplay/facecam motion from sampled frames. A missing signal is skipped and listed under "note:", never faked.

If `highlights.py` fails, fall back to `python "${CLAUDE_PLUGIN_ROOT}/scripts/moments.py" <url-or-file> --top 8` (the original loudness/chat detector) and say so.

Then:

1. Read `jobs/<id>/highlights.md`. Show each pick: rank, start-end, length, entertainment score and confidence, the "why" evidence, the transcript line.
2. Scores are estimates that rank moments inside this video. Never present them as a view or virality prediction.
3. Drop spoilers for story moments in launch week, nudity, and anything that looks like a leak.
4. Recommend the best 3 and say what hook angle each suggests, based only on what is said and shown.
5. Next step: `clip-cutter` with `--pick <rank>`.

Record the source link and creator in `media/raw` notes. The creator must be credited on every clip.
