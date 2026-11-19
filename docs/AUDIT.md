# Technical audit — gtavi-claudecode (branch `feature/ai-highlight-engine-v1`)

Audited at commit `67acdb4` on 2026-11-19. Scope: every file in `scripts/`, `skills/`, `.claude-plugin/`,
`requirements.txt` and `README.md`. Findings are ordered by priority (P0 = fix first).

## 1. What the system actually is

A Claude Code plugin: 16 `SKILL.md` prompt files that drive 13 standalone Python scripts. There is no
package, no tests, no CI, no lock file and no shared configuration. Each script is a CLI that reads and
writes JSON/Markdown under `jobs/<id>/`.

| Area | Script | How it works |
|---|---|---|
| Download / ffmpeg / paths | `common.py` | yt-dlp via `python -m yt_dlp`, ffprobe JSON, `jobs/<slug>-<sha1>` job folders |
| Moment detection | `moments.py` | Per-second score = loudness jump + hype-word count + YouTube "most replayed" heatmap + chat rate; greedy non-overlapping windows around peaks |
| Transcription | `moments.transcribe` | faster-whisper `small`, word timestamps, VAD, CUDA if ctranslate2 sees a GPU else CPU int8; cached in `transcript.json` |
| Cutting | `cut.py` | One ffmpeg pass: layout filter graph + ASS word captions + loudnorm −14 LUFS → 1080×1920 |
| Facecam | `facecam.py` | YuNet face detection on 12 frames, stable-face clustering, median static-edge rectangle search |
| Live | `live.py` + `twitch.py` | Segmented ffmpeg recording, YouTube live_chat tail or anonymous Twitch IRC, chat-spike auto cut |
| Other | `check.py`, `translate.py`, `streams.py`, `trends.py`, `questions.py`, `secrets_map.py`, `payouts.py` | Rule check, translation, discovery, content helpers |

## 2. Prioritised findings

### P0 — correctness of highlight detection (the core value)

1. **Multi-word hype phrases never match.** `moments.py` runs `HYPE.search()` on single transcript
   *words*, so `no way`, `oh my god`, `let's go`, `look at`, `go go` can never fire. Verified:
   `HYPE.search("no") → False`, `HYPE.search("way") → False`.
2. **Hype lexicon fires on ordinary speech.** `what`, `run`, `bro`, `wait`, `nah` match in normal talk,
   so the transcript signal mostly measures *how much someone talks*, not excitement.
3. **No chat-delay compensation.** Chat reacts 3–10 s after the event; the chat series is used unshifted,
   so chat peaks point after the moment. The fixed window (`peak − 0.55 × max`) only partly hides this.
4. **Loudness-dominated scoring.** Loudness jump has weight 1.0 with no notion of laughter, shouting
   timbre, speech rate or visuals. Mic bumps, music stings and sustained loud gameplay rank as highlights.
5. **Boundaries are not context-aware.** Start = peak − 22 s regardless of where the sentence or setup
   begins; end = last word inside a fixed window. Clips start mid-sentence and can end before the payoff.
6. **No duplicate detection.** Only exact time overlap is blocked; replays/re-watched moments and
   near-identical neighbouring peaks become separate candidates.
7. **Edge bias.** `np.convolve(..., mode="same")` zero-pads, which distorts the first/last ~15 s of the
   loudness baseline and chat rate.
8. **No visual signal at all**, although the README positions the product around streamer reactions.

### P1 — efficiency and reliability

9. **Full 1080p download before any analysis.** Stage-1 signals need only audio, yet `fetch()` downloads
   the full video first. For a 90-minute VOD this dominates wall time. (Not changed in this iteration;
   proposed in §4.)
10. **Silent truncation.** `--max-minutes 90` downloads only the first 90 minutes; later moments are never
    considered and nothing in the output says so.
11. **Whisper model fixed at `small`.** An RTX 4070 SUPER can run `large-v3-turbo` in float16 faster than
    a CPU runs `small`; there is no way to choose.
12. **Whole-file audio in RAM** (fine to ~2 h at 16 kHz mono, ~460 MB float32) — acceptable, noted.
13. **Twitch VOD chat is not implemented** (`twitch.vod_chat` raises). Chat for Twitch VODs only works if
    the user supplies a TwitchDownloader JSON via `--chat`. `live.py`'s `chat.twitch.jsonl` (wall-clock
    timestamps) cannot be fed back into `moments.py`.
14. **`cut.py --pick N` with an unknown N** raises a bare `StopIteration`.

### P1 — security and dependency risk

15. **Unverified model download.** `facecam.py` fetches an ONNX model from GitHub at runtime with no
    checksum or pinned commit (supply-chain risk; the file is executed by OpenCV's DNN runtime).
16. **Unbounded dependency versions** (`>=` only, no lock). A breaking faster-whisper/yt-dlp release
    breaks every install. yt-dlp must float (YouTube changes), the rest should be capped.
17. **Undocumented web endpoints.** `twitch.py` uses Twitch's private web GQL endpoint with the public
    web Client-Id; `streams.py` uses Kick's internal API. Both can break or violate platform terms at any
    time. Helix (official) is supported in `streams.py` when credentials are set; prefer it.
18. No `shell=True`, `eval` or pickle anywhere; subprocess calls use argument lists. Good.
19. ffmpeg `subtitles=` path escaping handles `\` and `:` but not `'` — a job path containing an
    apostrophe breaks rendering (robustness, not a security hole).

### P2 — documentation vs implementation

20. README: "moment-finder … chat spikes" — only true for YouTube VODs with chat replay or a manual
    TwitchDownloader file. Twitch VOD chat is not fetched.
21. README / skill: "best 15 to 40 s moments" — length bounds yes, "best" is loudness-driven (P0 #4).
22. `.claude-plugin/plugin.json` and the README install lines point at `Bodila51/gtavi-claudecode`, not
    this fork. Left unchanged (licence/attribution), flagged for the owner.
23. No tests exist anywhere, so none of the claims are verified automatically.

## 3. Reuse decisions for the new engine

| Reused as-is | Why |
|---|---|
| `moments.fetch` | Download, YouTube heatmap + chat replay capture |
| `moments.audio_rms` (its `audio.wav`) | 16 kHz mono WAV is exactly what the new audio features need |
| `moments.transcribe` + `transcript.json` cache | Same word format; extended with an optional model name, defaults unchanged |
| `facecam.find_cam` | Facecam box for the visual stage, cached per job |
| `candidates.json` format | New engine writes the same fields, so `cut.py`, `translate.py` and skills keep working |

## 4. Plan (this iteration vs later)

This iteration (implemented, see `docs/highlight-engine.md`): two-stage highlight engine, chat loaders
for all formats the repo produces, GPU/CPU device selection, context-aware boundaries, duplicate
detection, 7-dimension scoring with confidence and evidence, unit + integration tests, a reproducible
benchmark against a frozen copy of the original detector.

Proposed next (not implemented): audio-only first download with per-candidate video section fetch (#9),
Twitch VOD chat via official means (#13), pinned + checksummed model downloads (#15), dependency caps
and lock file (#16), live monitoring service, dashboard.
