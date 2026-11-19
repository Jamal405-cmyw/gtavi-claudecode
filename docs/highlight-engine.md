# Highlight engine v1

`scripts/highlights.py` replaces `moments.py` as the recommended moment finder. It reuses the same
download, audio extraction and cached Whisper transcript, and writes the same `candidates.json`, so
`cut.py`, `translate.py` and every skill keep working. `moments.py` is unchanged apart from optional
Whisper model/device arguments and stays available.

```
python scripts/highlights.py <url-or-file> --top 8
python scripts/highlights.py vod.mp4 --chat chat.json          # TwitchDownloader / yt-dlp / JSONL / CSV
python scripts/highlights.py vod.mp4 --cam 0.70:0.05:0.28:0.30 # known webcam box (else auto-detected)
python scripts/cut.py jobs/<id> --pick 1 --layout react --cam auto --credit "@streamer"
```

## Architecture

```
            whole video, cheap                                   shortlist only (default 3 x top, min 15)
┌───────────────────────────────────────┐            ┌─────────────────────────────────────────────────┐
│ Stage A (per second)                  │            │ Stage B (per candidate)                          │
│ audio: level jump, onset, voice       │  top peaks │ context-aware boundaries (sentence/pause edges)  │
│   brightness, laughter heuristic      │ ─────────▶ │ frame sampling 2 fps, 256 px, CUDA decode→CPU     │
│ transcript: multi-word phrases,       │  (NMS 20s) │   gameplay motion, cuts, flashes, facecam motion │
│   repetition, speech rate             │            │ 7 scores + confidence + evidence                 │
│ chat: unique-chatter bursts, lag-     │            │ duplicate removal (time, speech, frame hashes)   │
│   corrected, laugh/hype/surprise mix  │            │ rank by entertainment → highlights.json + md     │
│ YouTube most-replayed heatmap         │            │                         candidates.json (cut.py) │
└───────────────────────────────────────┘            └─────────────────────────────────────────────────┘
```

| Module (`scripts/highlight/`) | Role |
|---|---|
| `audio.py` | 10 ms frames in 60 s chunks → per-second dB, jump, onset, brightness, laughter, activity |
| `text.py` | phrase matching on the token sequence, sentence edges, dangling-opener list |
| `chat.py` | loaders for YouTube replay, TwitchDownloader, JSONL (relative or wall-clock), CSV; burst features |
| `candidates.py` | Stage A fusion with per-modality weights (0 when a modality is missing) and peak picking |
| `boundaries.py` | event region, setup start, payoff end, length limits, dead-air trim |
| `visual.py` | ffmpeg frame sampling, motion baselines, dHash |
| `scoring.py` | dimension scores, confidence, evidence strings, disclaimer |
| `dedupe.py` | same-video and cross-job duplicate detection |
| `engine.py` | orchestration, timings, output files |
| `accel.py` | CUDA detection, Whisper model choice, ffmpeg hwaccel choice |

## Scores

All 0–1. They rank moments within one video and come with the evidence that produced them. They do
not predict views; `highlights.md` says so at the top.

| score | built from |
|---|---|
| humour | laughter-like audio, laugh words ("haha", "I'm dead"), share of laughing chat × chat burst |
| surprise | sharp onsets, surprise phrases ("no way", "what the"), "WHAT??" chat, hard cuts, flashes, gameplay motion |
| emotional_intensity | loudness jump, shouting timbre, fast speech, facecam movement, hype/tension words, repetition |
| contextual_clarity | words of setup before the event, clean sentence start/end, no dangling opener |
| standalone_watchability | length near 15–35 s, little dead air, clarity, whole event inside the clip |
| hook_potential | seconds until something happens; opening line is a question/exclamation |
| entertainment | 0.25 strongest of humour/surprise/intensity + 0.15 their mean + 0.2 Stage A strength + 0.15 crowd (chat/heatmap) + 0.15 watchability + 0.1 hook |
| confidence | share of available modalities that agree on the moment, plus how many modalities were available; lowered without a transcript |

## GPU and CPU

| Step | NVIDIA GPU (e.g. RTX 4070 SUPER) | CPU fallback |
|---|---|---|
| Whisper | `large-v3-turbo`, float16 (`--whisper-model auto`) | `small`, int8; also used if CUDA libraries fail to load |
| Frame sampling | `ffmpeg -hwaccel cuda` decode | software decode, automatic on any hwaccel error |
| Stage A / scoring | numpy on CPU (≈2 s per hour of audio on 2 cores; not worth a GPU) | same |

Override with `--whisper-model small|medium|large-v3|...`, `--device cpu`, `--hwaccel none`.

## Known limitations (v1)

- The laughter detector is a rhythm heuristic, not a trained audio classifier. It can miss soft or
  irregular laughs and can fire on rhythmic sounds; sustained rhythms (> 8 s) are damped.
- "Semantic understanding" is lexical: phrase lists, repetition, punctuation and sentence structure.
  There is no language model judging jokes; transcripts in languages other than English get no
  phrase signal (audio, chat and visual still work).
- Visual analysis measures motion, cuts and flashes, not what is on screen. On the synthetic suite it
  did not change the ranking and cost ~4x the CPU time of the rest of the engine; it matters most
  for streams whose highlights are visual and quiet. `--no-visual` turns it off.
- Without chat, quiet funny moments compete with loud gameplay nobody reacts to (seen in the
  synthetic no-chat cases).
- Twitch VOD chat still has to be supplied as a file (`--chat`), as before.
- Thresholds were set on synthetic signals. They need checking against labelled real recordings with
  `bench/run_benchmark.py` before anyone relies on the numbers.
