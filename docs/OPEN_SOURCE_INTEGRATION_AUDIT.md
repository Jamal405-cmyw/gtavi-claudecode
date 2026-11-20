# Open-source integration audit

Date: 2026-11-20 · Branch `feature/ai-highlight-engine-v1` · Target machine: Windows, Python 3.14.5,
RTX 4070 SUPER (12 GB), i7-14700KF, 32 GB RAM.

## Method

Each repository was partial-cloned at its current default branch (metadata only, files fetched on
demand, nothing executed) and its licence file, dependency manifest and the source files behind the
feature in question were read. Windows / Python 3.14 compatibility was checked by asking PyPI for
`win_amd64` wheels for CPython 3.14 (`pip download --only-binary`, nothing installed). Statements
about GPU speed are expectations, not measurements: no GPU was available in this environment.

Decision key: **Integrate** = call or import it · **Adapt** = re-implement a specific, documented
technique in our code (no dependency) · **Already used** · **Defer** · **Reject**.

## Ranked summary

| # | Project (last commit) | Licence | Decision | Value | Risk | Effort | Status in this branch |
|---|---|---|---|---|---|---|---|
| 1 | TwitchDownloader (2026-10-06) | MIT | **Integrate** (optional external CLI) + read all its chat formats | High | Low | S | **Implemented** |
| 2 | Silero VAD (2026-10-22) | MIT (code + model) | **Integrate** via the copy bundled in faster-whisper (no new dependency) | High | Low | S | **Implemented** |
| 3 | ClipsAI (2024-01-17) | MIT | **Adapt** piecewise-static reframing split at scene cuts; reject the package | High | Low | M | **Implemented** (technique) |
| 4 | PySceneDetect (2026-11-19) | BSD-3 | **Adapt** content-delta + adaptive-ratio cut detection on sampled frames | Med | Low | S | **Implemented** (technique) |
| 5 | Auto-Editor (2026-11-20) | Unlicense | **Adapt** loudness-threshold + margin silence detection | Med | Low | S | **Implemented** (technique, VAD fallback) |
| 6 | OpenCV (2026-11-20) | Apache-2.0 | **Already used** (YuNet facecam); now also face-centred cam panel | Med | Low | S | **Implemented** |
| 7 | FFmpeg (2026-11-19) | LGPL/GPL (called as a binary) | **Already used**; time-varying `crop` expressions for reframing | High | Low | S | **Implemented** |
| 8 | faster-whisper (2026-11-12) | MIT; Whisper weights MIT | **Already used**; next: `BatchedInferencePipeline`, `hotwords`, `hallucination_silence_threshold` | Med | Low | S | Researched (needs GPU to verify) |
| 9 | Streamlink (2026-11-18) | BSD-2 | **Defer → integrate** as optional live-capture backend (Twitch ad filtering) | Med | Low | M | Researched |
| 10 | yt-dlp (2026-11-19) | Unlicense | **Already used**; next: `--download-sections` for audio-first analysis | Med | Low | M | Researched |
| 11 | WhisperX (2026-11-12) | BSD-2; align models vary | **Reject for now** (incompatible with Python 3.14) | Med | High | L | Researched |
| 12 | Subtitle Edit (2026-11-20) | MIT | **Adapt** timing rules for caption QC (later) | Low-Med | Low | S | Researched |
| 13 | Remotion (2026-11-20) | Remotion licence (source-available) | **Defer** (templates later; licence matters for a SaaS) | Med | Med | L | Researched |
| 14 | remotion-dev/skills (2026-11-18) | none stated (all rights reserved) | **Defer**: read for ideas only, do not copy | Low | Med | – | Researched |
| 15 | OBS Studio (2026-11-19) | GPL-2.0 | **Defer**: obs-websocket replay-buffer trigger for your own streams | Low-Med | Low | M | Researched |
| 16 | MediaPipe (2026-11-19) | Apache-2.0 | **Reject**: runtime model downloads, OpenCV package conflict | Low | Med | M | Researched |
| 17 | LosslessCut (2026-11-03) | GPL-2.0 | **Reject** (GUI app); technique noted: keyframe-aware stream copy | Low | – | – | Researched |
| 18 | MoviePy (2026-08-26) | MIT | **Reject**: slow per-frame Python, bundles its own ffmpeg | Low | Med | – | Researched |
| 19 | OpenCut (2026-09-24) | MIT | **Reject** (web editor app, not a library); dashboard inspiration later | Low | – | – | Researched |
| 20 | FunClip (2026-11-01) | MIT; FunASR models own terms | **Reject**: Chinese-first ASR, cloud LLM, pinned old deps | Low | High | – | Researched |

## Per-project findings

### 1. TwitchDownloader — integrate (optional CLI) · implemented
- **Source read:** `TwitchObjects/ChatRoot.cs`, `ChatDownloader.cs`, `Chat/ChatJson.cs`, `Chat/ChatText.cs`,
  `CLI/Modes/Arguments/ChatDownloadArgs.cs`.
- **Facts that matter for us:** `comments[].content_offset_seconds` is an offset from the *start of the
  VOD*, also for trimmed (`-b/-e`) downloads and clips; the trim window is stored in `video.start` /
  `video.end`. JSON can be gzip-compressed (`--compression Gzip`). Files carry `FileInfo.Version`
  (current 1.4.0). TXT export writes `[h:mm:ss] user: message` (relative = VOD offset) or UTC stamps.
- **We already had:** reading plain TwitchDownloader JSON.
- **Gaps fixed:** gzip files, TXT exports, chat for a trimmed/clip download being misaligned by
  `video.start` seconds, no way to fetch Twitch VOD chat at all (`twitch.vod_chat` raised; yt-dlp's
  Twitch extractor has no chat/rechat support).
- **Windows:** official self-contained `TwitchDownloaderCLI.exe` releases; no .NET install needed.
- **Risks:** it uses Twitch's unofficial GQL API (can break; ToS grey area, same as our `twitch.py`).
  We never download the binary for you; it must be on PATH or in `TWITCH_DOWNLOADER_CLI`.
- **Overhead:** network-bound; JSON parse of a 6 h VOD chat ≈ seconds.

### 2. Silero VAD — integrate via faster-whisper · implemented
- **Source read:** `snakers4/silero-vad` package layout; `faster_whisper/vad.py` (installed 1.2.1 ships
  `assets/silero_vad_v6.onnx` and `get_speech_timestamps(audio, VadOptions)` on onnxruntime).
- **We already had:** faster-whisper's internal VAD filter during transcription only.
- **Improvement:** speech segments for Stage B boundaries: never start or end a clip mid-utterance,
  snap to real speech onsets/offsets (Whisper word edges drift by a few hundred ms), and boundaries
  that respect speech even when there is no transcript.
- **Cost:** run only on shortlisted windows (±20 s around each candidate), CPU, no download, no new
  package (onnxruntime is already a faster-whisper dependency). Falls back to an Auto-Editor-style
  loudness method if faster-whisper is missing.
- **Caveat:** GTA NPC dialogue is also speech.

### 3. ClipsAI — adapt reframing technique · implemented
- **Source read:** `clipsai/resize/resizer.py`, `clip/clipfinder.py`, `clip/texttiler.py`, README.
- **How it works:** splits the clip at scene changes and speaker changes, samples ~13 frames per
  segment, detects faces (MTCNN), picks the active speaker by mouth-aspect-ratio (MediaPipe), and uses
  one **static crop per segment** (then merges identical neighbours) — no continuous panning, so no
  jitter. Clip finding is TextTiling over sentence embeddings (topic boundaries; built for podcasts).
- **Why not the package:** needs torch, pyannote with a gated Hugging Face token, MTCNN, MediaPipe,
  WhisperX; last commit January 2024; speaker-centric (a GTA stream's subject is gameplay, not a
  talking head).
- **Adapted:** piecewise-static gameplay crop per scene segment, chosen by motion saliency + a centre
  prior, excluding the webcam overlay; neighbouring segments with similar crops merged.

### 4. PySceneDetect — adapt detector · implemented
- **Source read:** `detectors/content_detector.py`, `detectors/adaptive_detector.py`.
- **Technique:** ContentDetector = mean absolute HSV/edge delta between frames vs threshold 27;
  AdaptiveDetector = ratio of a frame's score to the rolling mean of neighbours (threshold 3.0,
  min content 15). We sample few frames per clip, so we use the luma delta with the adaptive ratio.
- **Why not the package:** small (click, numpy, opencv-python, platformdirs, tqdm), but it decodes
  every frame; we only need cuts inside a 15–40 s clip at 4 fps. `opencv-python` would also duplicate
  our `opencv-python-headless`.

### 5. Auto-Editor — adapt silence technique · implemented (as fallback)
- **Source read:** `src/cli.nim`, `src/edit.nim` (now written in Nim, not a Python library).
- **Technique:** a frame is "loud" if its peak amplitude > 0.04 of full scale; 0.2 s margin around
  loud sections; smoothing with 0.2 s min-cut / 0.1 s min-clip.
- **Used for:** the speech-segment fallback and dead-air checks when VAD is unavailable. Jump-cut
  removal of pauses inside clips is a later editing feature.

### 6. OpenCV — already used · extended
- YuNet (`cv2.FaceDetectorYN`) already finds the webcam. `find_cam` also returns the face box; the cam
  panel now centres on that face (headroom above, face on the upper third) instead of the box centre.
- Windows/Py3.14: `opencv-python-headless` 5.1 has an abi3 `win_amd64` wheel.

### 7. FFmpeg — already used · extended
- Reframing uses `crop` with a per-frame `x` expression (piecewise constant), one encode pass, no extra
  dependency. Worth adopting later: `scdet` (scene detection filter), two-pass `loudnorm`,
  `silencedetect`.

### 8. faster-whisper — already used · next steps researched
- `BatchedInferencePipeline` (batched GPU inference), `hotwords` / `initial_prompt` (GTA names:
  Lucia, Jason, Leonida, Vice City), `hallucination_silence_threshold` (reduces invented text over
  long game-audio stretches). All exist in the installed 1.2.1 API. Need a GPU run to measure.

### 9. Streamlink — defer (next for live capture)
- BSD-2; deps: requests, urllib3, lxml, pycryptodome, pycountry, trio, websocket-client, isodate,
  PySocks, certifi; `streamlink-8.6.3-py3-none-win_amd64` wheel exists.
- Its Twitch HLS reader always filters ad segments (`should_filter_segment`) and supports low-latency
  playlists. Our `live.py` resolves one HLS URL with yt-dlp and records with ffmpeg, so mid-roll ads
  land in recordings. Plan: optional `streamlink --stdout` → ffmpeg segmenter backend.

### 10. yt-dlp — already used
- Next: `--download-sections` to fetch only candidate windows after an audio-only first pass
  (AUDIT.md P1 #9). The Twitch extractor provides no chat.

### 11. WhisperX — reject for now
- `pyproject.toml`: `requires-python >=3.10,<3.14` — **does not install on your Python 3.14.5**.
  Also pins `torch~=2.8`, needs `pyannote-audio>=4` (gated models) and pandas/transformers; English
  alignment downloads `facebook/wav2vec2-base-960h`.
- Revisit only if real-footage tests show word timing is the limiting factor, in a separate Python 3.13
  venv. Meanwhile VAD snapping addresses the clip-edge part of the problem.

### 12. Subtitle Edit — adapt rules later
- Defaults in `src/libse/Settings/Settings.cs`: min display 1000 ms, max 8000 ms, min gap 24 ms,
  max 25 chars/s, 43 chars/line, 2 lines. Good basis for caption QC in `check.py` (Phase 3).

### 13–14. Remotion and remotion-dev/skills — defer
- Remotion licence: free for individuals, non-profits and companies with ≤ 3 employees; a paid
  Company Licence otherwise — matters if this becomes a SaaS. Renders React frames in headless
  Chromium (Node toolchain), slower than one ffmpeg pass. Useful later for animated caption/overlay
  templates if ASS styling is not enough.
- remotion-dev/skills: agent guidance (`SKILL.md` per topic); no LICENSE file and no licence field →
  default copyright; read for ideas, do not copy text.

### 15. OBS Studio — defer
- GPL-2 application; integrate only over obs-websocket (built into OBS 28+) to save the replay buffer
  when the live detector fires on *your own* stream. No code reuse.

### 16. MediaPipe — reject
- The 1.2.3 wheel (`py3-none-win_amd64`, 25.7 MB) bundles **no** face models (Tasks API downloads
  `.tflite` files at runtime) and requires `opencv-contrib-python`, `matplotlib`, `sounddevice`.
  Installing contrib next to `opencv-python-headless` causes conflicting `cv2` packages. YuNet covers
  face detection; face landmarks (mouth movement) are not needed for gameplay framing.

### 17. LosslessCut — reject
- Electron GUI, GPL-2. Technique noted: keyframe-aware stream copy for instant raw cuts (useful for
  archiving source segments, not for 9:16 renders which must re-encode).

### 18. MoviePy — reject
- MIT; 2.2.1; renders frame by frame in Python and ships its own ffmpeg via `imageio-ffmpeg`.
  Everything we need is one ffmpeg filter graph.

### 19. OpenCut — reject
- MIT web video editor (Next.js app). Not a library. Possible UI reference for the deferred dashboard.

### 20. FunClip — reject
- MIT code; Paraformer ASR is Chinese-first, LLM clipping sends SRT text to cloud APIs (OpenAI, Qwen,
  g4f), boundaries are fixed millisecond offsets; pins `numpy==1.26.4`, `moviepy==1.0.3`, gradio 4.
  Our optional local judge already covers "LLM reads the transcript and rates clips".

## Security and licensing notes

- Every new integration is either a re-implemented technique (no third-party code copied) or an
  external program called with an argument list (no shell). GPL programs (FFmpeg GPL builds, OBS,
  LosslessCut) are only ever run as separate executables.
- No new Python dependencies were added in this branch for these integrations.
- TwitchDownloaderCLI is user-installed from its GitHub releases; we do not download executables.
- Model licences: Whisper weights MIT; Silero VAD MIT (bundled in faster-whisper); YuNet MIT
  (OpenCV Zoo). The existing YuNet download in `facecam.py` still has no checksum (AUDIT.md #15).

## Implemented integrations: evidence and measured overhead

Measured in the cloud development environment (2 CPU cores, no GPU) on synthetic material only.

| Integration | Tests | Overhead measured | Real-footage effect |
|---|---|---|---|
| TwitchDownloader formats + CLI | `tests/test_twitch_chat.py` (gzip, TXT, fragments, alignment, fake CLI arguments); the previous loader crashed on gzip chat | parse cost negligible; download time is network-bound | not measured (no real chat download possible here) |
| Silero VAD edge snapping | `tests/test_speech_boundaries.py` (rules on hand-built segments, VAD on synthetic voice, fallback, engine) | ~0.5 s per 5-min video (shortlisted windows only) | not measured; on the synthetic suite mean start/end error went 2.73/5.96 s → 2.27/5.75 s, which only shows the mechanics work |
| Reframing (ClipsAI + PySceneDetect techniques) | `tests/test_reframe.py` (cut rule, centre/cap/webcam rules, piecewise plan, face crop, rendered frames, byte-identical old graphs) | ~1.1 s planning per 30 s clip at 360p | not measured; on a synthetic frame the centred player went from ~67 % visible (old edge crop) to 100 %, webcam not duplicated |

## What still needs your PC

GPU speed of Whisper batching, real-footage effect of VAD snapping and reframing, and real Twitch chat
downloads. See `docs/real-vod-test.md`; nothing here was run on your machine.
