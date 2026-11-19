# Highlight benchmark

Compares the new two-stage engine (`scripts/highlights.py`) with the original detector, frozen
unmodified in `bench/baseline/` (copied from commit `67acdb4`).

```
pip install -r requirements.txt -r requirements-dev.txt
python bench/make_synthetic.py            # 9 deterministic synthetic cases -> bench/cases/
python bench/run_benchmark.py             # -> bench/results/<timestamp>/report.md + results.json
```

Systems: `baseline` (original moments.py), `new` (full engine), `new_novisual` (engine without the
frame-sampling stage, to show what the visual stage costs and adds).

## Two kinds of cases — only one of them measures clip quality

**Synthetic (`bench/make_synthetic.py`).** Generated audio, video, transcript and chat with labelled
events (laughter, quiet chuckle, shouting, explosion) and decoys (loud gameplay nobody reacts to, a
music bed, a mic bump, a chat spam bot). Deterministic, needs no downloads, runs in about a minute.
It proves the pipeline runs end to end and catches regressions. It **does not** prove real-world
quality: the same author wrote the events and the detectors, so the suite is biased toward what the
detectors look for. Treat its numbers as a smoke test.

**Real labelled recordings (what you should add).** Put each recording you are permitted to use in its
own folder:

```
bench/cases/my_stream_01/
  source.mp4          the VOD or a long excerpt (any length)
  labels.json         see below
  chat.json           optional: TwitchDownloader JSON, yt-dlp *.live_chat.json, JSONL or CSV
  transcript.json     optional: created once automatically and shared by all systems
```

```json
{"video": "source.mp4",
 "cam": [0.70, 0.05, 0.28, 0.30],
 "highlights": [
   {"start": 612.0, "end": 631.5, "peak": 620.0, "kind": "funny"},
   {"start": 1804.0, "end": 1822.0, "peak": 1811.0, "kind": "surprise"}]}
```

Labelling guide: watch the whole recording at 1x or 1.5x, mark every moment you would actually post,
`start` where a viewer needs to start to understand it, `end` after the payoff, `peak` at the key
second. Label *before* looking at either system's output, ideally two people per video. `cam` is
optional (fractions x:y:w:h); without it the engine auto-detects the webcam.

## Metrics

| metric | meaning |
|---|---|
| p@R, r@R | precision/recall in the top R picks, R = number of labelled highlights in that video |
| p@5 | precision of the top 5 |
| ap | average precision of the full ranked list (rewards putting real highlights first) |
| ndcg@5 | ranking quality of the top 5 |
| setup_included / payoff_included | matched clips that start before the label start (+1 s) / end after the label end (−1 s) |
| start_err_s / end_err_s | mean absolute boundary error of matched clips |
| mean_len_s | mean length of matched clips |
| sec_per_video_min | wall time per minute of video (transcription excluded; it is shared and reported separately) |
| cpu_s, peak_rss_mb | CPU seconds and peak memory of the process tree (needs `psutil`) |
| gpu_util_max | peak GPU utilisation from `nvidia-smi` (only on machines with an NVIDIA GPU) |

A prediction matches a label when the label's peak lies inside the clip or their temporal IoU ≥ 0.2;
matching is one-to-one, so one long clip cannot claim two highlights.
