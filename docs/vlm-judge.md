# Optional local vision-language judge (Stage C) — investigation and design

Status: **implemented, tested against a mock server only.** No real model has been run yet: the
development sessions so far had no GPU and could not download models. Whether it improves ranking is
unknown until it is measured on real VODs (see "How to evaluate" below). It is off by default.

## Question

Can a local VLM on an RTX 4070 SUPER (12 GB) improve highlight ranking without excessive GPU use?

## Design choices

| Choice | Why |
|---|---|
| Runs after Stage B, on the top 6 clips only | Stage A/B stay fast; GPU cost is bounded and independent of VOD length |
| One image per clip: a 4-frame contact sheet of the event | Some Ollama vision models accept one image per turn; 4 frames show motion/reaction over time at the cost of one image |
| Verbatim transcript + setup/reaction lines in the prompt | The model judges what was said, not just pixels; the transcript is the strongest semantic signal |
| JSON schema in Ollama's `format`, temperature 0, strict validation | Malformed or out-of-range answers are rejected, never half-used |
| Blend weight 0.35 into entertainment | The VLM can reorder close calls but cannot override strong multi-signal evidence on its own |
| Cache per clip (model + prompt + image hash) | Re-runs and benchmark repeats cost no GPU time |
| `keep_alive: 0` on the last call | VRAM is released after the run (OBS/NVENC/games get the GPU back) |
| Ollama over HTTP, `urllib` only | No PyTorch/transformers install; same setup on Windows; nothing leaves the machine |
| Model text labelled "model-generated; not a quote, not for captions" | Prevents invented dialogue reaching a clip |

## Model options for 12 GB

Sizes from published Ollama figures; speed on your card must be measured.

| Model | Approx. size | Notes |
|---|---|---|
| `qwen2.5vl:7b` (default) | ~5.6 GiB weights | Commonly recommended small VLM in Ollama |
| `gemma3:4b` | ~2.6 GB | Lighter/faster fallback |
| `gemma3:12b` | ~7.6 GiB | Fits, less headroom while streaming/recording |
| text-only (`--judge-images 0`) | model-dependent | Tests whether the transcript alone carries the gain |

Whisper `large-v3-turbo` (fp16) and the VLM do not run at the same time: transcription finishes
before Stage C starts.

## Expected cost (to verify)

Per run: 6 requests, each one ~1.5k-token prompt + one 1536×216 image, ~60 output tokens. Cost is
fixed per VOD, not per minute. The benchmark records `judge_s`, `judge_calls` and peak GPU
utilisation/memory (nvidia-smi) so this table can be filled with measured numbers.

| measured on RTX 4070 SUPER | qwen2.5vl:7b | gemma3:4b |
|---|---|---|
| seconds per clip | not yet measured | not yet measured |
| peak VRAM | not yet measured | not yet measured |
| ranking change vs `new` | not yet measured | not yet measured |

## How to evaluate on your PC

```
ollama pull qwen2.5vl:7b
python scripts/highlights.py vod.mp4 --judge ollama            # one VOD, see judge notes in highlights.md
python bench/run_benchmark.py --systems baseline,new,new_judge # labelled cases
JUDGE_MODEL=gemma3:4b python bench/run_benchmark.py --systems new,new_judge
python bench/compare_vod.py ... --judge ollama                 # blind rating workflow (docs/real-vod-test.md)
```

Keep it on only if `new_judge` beats `new` on labelled or blind-rated real VODs by more than
run-to-run noise, at an acceptable `judge_s`.

## Risks

- Four frames can miss the moment, and GTA HUD/gameplay is unusual for general VLMs: expect it to
  read the transcript more than the frames.
- Models can be confidently wrong; the 0.35 weight and the cap of 6 clips limit the damage.
- Model answers are not reproducible across model versions; cache keys include the model name.
