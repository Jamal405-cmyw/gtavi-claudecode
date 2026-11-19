"""Two-stage highlight engine.

Stage A (cheap, whole video): per-second audio, transcript and chat signals -> shortlist of peaks.
Stage B (shortlist only): context-aware boundaries, sampled-frame visual analysis, multi-dimension
scoring with confidence and evidence, duplicate removal, ranking.

Entry point: highlight.engine.run(...) or scripts/highlights.py on the command line.
"""
ENGINE_VERSION = "1.0.0"
