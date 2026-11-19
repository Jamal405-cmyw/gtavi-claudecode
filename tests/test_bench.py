"""Benchmark metrics and the synthetic suite generator."""
import json
import pytest
import metrics


def test_one_to_one_matching_and_ranking_metrics():
    labels = [{"start": 10, "end": 20, "peak": 15}, {"start": 100, "end": 110, "peak": 105}]
    preds = [{"start": 0, "end": 40},      # covers label 1
             {"start": 5, "end": 25},      # also covers label 1 -> must not count twice
             {"start": 98, "end": 112},    # label 2
             {"start": 300, "end": 320}]   # miss
    m = metrics.evaluate(preds, labels)
    assert m["matched"] == 2 and m["p@R"] == 0.5 and m["r@R"] == 0.5
    assert m["ap"] == pytest.approx((1 / 1 + 2 / 3) / 2)
    assert m["setup_included"] == 1.0 and m["payoff_included"] == 1.0
    assert m["start_err_s"] == pytest.approx((10 + 2) / 2)


def test_perfect_and_empty():
    labels = [{"start": 0, "end": 10}]
    assert metrics.evaluate([{"start": 0, "end": 10}], labels)["ap"] == 1.0
    m = metrics.evaluate([], labels)
    assert m["ap"] == 0.0 and m["start_err_s"] is None
    agg = metrics.aggregate([m, metrics.evaluate([{"start": 0, "end": 10}], labels)])
    assert agg["ap"] == 0.5 and agg["micro_recall@R"] == 0.5


def test_synthetic_suite_is_deterministic(tmp_path):
    import make_synthetic
    make_synthetic.main(["--out", str(tmp_path / "a"), "--n", "1", "--seed", "5", "--dur", "60"])
    make_synthetic.main(["--out", str(tmp_path / "b"), "--n", "1", "--seed", "5", "--dur", "60"])
    la = json.loads((tmp_path / "a" / "synth_5_full" / "labels.json").read_text())
    lb = json.loads((tmp_path / "b" / "synth_5_full" / "labels.json").read_text())
    assert la == lb and la["highlights"]
    assert (tmp_path / "a" / "synth_5_full" / "transcript.json").read_text() == \
        (tmp_path / "b" / "synth_5_full" / "transcript.json").read_text()
