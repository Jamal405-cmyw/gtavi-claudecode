"""Highlight-detection metrics. Predictions are ranked lists of {"start","end"}; labels {"start","end","peak"?}.

A prediction matches a label when the label's peak lies inside the clip, or their temporal IoU >= 0.2.
Matching is one-to-one in rank order, so one long clip cannot claim two labels.
"""
import math


def tiou(a, b):
    inter = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return inter / union if union > 0 else 0.0


def hit(p, l, iou_min=0.2):
    peak = l.get("peak", (l["start"] + l["end"]) / 2)
    return p["start"] <= peak <= p["end"] or tiou(p, l) >= iou_min


def match(preds, labels, iou_min=0.2):
    """[(pred_index, label_index)] greedy by prediction rank, best IoU label first."""
    used, pairs = set(), []
    for i, p in enumerate(preds):
        cands = [(tiou(p, l), j) for j, l in enumerate(labels) if j not in used and hit(p, l, iou_min)]
        if cands:
            j = max(cands)[1]
            used.add(j); pairs.append((i, j))
    return pairs


def evaluate(preds, labels):
    n = len(labels)
    pairs = match(preds, labels)
    rel = [0] * len(preds)
    for i, _ in pairs:
        rel[i] = 1
    out = {"n_labels": n, "n_preds": len(preds), "matched": len(pairs)}
    for k in sorted({3, 5, n} - {0}):
        top = sum(rel[:k])
        out[f"p@{k}" if k != n else "p@R"] = top / k
        out[f"r@{k}" if k != n else "r@R"] = top / n if n else 0.0
    # average precision over the ranking
    hits, ap = 0, 0.0
    for i, r in enumerate(rel, 1):
        if r:
            hits += 1; ap += hits / i
    out["ap"] = ap / n if n else 0.0
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel[:5]))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(5, n)))
    out["ndcg@5"] = dcg / idcg if idcg else 0.0
    se, ee, setup, payoff, lens = [], [], 0, 0, []
    for i, j in pairs:
        p, l = preds[i], labels[j]
        se.append(abs(p["start"] - l["start"])); ee.append(abs(p["end"] - l["end"]))
        setup += p["start"] <= l["start"] + 1.0
        payoff += p["end"] >= l["end"] - 1.0
        lens.append(p["end"] - p["start"])
    m = len(pairs)
    out["start_err_s"] = sum(se) / m if m else None
    out["end_err_s"] = sum(ee) / m if m else None
    out["setup_included"] = setup / m if m else None
    out["payoff_included"] = payoff / m if m else None
    out["mean_len_s"] = sum(lens) / m if m else None
    return out


def aggregate(rows):
    """Mean of each numeric metric over cases (None values skipped); recall/precision also micro-averaged."""
    keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float))})
    agg = {}
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        agg[k] = sum(vals) / len(vals) if vals else None
    lab = sum(r["n_labels"] for r in rows)
    agg["micro_recall@R"] = sum(r.get("r@R", 0) * r["n_labels"] for r in rows) / lab if lab else None
    agg["cases"] = len(rows)
    return agg
