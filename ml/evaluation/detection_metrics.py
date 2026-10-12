"""Detection metrics for (tiny) ball detection.

Matching uses centre distance as well as IoU: for objects a few pixels wide,
IoU is extremely sensitive to one-pixel box errors, so a prediction counts as a
true positive when its centre lies within ``max(match_px, 0.5 * gt_size)`` of an
unmatched ground-truth centre. AP is the area under the interpolated
precision-recall curve (all-point interpolation).
"""

from __future__ import annotations

import math

import numpy as np


def match_frame(preds: list[tuple[float, float, float, float, float]], gts: list[tuple[float, float, float, float]],
                match_px: float = 6.0):
    """preds: (x1,y1,x2,y2,score); gts: (x1,y1,x2,y2). Returns [(score, is_tp, centre_err_px)], n_gt."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i][4])
    used = set()
    out = []
    for i in order:
        px = (preds[i][0] + preds[i][2]) / 2
        py = (preds[i][1] + preds[i][3]) / 2
        best, best_d = None, math.inf
        for j, g in enumerate(gts):
            if j in used:
                continue
            gx, gy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
            size = max(g[2] - g[0], g[3] - g[1])
            d = math.hypot(px - gx, py - gy)
            if d <= max(match_px, 0.5 * size) and d < best_d:
                best, best_d = j, d
        if best is not None:
            used.add(best)
            out.append((preds[i][4], True, best_d))
        else:
            out.append((preds[i][4], False, None))
    return out, len(gts)


def summarize(per_frame: list[tuple[list, int]], score_threshold: float = 0.25, gt_sizes: list[float] | None = None) -> dict:
    allm = [m for ms, _ in per_frame for m in ms]
    n_gt = sum(n for _, n in per_frame)
    if n_gt == 0:
        return {"num_gt": 0, "note": "No ground-truth boxes: metrics undefined"}
    allm.sort(key=lambda m: -m[0])
    tps = np.cumsum([m[1] for m in allm]) if allm else np.array([0])
    fps = np.cumsum([not m[1] for m in allm]) if allm else np.array([0])
    rec = tps / n_gt
    prec = tps / np.maximum(tps + fps, 1)
    # all-point interpolated AP
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[1], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    sel = [m for m in allm if m[0] >= score_threshold]
    tp = sum(1 for m in sel if m[1])
    fp = len(sel) - tp
    p = tp / max(tp + fp, 1)
    r = tp / n_gt
    errs = [m[2] for m in sel if m[1]]
    out = {
        "num_gt": n_gt, "num_predictions": len(allm), "score_threshold": score_threshold,
        "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9), "ap_centre_match": ap,
        "true_positives": tp, "false_positives": fp, "false_negatives": n_gt - tp,
        "centre_error_px": {"median": float(np.median(errs)), "mean": float(np.mean(errs)), "p90": float(np.percentile(errs, 90))} if errs else None,
    }
    if gt_sizes:
        a = np.array(gt_sizes)
        out["gt_size_px"] = {"median": float(np.median(a)), "frac_lt_16px": float(np.mean(a < 16))}
    return out
