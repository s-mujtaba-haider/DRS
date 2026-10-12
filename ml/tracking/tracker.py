"""Single-ball tracker for tiny, fast-moving objects.

Online pass: gated (Mahalanobis + max speed) Hungarian association of
candidates to multiple hypotheses, so clutter from the motion detector can
spawn short-lived tracks without hijacking the ball track.

Offline pass: each confirmed track is re-filtered over its full frame span with
outlier rejection and RTS smoothing; fragments that are kinematically
consistent are merged. The ball track is chosen by a transparent score.

Observations and estimates are kept separate in the output: ``observed`` points
are raw detection centres, ``filtered`` are smoothed estimates at observed
frames, ``interpolated`` are smoothed estimates at frames without a detection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from ml.common.schemas import BallTrack, Detection, TrackingResult, TrackPoint, TrackQuality
from ml.tracking.kalman import KalmanCA2D, mahalanobis2, rts_smooth

CHI2_2DOF_99 = 9.21
CHI2_2DOF_999 = 13.82


@dataclass
class TrackerConfig:
    meas_sigma_px: float = 3.0
    init_vel_sigma: float = 1500.0  # px/s; a fast ball may cross a 1080p frame in ~0.5 s
    init_acc_sigma: float = 3000.0
    jerk_sigma: float = 2.0e5
    gate_chi2: float = CHI2_2DOF_99
    max_speed_px_s: float = 6000.0
    max_consecutive_misses: int = 5
    min_hits_to_confirm: int = 3
    # Optional hysteresis: only detections >= spawn_min_confidence may start a track; weaker ones
    # (>= extend_min_confidence) may only extend an established track. Off by default: on real
    # broadcast clips spawn=0.25 lost low-confidence ball tracks (2 -> 1 of 5 reliable; docs/EVALUATION.md).
    spawn_min_confidence: float = 0.05
    extend_min_confidence: float = 0.05
    max_tracks: int = 40
    min_displacement_px: float = 20.0
    merge_max_gap_frames: int = 6
    outlier_chi2: float = CHI2_2DOF_999
    # Minimum consistency for a track to be labelled "reliable" (necessary, not sufficient).
    reliable_min_observed: int = 8
    reliable_min_observed_ratio: float = 0.5
    reliable_max_mean_innovation_px: float = 12.0


@dataclass
class _Hyp:
    tid: int
    kf: KalmanCA2D
    obs: dict[int, Detection] = field(default_factory=dict)
    last_t: float = 0.0
    hits: int = 1
    misses: int = 0

    @property
    def confirmed_like(self) -> bool:
        return self.hits >= 2


class BallTracker:
    def __init__(self, config: TrackerConfig | None = None):
        self.cfg = config or TrackerConfig()

    def _R(self, det: Detection) -> np.ndarray:
        s = max(self.cfg.meas_sigma_px, det.uncertainty_px or 0.0)
        return np.eye(2) * s * s

    # ------------------------------------------------------------------ online
    def _associate(self, frames: list[tuple[int, float]], dets_by_frame: dict[int, list[Detection]],
                   breaks: set[int] | None = None):
        cfg = self.cfg
        active: list[_Hyp] = []
        finished: list[_Hyp] = []
        next_id = 0
        for fidx, t_ms in frames:
            t = t_ms / 1000.0
            dets = dets_by_frame.get(fidx, [])
            if breaks and fidx in breaks:  # camera cut: never associate across it
                finished.extend(active)
                active = []
            for h in active:
                h.kf.predict(max(t - h.last_t, 1e-4))
                h.last_t = t
            cost = np.full((len(active), len(dets)), 1e6)
            for i, h in enumerate(active):
                for j, d in enumerate(dets):
                    if d.confidence < cfg.extend_min_confidence:
                        continue
                    if d.confidence < cfg.spawn_min_confidence and h.hits < 2:
                        continue  # weak detections only extend established tracks
                    z = np.asarray(d.center)
                    d2 = mahalanobis2(h.kf, z, self._R(d))
                    if d2 > cfg.gate_chi2 and h.hits > 1:
                        continue
                    last = h.obs[max(h.obs)]
                    dt = max((t_ms - last.timestamp_ms) / 1000.0, 1e-4)
                    jump = math.dist(last.center, d.center)
                    if jump / dt > cfg.max_speed_px_s:
                        continue
                    # Prefer confident detections and established tracks.
                    cost[i, j] = d2 - 2.0 * d.confidence - 0.2 * min(h.hits, 10)
            matched_t, matched_d = set(), set()
            if active and dets:
                rows, cols = linear_sum_assignment(cost)
                for r, c in zip(rows, cols):
                    if cost[r, c] >= 1e5:
                        continue
                    h, d = active[r], dets[c]
                    h.kf.update(np.asarray(d.center), self._R(d))
                    h.obs[fidx] = d
                    h.hits += 1
                    h.misses = 0
                    matched_t.add(r)
                    matched_d.add(c)
            still: list[_Hyp] = []
            for i, h in enumerate(active):
                if i not in matched_t:
                    h.misses += 1
                limit = cfg.max_consecutive_misses if h.confirmed_like else 1
                (finished if h.misses > limit else still).append(h)
            active = still
            for j, d in enumerate(dets):
                if j in matched_d or d.confidence < cfg.spawn_min_confidence:
                    continue
                if len(active) >= cfg.max_tracks:
                    break
                kf = KalmanCA2D.init(
                    np.asarray(d.center), cfg.meas_sigma_px, cfg.init_vel_sigma, cfg.init_acc_sigma, cfg.jerk_sigma
                )
                active.append(_Hyp(next_id, kf, {fidx: d}, t))
                next_id += 1
        finished.extend(active)
        return [h for h in finished if h.hits >= self.cfg.min_hits_to_confirm]

    # ----------------------------------------------------------------- offline
    def smooth(self, obs: dict[int, Detection], frames: list[tuple[int, float]]):
        """Re-filter an observation set over its frame span; returns (points, stats)."""
        cfg = self.cfg
        f0, f1 = min(obs), max(obs)
        span = [(f, t) for f, t in frames if f0 <= f <= f1]
        first = obs[f0]
        kf = KalmanCA2D.init(np.asarray(first.center), cfg.meas_sigma_px, cfg.init_vel_sigma, cfg.init_acc_sigma, cfg.jerk_sigma)
        xs_f, Ps_f, xs_p, Ps_p, Fs, used = [], [], [], [], [], []
        innov, outliers = [], 0
        last_t = span[0][1] / 1000.0
        for k, (f, t_ms) in enumerate(span):
            t = t_ms / 1000.0
            if k == 0:
                xp, Pp, F = kf.x.copy(), kf.P.copy(), np.eye(6)
            else:
                xp, Pp, F = kf.predict(max(t - last_t, 1e-4))
            last_t = t
            d = obs.get(f)
            accepted = None
            if d is not None and k > 0:
                z, R = np.asarray(d.center), self._R(d)
                d2 = mahalanobis2(kf, z, R)
                # Outlier test only once velocity is reasonably established.
                if k >= 3 and d2 > cfg.outlier_chi2:
                    outliers += 1
                else:
                    y, _ = kf.innovation(z, R)
                    innov.append(float(np.hypot(*y)))
                    kf.update(z, R)
                    accepted = d
            elif d is not None:
                accepted = d
            xs_f.append(kf.x.copy())
            Ps_f.append(kf.P.copy())
            xs_p.append(xp)
            Ps_p.append(Pp)
            Fs.append(F)
            used.append(accepted)
        xs, Ps = rts_smooth(xs_f, Ps_f, xs_p, Ps_p, Fs)
        points: list[TrackPoint] = []
        longest_gap = gap = 0
        for (f, t_ms), x, P, d in zip(span, xs, Ps, used):
            sx, sy = float(math.sqrt(max(P[0, 0], 0))), float(math.sqrt(max(P[1, 1], 0)))
            if d is not None:
                gap = 0
                points.append(
                    TrackPoint(frame_index=f, timestamp_ms=t_ms, kind="observed", x=d.center[0], y=d.center[1],
                               sigma_x=d.uncertainty_px, sigma_y=d.uncertainty_px, confidence=d.confidence,
                               detection_source=d.source)
                )
                points.append(TrackPoint(frame_index=f, timestamp_ms=t_ms, kind="filtered", x=float(x[0]), y=float(x[1]),
                                         sigma_x=sx, sigma_y=sy))
            else:
                gap += 1
                longest_gap = max(longest_gap, gap)
                points.append(TrackPoint(frame_index=f, timestamp_ms=t_ms, kind="interpolated", x=float(x[0]), y=float(x[1]),
                                         sigma_x=sx, sigma_y=sy))
        n_obs = sum(1 for d in used if d is not None)
        confs = [d.confidence for d in used if d is not None]
        quality = TrackQuality(
            num_frames_spanned=len(span),
            num_observed=n_obs,
            num_missing=len(span) - n_obs,
            observed_ratio=n_obs / len(span),
            longest_gap=longest_gap,
            mean_innovation_px=float(np.mean(innov)) if innov else None,
            mean_confidence=float(np.mean(confs)) if confs else None,
        )
        end_state = (xs[-1], Ps[-1], span[-1][1])
        return points, quality, outliers, end_state

    def _merge_fragments(self, hyps: list[_Hyp], frames: list[tuple[int, float]],
                         breaks: set[int] | None = None) -> list[tuple[dict[int, Detection], int]]:
        cfg = self.cfg
        breaks = breaks or set()
        groups = [(dict(h.obs), 0) for h in sorted(hyps, key=lambda h: min(h.obs))]
        frame_pos = {f: i for i, (f, _) in enumerate(frames)}
        merged = True
        while merged:
            merged = False
            for i in range(len(groups)):
                a, na = groups[i]
                _, _, _, (xa, Pa, ta) = self.smooth(a, frames)
                for j in range(len(groups)):
                    if i == j:
                        continue
                    b, nb = groups[j]
                    gap = frame_pos[min(b)] - frame_pos[max(a)]
                    if not (1 <= gap <= cfg.merge_max_gap_frames) or len(a) < 3:
                        continue
                    if any(max(a) < f <= min(b) for f in breaks):
                        continue
                    first_b = b[min(b)]
                    dt = (first_b.timestamp_ms - ta) / 1000.0
                    kf = KalmanCA2D(xa.copy(), Pa.copy(), cfg.jerk_sigma)
                    kf.predict(max(dt, 1e-4))
                    if mahalanobis2(kf, np.asarray(first_b.center), self._R(first_b)) <= cfg.gate_chi2:
                        groups[i] = ({**a, **b}, na + nb + 1)
                        groups.pop(j)
                        merged = True
                        break
                if merged:
                    break
        return groups

    def run(self, frames: list[tuple[int, float]], detections: list[Detection],
            shot_breaks: set[int] | None = None) -> TrackingResult:
        """frames: every processed (frame_index, timestamp_ms) in order.
        shot_breaks: frame indices that start a new camera shot (no association across them)."""
        cfg = self.cfg
        warnings: list[str] = []
        by_frame: dict[int, list[Detection]] = {}
        for d in detections:
            by_frame.setdefault(d.frame_index, []).append(d)
        if not frames:
            return TrackingResult(tracks=[], primary_track_id=None, num_frames=0, num_frames_with_candidates=0,
                                  num_tracks_total=0, warnings=["No frames processed"])
        hyps = self._associate(frames, by_frame, shot_breaks)
        groups = self._merge_fragments(hyps, frames, shot_breaks) if hyps else []
        tracks: list[BallTrack] = []
        for tid, (obs, nmerged) in enumerate(groups):
            points, quality, outliers, _ = self.smooth(obs, frames)
            quality.fragments_merged = nmerged
            issues = []
            if quality.num_observed < cfg.reliable_min_observed:
                issues.append(f"only {quality.num_observed} observations")
            if quality.observed_ratio < cfg.reliable_min_observed_ratio:
                issues.append(f"observed in {quality.observed_ratio:.0%} of spanned frames")
            if quality.mean_innovation_px is not None and quality.mean_innovation_px > cfg.reliable_max_mean_innovation_px:
                issues.append(f"erratic motion (mean innovation {quality.mean_innovation_px:.0f}px)")
            quality.reliable, quality.reliability_issues = not issues, issues
            observed = [p for p in points if p.kind == "observed"]
            disp = math.dist((observed[0].x, observed[0].y), (observed[-1].x, observed[-1].y)) if observed else 0.0
            if disp < cfg.min_displacement_px:
                continue  # static clutter (logos, stumps, fielders standing still)
            # Transparent ranking heuristic: long, dense, confident, smooth tracks win.
            smooth_pen = 1.0 / (1.0 + (quality.mean_innovation_px or 0.0) / 5.0)
            score = quality.num_observed * quality.observed_ratio * (quality.mean_confidence or 0.0) * smooth_pen
            if outliers:
                warnings.append(f"Track {tid}: {outliers} observation(s) rejected as outliers")
            tracks.append(BallTrack(track_id=tid, points=points, quality=quality, score=float(score)))
        tracks.sort(key=lambda tr: -tr.score)
        primary = None
        # Prefer tracks that pass the consistency checks; fall back to the best-scoring one.
        tracks.sort(key=lambda tr: (not tr.quality.reliable, -tr.score))
        if tracks:
            tracks[0].is_primary = True
            primary = tracks[0].track_id
            if not tracks[0].quality.reliable:
                warnings.append("Primary track is UNRELIABLE (" + "; ".join(tracks[0].quality.reliability_issues)
                                + "): it is probably not the ball. Use a trained detector or manual annotations.")
            if len(tracks) > 1 and tracks[1].score > 0.7 * tracks[0].score:
                warnings.append("Ambiguous ball track: a second track scored within 30% of the primary track")
        else:
            warnings.append("No moving ball track found")
        frag_note = None
        if tracks and tracks[0].quality.fragments_merged:
            frag_note = f"Primary track assembled from {tracks[0].quality.fragments_merged + 1} fragments"
        return TrackingResult(
            tracks=tracks,
            primary_track_id=primary,
            num_frames=len(frames),
            num_frames_with_candidates=sum(1 for f, _ in frames if by_frame.get(f)),
            num_tracks_total=len(groups),
            fragmentation_note=frag_note,
            warnings=warnings,
        )
