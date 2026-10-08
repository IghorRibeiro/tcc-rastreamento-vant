"""
Lost-Query Bank for BoT-SORT (runtime monkey patch).

Goal:
- When a track becomes LOST, store its last center position, timestamp (frame_id),
  and the average motion direction (unit vector) and speed (px/frame) computed
  from the last up-to-5 frame-to-frame displacements before loss.
- During association, encourage LOST tracks to reattach to reappearing detections
  whose initial movement direction is compatible with the stored query and whose
  positions are kinematically plausible (near the extrapolated position).

Implementation strategy (non-invasive, no library edits):
- Patch `BOTrack.update` and `BOTrack.re_activate` to record recent centers per
  track (small history) — done by layering wrappers over the CURRENT methods
  (so this composes with NSA-Kalman patch, etc.).
- Patch `BOTrack.mark_lost` (if present) to push a LostQuery entry into a
  module-level bank when a track transitions to LOST.
- Patch `BOTSORT.get_dists` to bias the distance matrix for pairs
  (lost_track, detection) that satisfy direction and kinematic gates.
  This effectively adds a specialized, direction-aware association path for LOST
  tracks without changing Ultralytics’ control flow.

Notes:
- We do NOT modify any Ultralytics files; only dynamic monkey patches.
- Compatible with the existing `flit_association` patch — we call the currently
  installed `get_dists` and then post-adjust distances for LOST tracks.
- Conservative gating: if conditions are not met, the base association behavior
  remains unchanged.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import numpy as np

# Ultralytics types
from ultralytics.trackers.byte_tracker import TrackState  # type: ignore
from ultralytics.trackers.bot_sort import BOTrack, BOTSORT  # type: ignore

# --------------------
# Globals (module state)
# --------------------

@dataclass
class LostQuery:
    track_id: int
    last_center: np.ndarray  # shape (2,)
    last_frame: int
    avg_dir: np.ndarray      # unit vector shape (2,)
    avg_speed: float         # px/frame
    ttl: int                 # remaining frames to keep alive


# Track ID -> LostQuery
_LOST_BANK: Dict[int, LostQuery] = {}

# Keep small per-track history to compute average direction pre-loss
# We attach these to BOTrack instances at runtime.
_POS_ATTR = "_lqb_pos_hist"        # Deque[Tuple[frame_id, (cx, cy)]]
_MAX_POS_HISTORY = 10

# Previous method handles for clean unpatch
_prev_botrack_update = None
_prev_botrack_reactivate = None
_prev_botrack_mark_lost = None
_prev_botsort_get_dists = None

# Tuning parameters
_DIR_MIN_SAMPLES = 2         # need at least 2 displacements (3 positions)
_DIR_AVG_WINDOW = 5          # last up-to-5 displacements
_COS_SIM_THRESH = 0.70       # minimum cosine similarity to consider a match
_BASE_GATE_PX = 30.0         # minimum spatial gate in pixels
_GATE_MULT = 2.0             # gate = max(BASE_GATE_PX, GATE_MULT * speed * dt)
_COST_BLEND = 0.6            # blend factor toward direction-based cost when gated
_TTL_FRAMES = 30             # how long to keep a lost query in the bank


def _append_pos_hist(track: BOTrack, frame_id: int, cx: float, cy: float) -> None:
    dq: Deque[Tuple[int, Tuple[float, float]]]
    if not hasattr(track, _POS_ATTR):
        dq = deque(maxlen=_MAX_POS_HISTORY)
        setattr(track, _POS_ATTR, dq)
    else:
        dq = getattr(track, _POS_ATTR)
    dq.append((int(frame_id), (float(cx), float(cy))))


def _compute_avg_dir_speed_from_hist(dq: Deque[Tuple[int, Tuple[float, float]]]) -> Optional[Tuple[np.ndarray, float]]:
    if dq is None or len(dq) < (_DIR_MIN_SAMPLES + 1):
        return None
    # Use last up-to-(_DIR_AVG_WINDOW+1) points to make up to _DIR_AVG_WINDOW displacements
    pts = list(dq)[-(_DIR_AVG_WINDOW + 1):]
    # Compute displacements between consecutive recorded positions (ignore frame gaps for avg dir)
    vecs = []
    dists = []
    for i in range(1, len(pts)):
        (_, (x0, y0)) = pts[i - 1]
        (_, (x1, y1)) = pts[i]
        dx, dy = x1 - x0, y1 - y0
        v = np.array([dx, dy], dtype=float)
        n = float(np.linalg.norm(v))
        if n > 1e-6:
            vecs.append(v / n)
            dists.append(n)
    if len(vecs) == 0:
        return None
    avg_dir = np.mean(vecs, axis=0)
    norm = float(np.linalg.norm(avg_dir))
    if norm <= 1e-6:
        return None
    avg_dir = avg_dir / norm
    avg_speed = float(np.mean(dists)) if len(dists) > 0 else 0.0
    return avg_dir.astype(float), avg_speed


def _wrap_botrack_update(prev_update):
    def _update(self: BOTrack, new_track, frame_id):
        # Delegate to previous (may be NSA-Kalman patched)
        out = prev_update(self, new_track, frame_id)
        try:
            # After state update, record current center
            # BOTrack.xywh is available; take center from xywh[:2]
            cx, cy = float(self.xywh[0]), float(self.xywh[1])
            _append_pos_hist(self, int(frame_id), cx, cy)
        except Exception:
            pass
        return out
    return _update


def _wrap_botrack_reactivate(prev_reactivate):
    def _reactivate(self: BOTrack, new_track, frame_id, new_id: bool = False):
        out = prev_reactivate(self, new_track, frame_id, new_id)
        try:
            cx, cy = float(self.xywh[0]), float(self.xywh[1])
            _append_pos_hist(self, int(frame_id), cx, cy)
            # Track is re-activated (no longer LOST) — remove any stale query
            _LOST_BANK.pop(int(self.track_id), None)
        except Exception:
            pass
        return out
    return _reactivate


def _wrap_botrack_mark_lost(prev_mark_lost):
    def _mark_lost(self: BOTrack):
        # Before changing state to LOST, compute and push a LostQuery
        try:
            dq: Deque[Tuple[int, Tuple[float, float]]] = getattr(self, _POS_ATTR, None)  # type: ignore
            if dq and len(dq) >= 2:
                avg = _compute_avg_dir_speed_from_hist(dq)
                if avg is not None:
                    avg_dir, avg_speed = avg
                    last_frame, (lx, ly) = dq[-1]
                    lq = LostQuery(
                        track_id=int(self.track_id),
                        last_center=np.array([lx, ly], dtype=float),
                        last_frame=int(last_frame),
                        avg_dir=avg_dir,
                        avg_speed=float(avg_speed),
                        ttl=_TTL_FRAMES,
                    )
                    _LOST_BANK[int(self.track_id)] = lq
        except Exception:
            pass
        # Delegate to original mark_lost
        return prev_mark_lost(self)
    return _mark_lost


def _direction_bias_get_dists(prev_get_dists):
    def _get_dists(self: BOTSORT, tracks, detections):
        # Start from the currently active distance metric (IoU/FLIT/etc.)
        dists = prev_get_dists(self, tracks, detections)
        try:
            if not (len(tracks) and len(detections)):
                return dists
            # Build centers for detections once
            det_centers = np.array([d.xywh[:2] for d in detections], dtype=float)

            for ti, t in enumerate(tracks):
                # We only bias when the track is LOST or has a pending LostQuery
                # Note: TrackState may be int Enum; compare by value
                is_lost = getattr(t, "state", None) == TrackState.Lost
                lq = _LOST_BANK.get(int(getattr(t, "track_id", -1))) if is_lost else None
                if not lq:
                    continue

                # Compute time since loss in frames (fallback to 1 if current frame unknown)
                cur_frame = int(getattr(t, "frame_id", lq.last_frame + 1))
                dt = max(1, cur_frame - lq.last_frame)

                # Predict expected position using constant-velocity along avg_dir
                expected = lq.last_center + lq.avg_dir * lq.avg_speed * dt
                gate = max(_BASE_GATE_PX, _GATE_MULT * lq.avg_speed * dt)

                # Vector from last center to each detection center
                v_after = det_centers - lq.last_center[None, :]
                v_norm = np.linalg.norm(v_after, axis=1) + 1e-6
                dir_after = v_after / v_norm[:, None]

                # Cosine similarity with pre-loss avg_dir
                sims = np.clip(dir_after @ lq.avg_dir.reshape(2, 1), -1.0, 1.0).reshape(-1)

                # Kinematic gating: distance to expected position
                kdist = np.linalg.norm(det_centers - expected[None, :], axis=1)
                gated = (sims >= _COS_SIM_THRESH) & (kdist <= gate)
                if not np.any(gated):
                    continue

                # Blend base distance with a direction-derived cost for gated detections
                # Direction cost: d_dir = 1 - sim (lower is better when sim is high)
                d_dir = 1.0 - sims
                # Apply only to gated indices
                for dj in np.where(gated)[0]:
                    base = float(dists[ti, dj])
                    # Conservative blend toward d_dir
                    new_cost = (1.0 - _COST_BLEND) * base + _COST_BLEND * float(d_dir[dj])
                    # Ensure cost stays within [0, 1]
                    new_cost = float(np.clip(new_cost, 0.0, 1.0))
                    dists[ti, dj] = min(base, new_cost)

                # Decrease TTL and drop if expired
                lq.ttl -= 1
                if lq.ttl <= 0:
                    _LOST_BANK.pop(lq.track_id, None)

            return dists
        except Exception:
            # Be robust to any unexpected issues; fallback to base distances
            return dists

    return _get_dists


def patch_lost_query_bank() -> None:
    """Apply the monkey patches for Lost-Query Bank behavior."""
    global _prev_botrack_update, _prev_botrack_reactivate, _prev_botrack_mark_lost, _prev_botsort_get_dists

    # Wrap BOTrack.update / re_activate to record positions (compose with other patches)
    if _prev_botrack_update is None:
        _prev_botrack_update = BOTrack.update
        BOTrack.update = _wrap_botrack_update(_prev_botrack_update)

    if _prev_botrack_reactivate is None:
        _prev_botrack_reactivate = BOTrack.re_activate
        BOTrack.re_activate = _wrap_botrack_reactivate(_prev_botrack_reactivate)

    # Wrap BOTrack.mark_lost if available
    if _prev_botrack_mark_lost is None and hasattr(BOTrack, "mark_lost"):
        _prev_botrack_mark_lost = BOTrack.mark_lost
        BOTrack.mark_lost = _wrap_botrack_mark_lost(_prev_botrack_mark_lost)

    # Wrap BOTSORT.get_dists to add direction-aware bias for LOST tracks
    if _prev_botsort_get_dists is None:
        _prev_botsort_get_dists = BOTSORT.get_dists
        BOTSORT.get_dists = _direction_bias_get_dists(_prev_botsort_get_dists)


def unpatch_lost_query_bank() -> None:
    """Revert all Lost-Query Bank monkey patches."""
    global _prev_botrack_update, _prev_botrack_reactivate, _prev_botrack_mark_lost, _prev_botsort_get_dists

    # Restore BOTrack methods
    if _prev_botrack_update is not None and BOTrack.update != _prev_botrack_update:
        BOTrack.update = _prev_botrack_update
    if _prev_botrack_reactivate is not None and BOTrack.re_activate != _prev_botrack_reactivate:
        BOTrack.re_activate = _prev_botrack_reactivate
    if _prev_botrack_mark_lost is not None and hasattr(BOTrack, "mark_lost") and BOTrack.mark_lost != _prev_botrack_mark_lost:
        BOTrack.mark_lost = _prev_botrack_mark_lost

    # Restore BOTSORT.get_dists
    if _prev_botsort_get_dists is not None and BOTSORT.get_dists != _prev_botsort_get_dists:
        BOTSORT.get_dists = _prev_botsort_get_dists

    # Clear bank state
    _LOST_BANK.clear()
