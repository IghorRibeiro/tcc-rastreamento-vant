"""
Adaptive per-frame adjustment of ByteTrack/BOT-SORT "track_high_thresh" based on
current-frame detection confidences.

Rule (applied inside BYTETracker._split_detections for every frame):
- mean_conf < 0.40 → track_high_thresh -= 0.05 (min 0.15)
- mean_conf > 0.70 → track_high_thresh += 0.05 (max 0.35)
- otherwise → use base value 0.25

Implementation notes:
- We monkey‑patch `ultralytics.trackers.byte_tracker.BYTETracker._split_detections`.
- We compute the average confidence from the `scores` input of that method.
- We set `self.args.track_high_thresh` to the adjusted value just before delegating
  to the original method, and restore it immediately afterward to avoid side effects
  across frames or instances.
- Designed to work with Ultralytics v8.4.105; signature handling is defensive
  (`*args`, `**kwargs`) so it won’t break if minor changes occur.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

# Import after type ignores to avoid static complaints if ultralytics isn’t present at import-time
from ultralytics.trackers.byte_tracker import BYTETracker  # type: ignore

# Store original method handle
_original_split_detections = BYTETracker._split_detections  # type: ignore[attr-defined]


def _extract_scores_from_args(*args: Any, **kwargs: Any) -> Optional[np.ndarray]:
    """Attempt to extract a 1D array of detection confidences from the provided
    positional/keyword args used by `BYTETracker._split_detections`.

    Ultralytics (8.4.x) typically calls `_split_detections(self, dets, scores, cls, ...)`.
    We try, in order:
      1) A keyword argument named "scores".
      2) The second positional argument (index 1), if array-like.
      3) As a fallback, if the first positional arg looks like Nx6 or Nx5 dets with
         score at column 4, try to read that column.
    Returns None if no usable scores can be found.
    """
    # 1) Keyword
    if "scores" in kwargs and kwargs["scores"] is not None:
        try:
            arr = np.asarray(kwargs["scores"])  # torch.Tensor works too
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr.reshape(-1)
            if arr.ndim == 1 and arr.size > 0:
                return arr.astype(float)
        except Exception:
            pass

    # 2) Second positional arg
    if len(args) >= 2 and args[1] is not None:
        try:
            arr = np.asarray(args[1])
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr.reshape(-1)
            if arr.ndim == 1 and arr.size > 0:
                return arr.astype(float)
        except Exception:
            pass

    # 3) Fallback: dets matrix where score might be column 4 (0-based)
    if len(args) >= 1 and args[0] is not None:
        try:
            dets = np.asarray(args[0])
            if dets.ndim == 2 and dets.shape[0] > 0 and dets.shape[1] >= 5:
                scores = dets[:, 4]
                return np.asarray(scores, dtype=float)
        except Exception:
            pass

    return None


def _adaptive_split_detections(self: BYTETracker, *args: Any, **kwargs: Any):  # noqa: D401
    """Wrapper around the original `_split_detections` that adaptively adjusts
    `self.args.track_high_thresh` per frame using the current detections' confidences.
    The threshold is restored after the original call returns.
    """
    # Base value as specified by the task description
    base_thresh = 0.25

    # Try to get current-frame detection confidences
    scores = _extract_scores_from_args(*args, **kwargs)
    adj_thresh = base_thresh

    if scores is not None and scores.size > 0:
        mean_conf = float(np.mean(scores))
        if mean_conf < 0.40:
            adj_thresh = max(0.15, base_thresh - 0.05)
        elif mean_conf > 0.70:
            adj_thresh = min(0.35, base_thresh + 0.05)
        else:
            adj_thresh = base_thresh

    # Temporarily override the tracker high threshold
    prev_thresh = getattr(self.args, "track_high_thresh", base_thresh)
    prev_high_attr = getattr(self, "high_thresh", prev_thresh)
    try:
        # Update both the args value and any cached attribute used internally
        self.args.track_high_thresh = adj_thresh
        try:
            self.high_thresh = adj_thresh
        except Exception:
            pass
        return _original_split_detections(self, *args, **kwargs)
    finally:
        # Restore previous values to keep side-effects local to this frame
        try:
            self.args.track_high_thresh = prev_thresh
        except Exception:
            pass
        try:
            self.high_thresh = prev_high_attr
        except Exception:
            pass


def patch_adaptive_track_high_thresh() -> None:
    """Apply the monkey patch for adaptive track_high_thresh per frame."""
    # Avoid double-patching by checking attribute equality
    if BYTETracker._split_detections is not _adaptive_split_detections:  # type: ignore[attr-defined]
        BYTETracker._split_detections = _adaptive_split_detections  # type: ignore[attr-defined]


def unpatch_adaptive_track_high_thresh() -> None:
    """Revert the monkey patch, restoring the original method."""
    if BYTETracker._split_detections is _adaptive_split_detections:  # type: ignore[attr-defined]
        BYTETracker._split_detections = _original_split_detections  # type: ignore[attr-defined]
