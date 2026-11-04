"""Hybrid architecture (periodic detection + local tracking) extended to multiple targets.

Single-target version (SIGE 2026), kept unchanged:
  * one local tracker per target (CSRT, KCF, MOSSE or MedianFlow), initialised with the detected
    box and updated at every frame; its box is rounded to integers, with w, h >= 1;
  * the detector runs on the first frame and every ``n_red`` frames (1-based frame index);
  * a detection needs confidence >= ``init_conf`` (0.5) to start a track and >= ``switch_conf``
    (0.7) to re-anchor a track that the local tracker still follows;
  * when the local tracker fails the track stops being reported;
  * with no active track the detector runs at every frame.

Adaptations required by multiple targets (TCC Table 2):
  * all detections of the frame are used, not only the most confident one;
  * detections are associated to tracks with the Hungarian algorithm and the Geman-McClure
    cost of the HIT-Tracker (``get_dists``), fused with the detection score;
  * birth, loss, reactivation and death of tracks follow the ``lifecycle`` settings.

The Kalman filter does not run inside the loop: it is applied afterwards on the recorded
trajectory (``trajpred.kalman``).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

ACTIVE, LOST, DEAD = "active", "lost", "dead"


class _Box:
    """Minimal object accepted by the Ultralytics ``matching`` helpers and by ``get_dists``."""

    __slots__ = ("xyxy", "xywh", "score", "angle", "curr_feat")

    def __init__(self, x, y, w, h, score=1.0):
        x, y, w, h = float(x), float(y), float(w), float(h)
        self.xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)
        self.xywh = np.array([x + w / 2.0, y + h / 2.0, w, h], dtype=np.float32)
        self.score = float(score)
        self.angle = None
        self.curr_feat = None


class Track:
    __slots__ = ("tid", "tracker", "box", "conf", "state", "lost_at", "last_confirmed", "born")

    def __init__(self, tid, tracker, box, conf, index):
        self.tid = tid
        self.tracker = tracker
        self.box = box
        self.conf = conf
        self.state = ACTIVE
        self.lost_at = None
        self.last_confirmed = index
        self.born = index


def make_local_tracker(name: str):
    """OpenCV local tracker by name; needs opencv-contrib-python."""
    import cv2

    factories = {
        "csrt": "TrackerCSRT_create",
        "kcf": "TrackerKCF_create",
        "mosse": "TrackerMOSSE_create",
        "medianflow": "TrackerMedianFlow_create",
    }
    factory = factories[name]
    if hasattr(cv2, factory):
        return getattr(cv2, factory)()
    legacy = getattr(cv2, "legacy", None)
    if legacy is not None and hasattr(legacy, factory):
        return getattr(legacy, factory)()
    raise RuntimeError(f"{name.upper()} is not available in OpenCV {cv2.__version__}; "
                       "install opencv-contrib-python.")


def _round_box(x1, y1, x2, y2, score):
    x1i, y1i = max(0, int(round(x1))), max(0, int(round(y1)))
    return (x1i, y1i, max(1, int(round(x2)) - x1i), max(1, int(round(y2)) - y1i), float(score))


def detections_from_yolo(detector, frame, device: str, conf: float) -> list:
    """All detections of a frame as (x, y, w, h, score) with integer boxes."""
    boxes = detector(frame, device=device, conf=conf, verbose=False)[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy().tolist()
    scores = boxes.conf.detach().cpu().numpy().tolist()
    return [_round_box(x1, y1, x2, y2, s) for (x1, y1, x2, y2), s in zip(xyxy, scores)]


def detections_from_sahi(build_sahi_detections, sahi_model, frame, slice_size: int, overlap: float,
                         conf: float) -> list:
    """Tiled-inference variant; ``build_sahi_detections`` is ``src.tracking.run_sahi_tracker`` of DroneTrackingMOT."""
    dets = build_sahi_detections(frame=frame, det_model=sahi_model, slice_size=slice_size, overlap=overlap,
                                 conf_thresh=conf)
    return [_round_box(x1, y1, x2, y2, s) for x1, y1, x2, y2, s, _cls in np.asarray(dets).tolist()]


class HybridMOT:
    """One step per frame: local trackers -> (detection + Hungarian) -> track life cycle."""

    def __init__(self, hcfg: dict, detect_fn, make_local_tracker_fn, get_dists, linear_assignment):
        self.n_red = int(hcfg["n_red"])
        self.local_name = str(hcfg["local_tracker"])
        self.init_conf = float(hcfg["init_conf"])
        self.switch_conf = float(hcfg["switch_conf"])
        assoc = hcfg["association"]
        self.match_thresh = float(assoc["match_thresh"])
        life = hcfg["lifecycle"]
        self.max_lost = int(life["max_lost_frames"])
        self.max_unconfirmed = int(life["max_unconfirmed_frames"])
        self.detect_when_empty = bool(life["detect_when_empty"])
        self.redetect_on_failure = bool(life["redetect_on_failure"])
        self.detect_fn = detect_fn
        self.make_local_tracker = make_local_tracker_fn
        self.get_dists = get_dists
        self.linear_assignment = linear_assignment
        # stand-in for the tracker instance expected by BOTSORT.get_dists
        self.fake_self = SimpleNamespace(
            proximity_thresh=float(assoc["proximity_thresh"]),
            args=SimpleNamespace(fuse_score=bool(assoc["fuse_score"]), with_reid=False),
            encoder=None,
        )
        self.tracks: list[Track] = []
        self.next_id = 1
        self.failed_last_frame = False

    def _new_local(self, frame, box):
        """Create and initialise a local tracker; None if OpenCV rejects the box."""
        import cv2

        tracker = self.make_local_tracker(self.local_name)
        try:
            result = tracker.init(frame, tuple(int(v) for v in box))
        except cv2.error:
            return None
        if result is False:
            return None
        return tracker

    def _update_local(self, track, frame):
        import cv2

        try:
            success, box = track.tracker.update(frame)
        except cv2.error:
            return False, None, True
        if not success:
            return False, None, False
        x, y, w, h = (int(round(v)) for v in box)
        return True, (x, y, max(1, w), max(1, h)), False

    def step(self, index: int, frame):
        """Process one frame (1-based ``index``); returns (outputs, stats, anchors)."""
        stats = {"detect": 0, "n_dets": 0, "births": 0, "reanchors": 0, "reactivations": 0,
                 "failures": 0, "exceptions": 0, "init_failures": 0, "deaths_lost": 0,
                 "deaths_unconfirmed": 0}
        anchors = []

        for track in self.tracks:
            if track.state != ACTIVE:
                continue
            ok, box, exception = self._update_local(track, frame)
            if exception:
                stats["exceptions"] += 1
            if ok:
                track.box = box
            else:
                track.state = LOST
                track.lost_at = index
                track.tracker = None
                stats["failures"] += 1

        n_active = sum(1 for t in self.tracks if t.state == ACTIVE)
        do_detect = (index == 1 or index % self.n_red == 0
                     or (self.detect_when_empty and n_active == 0)
                     or (self.redetect_on_failure and self.failed_last_frame))
        self.failed_last_frame = stats["failures"] > 0

        if do_detect:
            dets = self.detect_fn(frame)
            stats["detect"] = 1
            stats["n_dets"] = len(dets)
            candidates = [t for t in self.tracks if t.state == ACTIVE]
            candidates += [t for t in self.tracks
                           if t.state == LOST and index - t.lost_at <= self.max_lost]
            if candidates and dets:
                dists = self.get_dists(self.fake_self,
                                       [_Box(*t.box) for t in candidates],
                                       [_Box(*d[:4], score=d[4]) for d in dets])
                matches, _, unmatched_dets = self.linear_assignment(dists, thresh=self.match_thresh)
                matches = [(int(m[0]), int(m[1])) for m in matches]
                unmatched_dets = [int(j) for j in unmatched_dets]
            else:
                matches, unmatched_dets = [], list(range(len(dets)))

            for ti, di in matches:
                track = candidates[ti]
                x, y, w, h, score = dets[di]
                if track.state == ACTIVE:
                    track.last_confirmed = index
                    if score >= self.switch_conf:                       # re-anchor
                        tracker = self._new_local(frame, (x, y, w, h))
                        if tracker is None:
                            stats["init_failures"] += 1
                        else:
                            track.tracker = tracker
                            track.box = (x, y, w, h)
                            track.conf = score
                            stats["reanchors"] += 1
                            anchors.append((track.tid, index, x, y, w, h, score, "reanchor"))
                elif score >= self.init_conf:                           # reactivate a lost track
                    tracker = self._new_local(frame, (x, y, w, h))
                    if tracker is None:
                        stats["init_failures"] += 1
                    else:
                        track.tracker = tracker
                        track.state = ACTIVE
                        track.lost_at = None
                        track.box = (x, y, w, h)
                        track.conf = score
                        track.last_confirmed = index
                        stats["reactivations"] += 1
                        anchors.append((track.tid, index, x, y, w, h, score, "reactivate"))

            for di in unmatched_dets:                                   # birth
                x, y, w, h, score = dets[di]
                if score < self.init_conf:
                    continue
                tracker = self._new_local(frame, (x, y, w, h))
                if tracker is None:
                    stats["init_failures"] += 1
                    continue
                track = Track(self.next_id, tracker, (x, y, w, h), score, index)
                self.next_id += 1
                self.tracks.append(track)
                stats["births"] += 1
                anchors.append((track.tid, index, x, y, w, h, score, "birth"))

            # death by lack of confirmation (only evaluated on detection frames)
            for track in self.tracks:
                if track.state == ACTIVE and index - track.last_confirmed >= self.max_unconfirmed:
                    track.state = DEAD
                    track.tracker = None
                    stats["deaths_unconfirmed"] += 1

        for track in self.tracks:
            if track.state == LOST and index - track.lost_at > self.max_lost:
                track.state = DEAD
                stats["deaths_lost"] += 1
        self.tracks = [t for t in self.tracks if t.state != DEAD]

        outputs = [(t.tid, t.box[0], t.box[1], t.box[2], t.box[3], t.conf)
                   for t in self.tracks if t.state == ACTIVE]
        stats["n_active"] = len(outputs)
        stats["n_lost"] = sum(1 for t in self.tracks if t.state == LOST)
        return outputs, stats, anchors
