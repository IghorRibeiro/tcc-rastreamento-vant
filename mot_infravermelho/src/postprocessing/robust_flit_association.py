import numpy as np
from ultralytics.trackers.utils import matching

_original_get_dists = None


def make_flit_get_dists(gamma: float = 0.25, frame_w: int = 640, frame_h: int = 512):
    frame_diag = np.sqrt(frame_w ** 2 + frame_h ** 2)

    def get_dists(self, tracks, detections):
        dists = matching.iou_distance(tracks, detections)
        dists_mask = dists > (1 - self.proximity_thresh)

        if len(tracks) and len(detections):
            track_centers = np.array([t.xywh[:2] for t in tracks])
            det_centers = np.array([d.xywh[:2] for d in detections])
            diff = track_centers[:, None, :] - det_centers[None, :, :]
            l2 = np.sqrt((diff ** 2).sum(axis=2))
            l2_norm = np.clip(l2 / frame_diag, 0.0, 1.0)
            sigma = 0.15
            robust_l2 = (l2_norm ** 2) / (l2_norm ** 2 + sigma ** 2)
            dists = (1 - gamma) * dists + gamma * robust_l2

        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)

        if self.args.with_reid and self.encoder is not None:
            emb_dists = matching.embedding_distance(tracks, detections) / 2.0
            emb_dists[emb_dists > (1 - self.appearance_thresh)] = 1.0
            emb_dists[dists_mask] = 1.0
            dists = np.minimum(dists, emb_dists)

        return dists

    return get_dists


def patch_flit(gamma: float = 0.25, frame_w: int = 640, frame_h: int = 512):
    global _original_get_dists
    from ultralytics.trackers.bot_sort import BOTSORT

    if _original_get_dists is None:
        _original_get_dists = BOTSORT.get_dists
    BOTSORT.get_dists = make_flit_get_dists(gamma, frame_w, frame_h)


def unpatch_flit():
    from ultralytics.trackers.bot_sort import BOTSORT

    if _original_get_dists is not None:
        BOTSORT.get_dists = _original_get_dists