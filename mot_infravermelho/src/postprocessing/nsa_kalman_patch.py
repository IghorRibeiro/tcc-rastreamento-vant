"""
NSA-Kalman (Noise Scale Adaptive) -- técnica já existente na
biblioteca (kalman_filter.py: project(..., confidence=...)) mas nunca
conectada no fluxo padrão do BoT-SORT/ByteTrack. Aqui reescrevemos
update()/re_activate() para propagar a confiança da detecção, fazendo
o filtro confiar menos em detecções de baixa confiança (aumentando o
ruído de medição R proporcionalmente).
"""

from ultralytics.trackers.bot_sort import BOTrack
from ultralytics.trackers.byte_tracker import TrackState

_original_update = BOTrack.update
_original_re_activate = BOTrack.re_activate


def _nsa_re_activate(self, new_track, frame_id, new_id: bool = False) -> None:
    if new_track.curr_feat is not None:
        self.update_features(new_track.curr_feat)

    self.mean, self.covariance = self.kalman_filter.update(
        self.mean, self.covariance, self.convert_coords(new_track.tlwh),
        confidence=new_track.score,
    )
    self.tracklet_len = 0
    self.state = TrackState.Tracked
    self.is_activated = True
    self.frame_id = frame_id
    if new_id:
        self.track_id = self.next_id()
    self.score = new_track.score
    self.cls = new_track.cls
    self.angle = new_track.angle
    self.idx = new_track.idx


def _nsa_update(self, new_track, frame_id) -> None:
    if new_track.curr_feat is not None:
        self.update_features(new_track.curr_feat)

    self.frame_id = frame_id
    self.tracklet_len += 1
    new_tlwh = new_track.tlwh
    self.mean, self.covariance = self.kalman_filter.update(
        self.mean, self.covariance, self.convert_coords(new_tlwh),
        confidence=new_track.score,
    )
    self.state = TrackState.Tracked
    self.is_activated = True
    self.score = new_track.score
    self.cls = new_track.cls
    self.angle = new_track.angle
    self.idx = new_track.idx


def patch_nsa_kalman():
    BOTrack.update = _nsa_update
    BOTrack.re_activate = _nsa_re_activate


def unpatch_nsa_kalman():
    BOTrack.update = _original_update
    BOTrack.re_activate = _original_re_activate