"""Arquitetura hibrida do SIGE levada ao MOT (Experimento B).

Origem (SIGE): publicar_github/.../src/04_rastreadores.py::run_hibrido.
Mantido do SIGE, sem mudanca:
  * rastreador local por make_local_tracker() do proprio SIGE (CSRT, KCF, MOSSE,
    MedianFlow), inicializado com a caixa da deteccao e atualizado a cada quadro;
  * caixa do rastreador local arredondada para inteiros, w e h >= 1;
  * deteccao na inicializacao e a cada N_red quadros (index % N_red == 0,
    index 1-based do quadro), com o detector no quadro inteiro;
  * arredondamento da caixa detectada identico a detect_best();
  * confianca >= init_conf (0,5) para iniciar; >= switch_conf (0,7) para
    reancorar uma trilha que o rastreador local ainda segue;
  * falha do rastreador local -> a trilha deixa de ser reportada (SIGE: tracker = None);
  * sem nenhuma trilha ativa -> deteccao em todo quadro (SIGE: tracker is None).
Adaptacoes exigidas pelo MOT (ASSUMPTIONS.md, B-01 a B-09):
  * todas as deteccoes do quadro (nao so a de maior confianca);
  * associacao deteccao-trilha pelo hungaro (lapjv do Ultralytics) com o custo
    Geman-McClure do HIT-Tracker (make_flit_get_dists do LAFusion) + fuse_score;
  * nascimento, perda, reativacao e morte de trilhas configuraveis.
O filtro de Kalman NAO roda dentro do laco: como no SIGE (docstring do
04_rastreadores.py), ele e aplicado depois sobre a trajetoria gravada.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

ACTIVE, LOST, DEAD = "active", "lost", "dead"


class _Box:
    """Objeto minimo aceito por matching.iou_distance / fuse_score / get_dists do Ultralytics."""

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


def detections_from_yolo(detector, frame, device: str, conf: float) -> list:
    """Todas as deteccoes, com o arredondamento de SIGE 04_rastreadores.py::detect_best."""
    results = detector(frame, device=device, conf=conf, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy().tolist()
    confidences = boxes.conf.detach().cpu().numpy().tolist()
    out = []
    for (x1, y1, x2, y2), score in zip(xyxy, confidences):
        x1i, y1i = max(0, int(round(x1))), max(0, int(round(y1)))
        w, h = max(1, int(round(x2)) - x1i), max(1, int(round(y2)) - y1i)
        out.append((x1i, y1i, w, h, float(score)))
    return out


def detections_from_sahi(build_sahi_detections, sahi_model, frame, slice_size: int, overlap: float,
                         conf: float) -> list:
    """Variante SAHI: build_sahi_detections do runner do HIT-Tracker + mesmo arredondamento."""
    dets = build_sahi_detections(frame=frame, det_model=sahi_model, slice_h=slice_size, slice_w=slice_size,
                                 overlap=overlap, conf_thresh=conf)
    out = []
    for x1, y1, x2, y2, score, _cls in np.asarray(dets).tolist():
        x1i, y1i = max(0, int(round(x1))), max(0, int(round(y1)))
        w, h = max(1, int(round(x2)) - x1i), max(1, int(round(y2)) - y1i)
        out.append((x1i, y1i, w, h, float(score)))
    return out


class HybridMOT:
    """Um passo por quadro: rastreador local -> (deteccao + hungaro) -> ciclo de vida."""

    def __init__(self, hcfg: dict, detect_fn, make_local_tracker, get_dists, linear_assignment):
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
        self.make_local_tracker = make_local_tracker
        self.get_dists = get_dists
        self.linear_assignment = linear_assignment
        # 'self' falso para BOTSORT.get_dists (robust_flit_association.make_flit_get_dists)
        self.fake_self = SimpleNamespace(
            proximity_thresh=float(assoc["proximity_thresh"]),
            args=SimpleNamespace(fuse_score=bool(assoc["fuse_score"]), with_reid=False),
            encoder=None,
        )
        self.tracks: list[Track] = []
        self.next_id = 1
        self.failed_last_frame = False

    # ------------------------------------------------------------ local tracker
    def _new_local(self, frame, box):
        """Cria e inicializa um rastreador local; None se o OpenCV recusar a caixa."""
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

    # ------------------------------------------------------------------- passo
    def step(self, index: int, frame):
        stats = {"detect": 0, "n_dets": 0, "births": 0, "reanchors": 0, "reactivations": 0,
                 "failures": 0, "exceptions": 0, "init_failures": 0, "deaths_lost": 0,
                 "deaths_unconfirmed": 0}
        anchors = []

        # 1) rastreador local em todas as trilhas ativas (SIGE: tracker.update)
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

        # 2) deteccao global + hungaro com custo GM
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
                    if score >= self.switch_conf:                       # SIGE: reancoragem
                        tracker = self._new_local(frame, (x, y, w, h))
                        if tracker is None:
                            stats["init_failures"] += 1
                        else:
                            track.tracker = tracker
                            track.box = (x, y, w, h)
                            track.conf = score
                            stats["reanchors"] += 1
                            anchors.append((track.tid, index, x, y, w, h, score, "reanchor"))
                elif score >= self.init_conf:                           # reativacao de trilha perdida
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

            for di in unmatched_dets:                                   # nascimento
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

            # morte por falta de confirmacao (so avaliada em quadros com deteccao)
            for track in self.tracks:
                if track.state == ACTIVE and index - track.last_confirmed >= self.max_unconfirmed:
                    track.state = DEAD
                    track.tracker = None
                    stats["deaths_unconfirmed"] += 1

        # 3) morte de trilhas perdidas ha mais de max_lost quadros
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
