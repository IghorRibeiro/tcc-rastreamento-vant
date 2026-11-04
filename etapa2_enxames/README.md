# Stage 2 — swarms, infrared (Track 3 of the 4th Anti-UAV Challenge)

The code of this stage is the one of the **LAFusion 2026** paper (Springer CCIS, to appear), kept in
[`DroneTrackingMOT`](https://github.com/IghorRibeiro/DroneTrackingMOT) and pinned to tag **`v1.0-lafusion2026`**:

```bash
git clone --branch v1.0-lafusion2026 https://github.com/IghorRibeiro/DroneTrackingMOT
```

Nothing of it is copied here.

## What comes from there

| Item | Where |
|---|---|
| Baseline (YOLO26n + BoT-SORT) and the five other trackers | `configs/trackers/`, `src/tracking/` |
| Hard-example mining | `src/detector/` |
| Tiled inference (SAHI, 512 px crops) | `src/tracking/run_sahi_tracker.py` |
| Geman-McClure cost and NSA-Kalman (HIT-Tracker, gamma = 0.55) | `src/tracking/gm_cost.py`, `src/tracking/nsa_kalman.py` |
| Weights `best.pt` (original detector) and `best_hardmining.pt` | `weights/` |
| Root-cause analysis, paired tests and the paper figures | `src/analysis/`, `src/figures/` |

## What is specific to the TCC

- the hybrid architecture extended to multiple targets (one local tracker per target, Hungarian association with the
  Geman-McClure cost, track life cycle):
  [`etapa3_predicao/trajpred/hybrid_mot.py`](../etapa3_predicao/trajpred/hybrid_mot.py) and
  [`etapa3_predicao/run_hybrid_mot.py`](../etapa3_predicao/run_hybrid_mot.py), which use the Geman-McClure cost of that
  repository;
- the cost per target (regression of the time per frame, hypothesis H4) and the identity-aware prediction error on the
  trajectories of the baseline, the HIT-Tracker and the hybrid architecture.

## Data

Track 3 of the 4th Anti-UAV Challenge (DOI 10.5281/zenodo.15103888), not redistributed. See the "Dados" section of the
[main README](../README.md).
