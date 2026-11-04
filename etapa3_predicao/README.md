# Stage 3 — trajectory prediction (the TCC's own code)

Package `trajpred/` and three commands. The Kalman filter is external: it runs after tracking, on the recorded trajectory
of each method, so differences in the error reflect the quality of the trajectories and not the predictor.

| Module | What it does |
|---|---|
| `trajpred/kalman.py` | constant-velocity Kalman filter (Q = 0.03 I, R = 0.1 I, P0 = I), persistent or damped with gamma = 0.95, extrapolation at H5, H10, H15 and H30 |
| `trajpred/matching.py` | CLEAR MOT and IDF1 evaluation per sequence; track-to-target correspondences `clear` (per frame) and `idf1` (global) |
| `trajpred/errors.py` | identity-aware prediction error: track k is compared with the target it was following when the prediction was made; normalisation by the target diagonal; common support |
| `trajpred/sot.py` | single-target prediction error and common support between methods |
| `trajpred/hybrid_mot.py` | hybrid architecture extended to multiple targets (Table 2 of the TCC) |

| Command | Input | Output |
|---|---|---|
| `predict_sot.py` | SIGE trajectories (`videoNN.txt`) and official annotations | error per video, H5 to H30, persistent and damped |
| `predict_mot.py` | MOTChallenge trajectories (`<seq>.txt`) and Track 3 `gt.txt` | MOTA, IDF1, identity switches and identity-aware error per sequence |
| `run_hybrid_mot.py` | Track 3 videos, `best.pt` weights and a DroneTrackingMOT clone | trajectories of the hybrid architecture and time per frame |

Settings are in [`../configs/`](../configs/). The tests in [`../tests/`](../tests/) use synthetic data; `predict_sot.py` and
`predict_mot.py` were checked against the files in `results/` (maximum difference of 2e-4 px for the single target and zero
for Track 3, on the sequences tested), and `run_hybrid_mot.py` reproduced the first 60 frames of the stored trajectory of
sequence UAV009 line by line.
