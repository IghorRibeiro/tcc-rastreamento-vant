# Negative results

Section 5.1 of the TCC states that the negative results "delimit directions that do not need to be repeated". They are
listed here with the numbers of the text, followed by the code that left the main tree of this repository.

## Reported in the TCC

| Direction tested | Configuration | Result | TCC reference |
|---|---|---|---|
| Inference-time preprocessing: MOG2 background subtraction | detector fine-tuned on the original images; technique applied at inference only (Track 3) | MOTA −168.7% and IDF1 16.0% (baseline: 76.0% and 71.2%) | Table 8 |
| Exponential moving average | same | MOTA 2.3%, IDF1 38.0%, 4,823 identity switches | Table 8 |
| Dense optical-flow masks | same | MOTA 22.5%, IDF1 26.4%, 5,180 identity switches | Table 8 |
| Image enhancement before detection: CLAHE, gamma correction, unsharp masking | worst-performing sequences | mAP@0.5 from 0.547 to 0.529 (−3.3%) | Table 9 |
| Image enhancement before detection: sharpening | same | mAP@0.5 from 0.547 to 0.167 (−69.4%) | Table 9 |
| Hard-example mining alone | detector fine-tuned on the worst-performing sequences, default association | MOTA 76.0% to 76.7%, IDF1 unchanged at 71.2% | Subsection 4.2.3 |
| Hybrid architecture on swarms | CSRT, N_red = 15, no re-tuning | MOTA 60.9% and IDF1 59.4% against 76.0% and 71.2%; 2.6 FPS against 54.8; 32.9 ms per target | Table 12, Section 4.2.7 |
| TrackTrack as the baseline | default configuration | IDF1 68.9%, but MOTA 63.1% and about 2.6 times more FN | Table 6 |
| Velocity damping on Track 3 | gamma = 0.95 over the persistent Kalman filter | H30 error not reduced for any of the three trackers (paired difference of +0.085 and +0.110, not significant) | Table 15, hypothesis H1 |
| IDF1 as a predictor of prediction error | per-sequence Spearman correlation | rho = −0.29 and −0.28, not significant | Figure 8, hypothesis H2 |
| Choosing the local tracker by tracking metrics | MOSSE: higher SA0.5 and SR@0.5 and 6 times the FPS | H30 error 41.6% above CSRT, with 2.5 times the standard deviation | Table 13 |

The TCC lists as future work preprocessing techniques incorporated into detector training, adaptive damping, and
propagating tracks only with the Kalman filter between detections (Section 5.2).

## Code removed from the main tree

The repository before the reorganisation (commit `319ed05`) remains
in the Git history. The following left the main tree because it is not part of the final text of the TCC or already lives
in the repositories of the papers:

| Removed code | Destination |
|---|---|
| `mot_infravermelho/src/postprocessing/` (static, spatial-recurrence and hybrid spatio-dynamic filters; lost-track query bank; adaptive threshold; Geman-McClure and NSA-Kalman) | Geman-McClure and NSA-Kalman: `src/tracking/` of DroneTrackingMOT. The rest: Git history |
| `mot_infravermelho/src/inference/` (runs, sweeps and evaluation) and `src/analysis/` | DroneTrackingMOT (`src/tracking/`, `src/evaluation/`, `src/analysis/`) |
| `mot_infravermelho/mot_prediction/motpred/drift.py` and `dut_sige.py`, `steps/`, `make_summary.py` (analysis of CSRT drift between redetections, instrumentation of the SIGE hybrid) | Git history |
| `mot_infravermelho/paper_figures/` | LAFusion paper figures: `src/figures/` of DroneTrackingMOT |
| `sot_visivel/projeto_sige/` (data conversions, reviewer validations, hybrid variants with MOG2 and n-step Kalman) | Git history; the published method is in the SIGE repository |
| `sot_visivel/hybrid-uav-tracking-prediction/` (copy of the SIGE repository) | SIGE repository, commit `91725902` |
| `mot_infravermelho/tcc_hoje/` and `mot_prediction/` (Part 1 to 4 runs) | replaced by `etapa3_predicao/`, `stats/` and `results/` |
