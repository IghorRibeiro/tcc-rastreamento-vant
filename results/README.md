# results/

Per-sequence files (one row per video or sequence) behind every table and number of the TCC. Headers are in English; MOTA
and IDF1 are in percent, SA, SR, IoU and AUC in [0, 1], errors in pixels except columns named `norm` (fraction of the target
diagonal), and times in milliseconds.

| File | Content | TCC table or figure |
|---|---|---|
| `detectors.csv` | detector performance on the DUT Anti-UAV frames | Table 4 |
| `sot_tracking_per_video.csv` | SA, SR@0.5, mean IoU and AUC per video and method | Table 5 |
| `sot_fps.csv` | FPS of each method (Apple M2) and the origin of the value | Tables 5 and 13 |
| `sot_prediction_per_video.csv` | prediction error at H5 to H30, persistent and damped, per video | Tables 13 and 14, Figure 7 |
| `sot_common_support_per_video.csv` | hybrid vs each reference on the same frames and horizons | Subsection 4.3.1 |
| `mot_tracking_per_seq.csv` | MOTA, IDF1, identity switches, FP and FN per sequence | Tables 10 and 12 |
| `mot_prediction_per_seq.csv` | identity-aware error at H5 to H30 per sequence (`clear` and `idf1` correspondences) | Table 15, Figure 8 |
| `mot_common_support_per_seq.csv` | comparison between trackers on the common support | Subsection 4.3.2 |
| `incremental_runs_per_seq.csv`, `incremental_runs_summary.csv` | tracker, hard-example mining, SAHI and Geman-McClure runs | Tables 6, 7 and 10, Figure 5 |
| `cost_per_seq.csv`, `cost_by_num_targets.csv`, `cost_sufficient_stats.csv` | time per frame and per number of targets | Table 12, Figure 6, hypothesis H4 |
| `root_cause_per_seq.csv` | baseline performance factors per sequence | Subsection 4.2.2, Figure 4 |
| `mot_target_size_per_seq.csv`, `sot_target_size_per_video.csv` | target size (diagonal) | Subsection 4.3 |
| `hypothesis_tests.csv` | the six tests of H1 to H4 and the complementary families, with Holm-adjusted p-values | Table 3, Sections 4.2.7 and 4.3 |
| `summary.csv` | every number recomputed from the files above | Tables 4 to 15 |

`hypothesis_tests.csv` and `summary.csv` are produced by `stats/hypothesis_tests.py` and `stats/make_tables.py`; the other
files are the per-sequence outputs of the experiments.

Origin of the per-sequence outputs: runs made on 7 October 2026 (Apple M2) from the saved trajectories of the SIGE 2026
and LAFusion 2026 code. Processing times were measured on a single machine and are comparable only between methods of the
same scenario. The trajectories (about 280 MB) are not in the repository.

FPS of the local trackers (CSRT, KCF, MOSSE, MedianFlow) in `sot_fps.csv` are the values reported in Table 2 of the SIGE 2026
paper; the FPS of the other single-target methods is the mean of the per-video FPS logged during the runs.
