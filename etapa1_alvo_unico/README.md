# Stage 1 — single target, visible spectrum (DUT Anti-UAV)

This stage has no code of its own in this repository: the code that produces the single-drone trajectories is the one of
the **SIGE 2026** paper, kept in
[`hybrid-uav-tracking-prediction`](https://github.com/IghorRibeiro/hybrid-uav-tracking-prediction) and pinned to commit

```
91725902bc2824441b430c3f205ecd3e51f072d8
```

```bash
git clone https://github.com/IghorRibeiro/hybrid-uav-tracking-prediction
git -C hybrid-uav-tracking-prediction checkout 91725902bc2824441b430c3f205ecd3e51f072d8
```

## What comes from there

| Item | Where |
|---|---|
| Hybrid architecture (YOLO26n every 15 frames + CSRT, confidence thresholds 0.5 and 0.7) | `src/04_rastreadores.py` |
| BoT-SORT, ByteTrack and DeepSORT by detection; OSTrack and ODTrack | `src/04_rastreadores.py`, `src/extras/` |
| SA, SR@0.5, mean IoU and AUC | `src/05_avaliar.py` |
| Fine-tuning of YOLO26n and YOLO26m on DUT Anti-UAV | `src/03_treinar_detector.py` |

## What is specific to the TCC

- choosing the local tracker (CSRT, KCF, MOSSE, MedianFlow) by prediction quality, not only by tracking metrics;
- the trajectory-prediction error with the standardised external Kalman filter, persistent and damped (gamma = 0.95), at
  H5, H10, H15 and H30, and the common support between methods:
  [`etapa3_predicao/predict_sot.py`](../etapa3_predicao/predict_sot.py);
- the paired tests and the Holm correction: [`stats/`](../stats/).

The trajectories used by the TCC are in `avaliacao_tracking/predicoes/<method>/videoNN.txt` after running the SIGE code;
`predict_sot.py` reads those files and the official annotations (`videoNN_gt.txt`).

## Data

DUT Anti-UAV (Zhao et al., 2022), not redistributed. See the "Dados" section of the [main README](../README.md).
