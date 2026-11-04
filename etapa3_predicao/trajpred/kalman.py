"""External constant-velocity Kalman filter used for trajectory prediction.

The filter is applied after tracking, on the recorded trajectory, so every tracker is
evaluated with the same predictor. State is [x, y, vx, vy] in image pixels, one step
per reported observation (frames without a reported position are skipped).

Parameters follow the TCC (Section 3.4.4.1): Q = 0.03 I, R = 0.1 I, P0 = I, zero initial
velocity. Two configurations are supported:

* persistent: state and covariance are kept along the whole trajectory (damping = 1);
* damped: after every correction the velocity components are multiplied by ``damping``
  (0.95 in the TCC), and the same factor is applied at every step of the extrapolation.
"""

from __future__ import annotations

import cv2
import numpy as np

HORIZONS = (5, 10, 15, 30)

PROCESS_NOISE = 0.03
MEASUREMENT_NOISE = 0.1

TRANSITION = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
MEASUREMENT = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)


def init_kalman(cx: float, cy: float) -> cv2.KalmanFilter:
    """Filter initialised at the first observed centre, with zero velocity."""
    kalman = cv2.KalmanFilter(4, 2)
    kalman.transitionMatrix = TRANSITION.copy()
    kalman.measurementMatrix = MEASUREMENT.copy()
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * PROCESS_NOISE
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * MEASUREMENT_NOISE
    kalman.errorCovPost = np.eye(4, dtype=np.float32)
    state = np.array([[cx], [cy], [0.0], [0.0]], np.float32)
    kalman.statePre = state.copy()
    kalman.statePost = state.copy()
    return kalman


def forward(state: np.ndarray, horizon: int, damping: float = 1.0) -> np.ndarray:
    """Propagate ``state`` ``horizon`` frames ahead; the damping is applied at every step."""
    predicted = state.copy()
    for _ in range(horizon):
        predicted = TRANSITION @ predicted
        if damping != 1.0:
            predicted[2, 0] *= damping
            predicted[3, 0] *= damping
    return predicted


def corrected_states(centers: dict, damping: float = 1.0):
    """Yield ``(frame, corrected_state)`` for every observation after the first one.

    ``centers`` maps frame -> (cx, cy). Frames are visited in increasing order.
    """
    kalman = None
    for frame in sorted(centers):
        cx, cy = centers[frame]
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        if kalman is None:
            kalman = init_kalman(cx, cy)
            continue

        kalman.predict()
        kalman.correct(measurement)
        if damping != 1.0:
            damped = kalman.statePost.copy()
            damped[2, 0] *= damping
            damped[3, 0] *= damping
            kalman.statePost = damped

        yield frame, kalman.statePost.copy()


def single_target_errors(centers: dict, gt_centers: dict, damping: float = 1.0, horizons=HORIZONS) -> dict:
    """Prediction error in pixels per horizon for a sequence with a single target.

    ``gt_centers`` maps frame -> (cx, cy) and holds only the frames in which the target exists.
    """
    errors = {h: [] for h in horizons}
    for frame, state in corrected_states(centers, damping):
        for horizon in horizons:
            target = gt_centers.get(frame + horizon)
            if target is None:
                continue
            predicted = forward(state, horizon, damping)
            errors[horizon].append(float(np.hypot(predicted[0, 0] - target[0], predicted[1, 0] - target[1])))
    return errors
