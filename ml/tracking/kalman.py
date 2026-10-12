"""Constant-acceleration Kalman filter in image coordinates with RTS smoothing.

State: [x, y, vx, vy, ax, ay] in pixels, pixels/s, pixels/s^2. Time steps come
from frame timestamps, so missing or irregular frames are handled naturally.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

H = np.zeros((2, 6))
H[0, 0] = H[1, 1] = 1.0


def transition(dt: float) -> np.ndarray:
    F = np.eye(6)
    F[0, 2] = F[1, 3] = dt
    F[2, 4] = F[3, 5] = dt
    F[0, 4] = F[1, 5] = 0.5 * dt * dt
    return F


def process_noise(dt: float, jerk_sigma: float) -> np.ndarray:
    """Discrete white-jerk noise for each axis."""
    q = jerk_sigma**2
    d2, d3, d4, d5 = dt**2, dt**3, dt**4, dt**5
    block = q * np.array(
        [[d5 / 20, d4 / 8, d3 / 6], [d4 / 8, d3 / 3, d2 / 2], [d3 / 6, d2 / 2, dt]]
    )
    Q = np.zeros((6, 6))
    for axis in (0, 1):
        idx = [axis, axis + 2, axis + 4]
        Q[np.ix_(idx, idx)] = block
    return Q


@dataclass
class KalmanCA2D:
    x: np.ndarray
    P: np.ndarray
    jerk_sigma: float

    @classmethod
    def init(cls, z: np.ndarray, meas_sigma: float, vel_sigma: float, acc_sigma: float, jerk_sigma: float):
        x = np.array([z[0], z[1], 0.0, 0.0, 0.0, 0.0])
        P = np.diag([meas_sigma**2] * 2 + [vel_sigma**2] * 2 + [acc_sigma**2] * 2)
        return cls(x, P, jerk_sigma)

    def predict(self, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        F = transition(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + process_noise(dt, self.jerk_sigma)
        return self.x.copy(), self.P.copy(), F

    def innovation(self, z: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        return y, S

    def update(self, z: np.ndarray, R: np.ndarray) -> float:
        y, S = self.innovation(z, R)
        Sinv = np.linalg.inv(S)
        K = self.P @ H.T @ Sinv
        self.x = self.x + K @ y
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph form for numerical stability
        return float(y @ Sinv @ y)


def mahalanobis2(kf: KalmanCA2D, z: np.ndarray, R: np.ndarray) -> float:
    y, S = kf.innovation(z, R)
    return float(y @ np.linalg.solve(S, y))


def rts_smooth(xs_f, Ps_f, xs_p, Ps_p, Fs) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Rauch-Tung-Striebel smoother. Fs[k] maps state k-1 -> k (Fs[0] unused)."""
    n = len(xs_f)
    xs, Ps = [None] * n, [None] * n
    xs[-1], Ps[-1] = xs_f[-1], Ps_f[-1]
    for k in range(n - 2, -1, -1):
        C = Ps_f[k] @ Fs[k + 1].T @ np.linalg.pinv(Ps_p[k + 1])
        xs[k] = xs_f[k] + C @ (xs[k + 1] - xs_p[k + 1])
        Ps[k] = Ps_f[k] + C @ (Ps[k + 1] - Ps_p[k + 1]) @ C.T
    return xs, Ps
