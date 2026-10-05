import numpy as np
from scipy.signal import butter, filtfilt

def lowpass(x: np.ndarray, fps: float, cutoff_hz: float = 6.0) -> np.ndarray:
    b, a = butter(4, cutoff_hz / (fps / 2.0))
    return filtfilt(b, a, x, axis=0)

def derivative(x: np.ndarray, fps: float) -> np.ndarray:
    return np.gradient(x, 1.0 / fps, axis=0)

def angular_velocity(rotation: np.ndarray, fps: float) -> np.ndarray:
    wmat = derivative(rotation, fps) @ np.swapaxes(rotation, -1, -2)
    return np.stack(
        [
            wmat[..., 2, 1] - wmat[..., 1, 2],
            wmat[..., 0, 2] - wmat[..., 2, 0],
            wmat[..., 1, 0] - wmat[..., 0, 1],
        ],
        axis=-1,
    ) / 2.0
