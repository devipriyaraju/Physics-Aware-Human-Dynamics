import numpy as np

JOINT_NAMES = [
    "pelvis", "L hip", "R hip", "spine1", "L knee", "R knee", "spine2",
    "L ankle", "R ankle", "spine3", "L toe", "R toe", "neck", "L collar",
    "R collar", "head", "L shoulder", "R shoulder", "L elbow", "R elbow",
    "L wrist", "R wrist",
]

JOINT_GROUPS = {
    "ankle": [7, 8],
    "knee": [4, 5],
    "hip": [1, 2],
    "lower back": [3],
    "shoulder": [16, 17],
    "elbow": [18, 19],
}

G_ACC = 9.81
GRAVITY_VECTOR = np.array([0.0, 0.0, -G_ACC])
