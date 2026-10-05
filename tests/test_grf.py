import numpy as np
from physics_human_dynamics.grf import physics_grf

def test_static_single_support():
    n = 5
    trial = {
        "acom": np.zeros((n, 3)),
        "Hdot": np.zeros((n, 3)),
        "z0": 0.0,
        "com": np.tile(np.array([[0.0, 0.0, 1.0]]), (n, 1)),
        "feet": np.tile(
            np.array([[[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]]),
            (n, 1, 1),
        ),
    }
    contact = np.zeros((n, 2), dtype=bool)
    contact[:, 0] = True
    per_foot, _, total = physics_grf(trial, contact)
    assert np.allclose(total[:, 2], 1.0)
    assert np.allclose(per_foot[:, 0, 2], 1.0)
    assert np.allclose(per_foot[:, 1, 2], 0.0)
