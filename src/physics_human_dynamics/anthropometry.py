import numpy as np
from scipy.spatial import ConvexHull, Delaunay

def children_from_parents(parents: list[int], n_joints: int = 22) -> list[list[int]]:
    return [[c for c in range(n_joints) if parents[c] == i] for i in range(n_joints)]

def segment_parameters(model, n_samples: int = 40000, seed: int = 0):
    rng = np.random.RandomState(seed)
    vertices = model.v_template.detach().cpu().numpy()
    joints = (model.joint_regressor @ model.v_template).detach().cpu().numpy()
    owner = model.weights.argmax(1).detach().cpu().numpy().copy()

    for joint in range(22, model.n_joints):
        ancestor = joint
        while ancestor >= 22:
            ancestor = model.parents[ancestor]
        owner[owner == joint] = ancestor

    volume = np.zeros(22)
    com = np.zeros((22, 3))
    inertia = np.zeros((22, 3, 3))

    for i in range(22):
        points = vertices[owner == i]
        hull = ConvexHull(points)
        tessellation = Delaunay(points[hull.vertices])
        lo = points.min(0)
        hi = points.max(0)
        samples = lo + rng.rand(n_samples, 3) * (hi - lo)
        samples = samples[tessellation.find_simplex(samples) >= 0]
        centered = samples - samples.mean(0)

        volume[i] = hull.volume
        com[i] = samples.mean(0) - joints[i]
        inertia[i] = (
            np.eye(3) * (centered ** 2).sum(1).mean()
            - (centered[:, :, None] * centered[:, None, :]).mean(0)
        )

    mass_fraction = volume / volume.sum()
    inertia = inertia * mass_fraction[:, None, None]
    return mass_fraction, com, inertia

def winter_table_parameters(model):
    joints = (model.joint_regressor @ model.v_template).detach().cpu().numpy()
    parents = model.parents[:22]
    children = children_from_parents(parents)

    mass_fraction = np.array([
        .142, .100, .100, .0695, .0465, .0465, .0695, .0145, .0145, .216,
        0, 0, .020, 0, 0, .061, .028, .028, .016, .016, .006, .006
    ])
    com = np.zeros((22, 3))
    inertia = np.zeros((22, 3, 3))

    for i in range(22):
        if children[i]:
            bone = joints[children[i][0]] - joints[i]
            fraction = 0.433 if i in (1, 2, 4, 5, 16, 17, 18, 19) else 0.5
            com[i] = fraction * bone
            inertia[i] = np.eye(3) * mass_fraction[i] * (0.3 * np.linalg.norm(bone)) ** 2
    return mass_fraction, com, inertia
