import numpy as np
from .kinematics import lowpass, derivative, angular_velocity
from .constants import GRAVITY_VECTOR

def body_state(
    joints_relative,
    global_rotation,
    pelvis,
    scale,
    body_parameters,
    fps,
    cutoff_hz=6.0,
    apply_filter=True,
):
    mass_fraction, com_offset, inertia_local = body_parameters
    filt = (lambda x: lowpass(x, fps, cutoff_hz)) if apply_filter else (lambda x: x)

    joints_world = filt(pelvis)[:, None] + scale * filt(joints_relative)
    segment_com = joints_world + scale * np.einsum(
        "nkij,kj->nki", global_rotation, com_offset
    )
    acceleration = derivative(derivative(segment_com, fps), fps)
    omega = filt(angular_velocity(global_rotation, fps))
    alpha = derivative(omega, fps)
    inertia_world = np.einsum(
        "nkij,kjl,nkml->nkim",
        global_rotation,
        inertia_local * scale ** 2,
        global_rotation,
    )

    segment_force = mass_fraction[None, :, None] * (
        acceleration - GRAVITY_VECTOR
    )
    segment_moment = (
        np.einsum("nkij,nkj->nki", inertia_world, alpha)
        + np.cross(
            omega,
            np.einsum("nkij,nkj->nki", inertia_world, omega),
        )
    )
    whole_com = (mass_fraction[None, :, None] * segment_com).sum(1)
    whole_com_acc = (mass_fraction[None, :, None] * acceleration).sum(1)
    hdot = (
        np.cross(
            segment_com - whole_com[:, None],
            mass_fraction[None, :, None] * acceleration,
        )
        + segment_moment
    ).sum(1)

    return {
        "Jw": joints_world,
        "p": segment_com,
        "f": segment_force,
        "nm": segment_moment,
        "com": whole_com,
        "acom": whole_com_acc,
        "Hdot": hdot,
    }

def newton_euler(state, external_force, external_point, children):
    n = len(state["Jw"])
    joint_force = np.zeros((n, 22, 3))
    joint_moment = np.zeros((n, 22, 3))
    external_map = {7: 0, 8: 1}

    for i in reversed(range(22)):
        force_i = state["f"][:, i].copy()
        moment_i = state["nm"][:, i].copy()

        for child in children[i]:
            force_i += joint_force[:, child]
            moment_i += joint_moment[:, child]
            moment_i += np.cross(
                state["Jw"][:, child] - state["p"][:, i],
                joint_force[:, child],
            )

        if i in external_map:
            foot = external_map[i]
            force_i -= external_force[:, foot]
            moment_i -= np.cross(
                external_point[:, foot] - state["p"][:, i],
                external_force[:, foot],
            )

        moment_i -= np.cross(
            state["Jw"][:, i] - state["p"][:, i],
            force_i,
        )
        joint_force[:, i] = force_i
        joint_moment[:, i] = moment_i

    return joint_force, joint_moment
