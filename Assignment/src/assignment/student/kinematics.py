"""Kinematics of a serial chain.

The conventions are: links are numbered 0..n, joints 1..n,
joint i connects link i-1 to link i, and the unit twist of
joint i is expressed in the child link frame Psi_i.

The kinematic description of a chain is constant and lives in a `ChainModel`;
the quantities that depend on the configuration live in a `ChainData`, which is
allocated once per chain and refilled by `update_kinematics` at each new
configuration.
"""

from pyexpat import model

import numpy as np

from assignment.kinematics_io import ChainModel
# Add any other necessary imports from lie_groups if needed.
from assignment.student.lie_groups import so3_exp, skew, unskew, se3_inverse, se3_exp



class ChainData:
    """Quantities of a serial chain at one configuration.

    Allocated once for a given chain and refilled in place by
    `update_kinematics`, so that a control loop evaluating the kinematics at
    every step does not keep reallocating. All fields refer to the
    configuration held in `q`, and are meaningless before the first update,
    where `q` is filled with NaN to make that obvious.

    Attributes:
        q: the configuration the other fields were computed at, shape (n,).
        H_rel: H_i^{i-1}(q^i) for each joint, shape (n, 4, 4).
        H_link_w: H_i^w, the link frames relative to the world frame, shape (n+1, 4, 4),
            indexed by the link number 0..n.
        H_ee_w: H_ee^w, the end-effector frame relative to the world frame, shape (4, 4).
        J_ee: the geometric Jacobian expressed in the end-effector frame,
            shape (6, n), with columns T = (omega, v).
    """

    def __init__(self, model: ChainModel):
        self.q = np.full(model.n, np.nan)
        self.H_rel = np.empty((model.n, 4, 4))
        self.H_link_w = np.empty((model.n + 1, 4, 4))
        self.H_ee_w = np.empty((4, 4))
        self.J_ee = np.empty((6, model.n))





def update_kinematics(model: ChainModel, data: ChainData, q) -> None:
    """Fill `data` with the kinematics of `model` at the configuration `q`.

    Args:
        model: the kinematic description of the serial chain.
        data: the container to fill, allocated for this chain.
        q: the joint variables, an array of n numbers.
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (model.n,):
        raise ValueError(f"expected {model.n} joint variables, got shape {q.shape}")

    data.q = q
    data.H_link_w[0] = model.H_0_w

    for i in range(model.n):
        data.H_rel[i] = model.H_ref[i] @ se3_exp(model.unit_twists[i] * q[i])
        data.H_link_w[i + 1] = data.H_link_w[i] @ data.H_rel[i]

    data.H_ee_w = data.H_link_w[model.n] @ model.H_ee_n #model.n = 6 --> 7th link frame

    # World frame relative to end-effector
    H_w_ee = se3_inverse(data.H_ee_w)

    for i in range(model.n):
        H_i_ee = H_w_ee @ data.H_link_w[i + 1]

        R = H_i_ee[:3, :3]
        p = H_i_ee[:3, 3]

        Ad = np.block([
            [R,                np.zeros((3, 3))],
            [skew(p) @ R,      R]
        ])

        data.J_ee[:, i] = Ad @ model.unit_twists[i]