"""The mathematical operations for matrix Lie groups and Lie algebras.

Implement here the Lie group functions your kinematics and controller need,
split into small elementary functions. The four functions below must keep
their names and signatures: the provided code imports `so3_exp`, `unskew` and
`se3_inverse`, and `se3_exp` is its natural counterpart for the kinematics.
Add any other functions you find useful.
"""

import numpy as np

# Below this norm an angular velocity counts as zero. The exponential of a
# twist divides by the squared norm of its angular part, which is what makes
# the distinction necessary.
ANGULAR_TOLERANCE = 1e-12

def skew(omega) -> np.ndarray:
    """Convert a 3D vector into a skew-symmetric matrix."""
    x, y, z = omega

    return np.array([
        [0, -z,  y],
        [z,  0, -x],
        [-y, x,  0]
    ])

def unskew(omega_tilde) -> np.ndarray:
    """Coordinate vector of a skew-symmetric matrix: so(3) -> R^3.

    Args:
        omega_tilde: a skew-symmetric 3x3 matrix.

    Returns:
        The vector omega with skew(omega) = omega_tilde, an array of 3 numbers.
    """

    omega = np.array([
        omega_tilde[2, 1],
        omega_tilde[0, 2],
        omega_tilde[1, 0]
    ])

    return omega

    raise NotImplementedError("unskew is not implemented yet")


def so3_exp(omega) -> np.ndarray:
    """Exponential map of SO(3): R^3 -> SO(3), by Rodrigues' formula.

    Args:
        omega: the exponential coordinates of a rotation, 3 numbers.

    Returns:
        The rotation matrix exp(omega_tilde), 3x3.
    """

    omega = np.asarray(omega, dtype=float)
    theta = np.linalg.norm(omega)

    if theta < ANGULAR_TOLERANCE:
        W = skew(omega)
        return np.eye(3) + W + 0.5 * (W @ W)

    # Normalized rotation axis
    w = omega / theta

    # Skew-symmetric matrix
    w_tilde = skew(w)

    # Rodrigues' formula
    R = (np.eye(3)
         + np.sin(theta) * w_tilde
         + (1 - np.cos(theta)) * (w_tilde @ w_tilde))

    return R

    raise NotImplementedError("so3_exp is not implemented yet")


def se3_exp(twist) -> np.ndarray:
    """Exponential map of SE(3): R^6 -> SE(3).

    Args:
        twist: T = (omega, v), 6 numbers.

    Returns:
        The homogeneous matrix exp(T_tilde), 4x4.
    """

    omega = np.asarray(twist[:3], dtype=float)
    v = np.asarray(twist[3:], dtype=float)

    theta = np.linalg.norm(omega)

    omega_tilde = skew(omega)

    R = so3_exp(omega)

    if theta < 1e-8:
        V = (np.eye(3)
             + 0.5 * omega_tilde
             + (1/6) * (omega_tilde @ omega_tilde))
    else:
        V = (np.eye(3)
             + (1 - np.cos(theta)) / theta**2 * omega_tilde
             + (theta - np.sin(theta)) / theta**3
             * (omega_tilde @ omega_tilde))

    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = V @ v

    return H

    raise NotImplementedError("se3_exp is not implemented yet")


def se3_inverse(H) -> np.ndarray:
    """Inverse of a homogeneous matrix, computed in closed form.

    Args:
        H: a 4x4 homogeneous matrix.

    Returns:
        The inverse of H, 4x4.
    """
    R = H[:3, :3]
    p = H[:3, 3]

    H_inv = np.eye(4)
    H_inv[:3, :3] = R.T
    H_inv[:3, 3] = -R.T @ p

    return H_inv

    raise NotImplementedError("se3_inverse is not implemented yet")