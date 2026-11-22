"""Control of the arm in the task space."""

import mujoco
import numpy as np

from assignment.student import config as default_settings
from assignment.student.kinematics import ChainData, update_kinematics
from assignment.student.lie_groups import se3_inverse, skew, unskew

def spring_wrench(H_ee_d, k_rot: float, k_trans: float, k_coupling: float = 0.0) -> np.ndarray:
    """Wrench of a virtual spatial spring pulling the end-effector to a target
    expressed in the end-effector frame Psi_ee.

    Args:
        H_ee_d: H_ee^d, the pose of the end-effector frame relative to the desired
            frame, a 4x4 homogeneous matrix.
        k_rot: rotational stiffness.
        k_trans: translational stiffness.
        k_coupling: coupling stiffness, zero for a spring whose rotational and
            translational actions are independent.

    Returns:
        W^{ee}, the wrench of the spring on the end-effector expressed in the
        end-effector frame, an array of 6 numbers (m, f).
    """
    H_ee_d = np.asarray(H_ee_d, dtype=float)
    if H_ee_d.shape != (4, 4):
        raise ValueError(f"expected a 4x4 homogeneous matrix, got {H_ee_d.shape}")

    # Relative rotation and translation
    R_ee_d = H_ee_d[:3, :3]
    p_ee_d = H_ee_d[:3, 3]

    R_d_ee = R_ee_d.T
    p_tilde = skew(p_ee_d)

    # Co-stiffness matrices (Equation 6)
    G_o = (k_rot / 2) * np.eye(3)
    G_t = (k_trans / 2) * np.eye(3)
    G_c = (k_coupling / 2) * np.eye(3)

    # Skew-symmetric part of a matrix
    def asym(A):
        return 0.5 * (A - A.T)

    # Moment (Equation 5)
    m_tilde = (
        -2 * asym(G_o @ R_ee_d)
        - asym(G_t @ R_d_ee @ p_tilde @ p_tilde @ R_ee_d)
        - 2 * asym(G_c @ p_tilde @ R_ee_d)
    )

    # Force (Equation 5)
    f_tilde = (
        - R_d_ee @ asym(G_t @ p_tilde) @ R_ee_d
        - asym(G_t @ R_d_ee @ p_tilde @ R_ee_d)
        - 2 * asym(G_c @ R_ee_d)
    )

    # Convert skew-symmetric matrices to vectors
    m = unskew(m_tilde)
    f = unskew(f_tilde)

    return np.concatenate((m, f))


def clip_wrench(wrench, max_moment=None, max_force=None) -> np.ndarray:
    """Cap the moment and the force of a wrench, preserving their directions.

    Args:
        wrench: W = (m, f), 6 numbers.
        max_moment: largest allowed ||m||, or None for no limit.
        max_force: largest allowed ||f||, or None for no limit.

    Returns:
        The capped wrench, an array of 6 numbers.
    """
    wrench = np.array(wrench, dtype=float)

    m = wrench[:3]
    f = wrench[3:]

    # Clip moment (Equation 7)
    m_norm = np.linalg.norm(m)

    if max_moment is not None and m_norm > max_moment:
        m *= max_moment / m_norm

    # Clip force (Equation 7)
    f_norm = np.linalg.norm(f)

    if max_force is not None and f_norm > max_force:
        f *= max_force / f_norm

    return wrench


def control_wrench(
    H_ee_w,
    twist_ee,
    H_d_w,
    twist_d,
    k_rot: float,
    k_trans: float,
    k_coupling: float = 0.0,
    *,
    d_rot: float,
    d_trans: float,
    max_moment=None,
    max_force=None,
) -> np.ndarray:
    """Wrench of a virtual spring and damper connecting the end-effector to a target.

    The spring pulls the end-effector towards the target pose, and the damper
    opposes the motion of the end-effector relative to the target.
    """
    H_ee_w = np.asarray(H_ee_w, dtype=float)
    H_d_w = np.asarray(H_d_w, dtype=float)
    twist_ee = np.asarray(twist_ee, dtype=float)
    twist_d = np.asarray(twist_d, dtype=float)

    if twist_ee.shape != (6,) or twist_d.shape != (6,):
        raise ValueError(
            f"expected twists of shape (6,), got {twist_ee.shape} and {twist_d.shape}"
        )

    # Relative pose: end-effector expressed in target frame (Equation 4)
    H_ee_d = se3_inverse(H_d_w) @ H_ee_w

    # Spring wrench (Equation 5)
    W_spring = spring_wrench(
        H_ee_d,
        k_rot,
        k_trans,
        k_coupling
    )

    # Clip only the spring wrench (Equation 7)
    W_spring = clip_wrench(W_spring, max_moment, max_force)

    # Target frame expressed in end-effector frame
    H_d_ee = se3_inverse(H_ee_d)

    R = H_d_ee[:3, :3]
    p = H_d_ee[:3, 3]

    # Adjoint transformation of the target twist
    omega_d_ee = R @ twist_d[:3]
    v_d_ee = R @ twist_d[3:] + np.cross(p, omega_d_ee)

    twist_d_ee = np.concatenate((omega_d_ee, v_d_ee))

    # Relative twist (Equation 8)
    twist_rel = twist_ee - twist_d_ee

    # Damping wrench (Equation 9)
    W_damping = np.concatenate((
        -d_rot * twist_rel[:3],
        -d_trans * twist_rel[3:]
    ))

    # Total control wrench
    return W_spring + W_damping


def control_torque(J_ee, wrench) -> np.ndarray:
    """Joint torques realising a task-space wrench on a torque-driven arm.

    Args:
        J_ee: the Jacobian in the end-effector frame, a 6 x n array.
        wrench: W, the wrench on the end-effector expressed in the
            end-effector frame, 6 numbers (m, f).

    Returns:
        The joint torques, an array of n numbers.
    """
    J_ee = np.asarray(J_ee, dtype=float)
    wrench = np.asarray(wrench, dtype=float)

    if wrench.shape != (6,):
        raise ValueError(f"expected a wrench of shape (6,), got {wrench.shape}")

    # Jacobian transpose mapping (Equation 10)
    tau = J_ee.T @ wrench

    return tau


def joint_speed_guard(tau, qvel, torque_limits, *, soft_speed: float, max_speed: float):
    """Brake the joints that run too fast, on top of the control torques.

    Near a singular configuration a modest wrench asks for very large joint speeds.
    This adds a braking torque that opposes the motion of any joint above `soft_speed`,

        tau_guard = -c * max(0, |dq| - soft_speed)^2 * sign(dq),

    with c chosen per joint so that the brake reaches the full torque of the motor at `max_speed`.

    Args:
        tau: the control torques, n numbers.
        qvel: dq, the joint speeds, n numbers.
        torque_limits: the largest torque of each motor, n positive numbers.
        soft_speed: the speed above which braking starts.
        max_speed: the speed at which braking reaches the torque limits.

    Returns:
        The guarded joint torques, an array of n numbers.
    """
    tau = np.asarray(tau, dtype=float)
    qvel = np.asarray(qvel, dtype=float)
    torque_limits = np.asarray(torque_limits, dtype=float)
    if not max_speed > soft_speed:
        raise ValueError("expected soft_speed < max_speed")

    coefficient = torque_limits / (max_speed - soft_speed) ** 2
    excess = np.maximum(0.0, np.abs(qvel) - soft_speed)
    return tau - coefficient * excess**2 * np.sign(qvel)


class TargetTracker:
    """One step of the task-space controller, run against a MuJoCo model.

    The model has to be one with torque actuators that compensates its own
    gravity, as `scene_torque.xml` does.

    Attributes:
        kinematics: the `ChainData` filled at the last step, so that a caller
            can draw the end-effector frame or read the Jacobian.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, chain, settings=None):
        """
        Args:
            model: the compiled torque-controlled model, which compensates
                its own gravity.
            data: its state, which this tracker reads and writes `ctrl` of.
            chain: the `ChainModel` describing the kinematics.
            settings: where the gains are read from, by default the module
                `assignment.student.config`. Anything carrying the same names
                will do, which is how a script can sweep a gain without
                editing the config.
        """
        self.model = model
        self.data = data
        self.chain = chain
        self.settings = default_settings if settings is None else settings
        self.kinematics = ChainData(chain)
        self._torque_limits = np.abs(model.actuator_ctrlrange[:, 1])
        self._warned_about_speed = False

    def step(self, H_d_w, twist_d=None) -> np.ndarray:
        """Compute the control torques for the current state and command them.

        Args:
            H_d_w: H_d^w, the pose of the target frame relative to the world frame.
            twist_d: T_d^{d,w}, the twist of the target frame expressed in that frame,
                6 numbers, or None for a target standing still.

        Returns:
            The commanded wrench on the end-effector, (m, f) in the
            end-effector frame, for a caller that wants to record it.
        """
        settings = self.settings
        update_kinematics(self.chain, self.kinematics, self.data.qpos)

        wrench = control_wrench(
            self.kinematics.H_ee_w,
            self.kinematics.J_ee @ self.data.qvel,
            H_d_w,
            np.zeros(6) if twist_d is None else twist_d,
            k_rot=settings.K_ROT,
            k_trans=settings.K_TRANS,
            k_coupling=settings.K_COUPLING,
            d_rot=settings.D_ROT,
            d_trans=settings.D_TRANS,
            max_moment=settings.MAX_MOMENT,
            max_force=settings.MAX_FORCE,
        )

        tau = control_torque(self.kinematics.J_ee, wrench)
        self.data.ctrl[:] = joint_speed_guard(
            tau,
            self.data.qvel,
            self._torque_limits,
            soft_speed=settings.JOINT_SPEED_SOFT,
            max_speed=settings.JOINT_SPEED_MAX,
        )
        self._check_joint_speed(settings.JOINT_SPEED_SAFETY)
        return wrench

    def _check_joint_speed(self, safety_speed: float) -> None:
        """Warn, once per run, if a joint has passed the safety speed.

        The guard is meant to keep every joint well below it, so this firing
        means the arm is doing something the controller did not anticipate.
        """
        if self._warned_about_speed:
            return
        fastest = int(np.argmax(np.abs(self.data.qvel)))
        speed = float(abs(self.data.qvel[fastest]))
        if speed > safety_speed:
            self._warned_about_speed = True
            print(
                f"warning: joint {fastest} reached {speed:.2f} rad/s, "
                f"past the safety limit of {safety_speed:.2f} rad/s"
            )

    def error(self, H_d_w) -> tuple[float, float]:
        """How far the end-effector is from a target, after the last step.

        Args:
            H_d_w: the same target pose that was passed to `step`.

        Returns:
            The distance in metres and the angle of the residual rotation in
            radians, both zero when the end-effector sits on the target.
        """
        H_ee_d = se3_inverse(np.asarray(H_d_w, dtype=float)) @ self.kinematics.H_ee_w
        distance = float(np.linalg.norm(H_ee_d[:3, 3]))
        cosine = (np.trace(H_ee_d[:3, :3]) - 1.0) / 2.0
        return distance, float(np.arccos(np.clip(cosine, -1.0, 1.0)))
