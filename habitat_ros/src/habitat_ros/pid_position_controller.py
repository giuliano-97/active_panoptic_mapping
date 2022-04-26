from dataclasses import dataclass
from typing import Tuple

import numpy as np
from geometry_msgs.msg import Twist


@dataclass
class XYZYaw:
    xyz: np.ndarray = np.zeros((3,))
    yaw: float = 0.0


@dataclass
class PIDPositionControllerParameters:
    kp_x: float = 0.5
    kp_y: float = 0.5
    kp_z: float = 0.10
    kp_yaw: float = 0.4
    kd_x: float = 0.3
    kd_y: float = 0.3
    kd_z: float = 0.2
    kd_yaw: float = 0.1
    max_horz_speed: float = 1.0
    max_vert_speed: float = 0.5
    max_yaw_rate_degrees: float = 10.0
    reached_thresh_xyz: float = 0.1
    reached_thresh_yaw_degrees: float = 0.1


def wrap(angle: float):
    r"""
    Limit angle to [-pi, pi)
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


class PIDPositionController:
    def __init__(self, params: PIDPositionControllerParameters):
        self.params = params
        self.kp_xyz = np.array([params.kp_x, params.kp_y, params.kp_z])
        self.kd_xyz = np.array([params.kd_x, params.kd_y, params.kd_z])
        self.target_position = XYZYaw()
        self.curr_position = XYZYaw()
        self.cmd_vel = Twist()
        self.prev_error = XYZYaw()
        self.curr_error = XYZYaw()

    def _reset_errors(self):
        self.prev_error.xyz = np.zeros((3,))
        self.prev_error.yaw = 0.0
        self.curr_error.xyz = np.zeros((3,))
        self.curr_error.yaw = 0.0

    def set_target(self, xyz: np.ndarray, yaw: float):
        self.target_position.xyz = xyz
        self.target_position.yaw = wrap(yaw)
        self._reset_errors()

    def is_goal_reached(self) -> bool:
        dist_xyz = np.linalg.norm(self.target_position.xyz - self.curr_position.xyz)
        dist_yaw = wrap(self.target_position.yaw - self.curr_position.yaw)

        return dist_xyz < self.params.reached_thresh_xyz and dist_yaw < np.radians(
            self.params.reached_thresh_yaw_degrees
        )

    def _clip_control_cmd(self, linear_vel: np.ndarray, yaw_rate: float) -> np.ndarray:
        horz_speed = np.linalg.norm(linear_vel[:2])
        if horz_speed > self.params.max_horz_speed:
            linear_vel[:2] = (linear_vel[:2] / horz_speed) * self.params.max_horz_speed

        linear_vel[2] = np.sign(linear_vel[2]) * min(
            linear_vel[2], self.params.max_vert_speed
        )

        yaw_rate = np.sign(yaw_rate) * min(
            yaw_rate, np.radians(self.params.max_yaw_rate_degrees)
        )
        return linear_vel, yaw_rate

    def compute_control_cmd(
        self, curr_xyz: np.ndarray, curr_yaw: float
    ) -> Tuple[np.ndarray, float]:
        # Update the current position
        self.curr_position.xyz = curr_xyz.copy()
        self.curr_position.yaw = wrap(curr_yaw)

        self.prev_error.xyz = self.curr_error.xyz.copy()
        self.prev_error.yaw = self.curr_error.yaw

        self.curr_error.xyz = self.target_position.xyz - self.curr_position.xyz
        self.curr_error.yaw = wrap(self.target_position.yaw - self.curr_position.yaw)

        p_term_xyz = np.multiply(self.curr_error.xyz, self.kp_xyz)
        p_term_yaw = self.params.kp_yaw * self.curr_error.yaw
        d_term_xyz = np.multiply(self.prev_error.xyz, self.kd_xyz)
        d_term_yaw = self.params.kd_yaw * self.prev_error.yaw

        linear_vel = p_term_xyz + d_term_xyz
        yaw_rate = p_term_yaw + d_term_yaw
        self._clip_control_cmd(linear_vel, yaw_rate)

        return linear_vel, yaw_rate
