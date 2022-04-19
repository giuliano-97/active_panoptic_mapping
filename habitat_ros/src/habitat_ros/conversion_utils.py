import numpy as np
from geometry_msgs.msg import Vector3


def vector3_to_numpy(msg: Vector3) -> np.ndarray:
    return np.array([msg.x, msg.y, msg.z])


def numpy_to_vector3(arr: np.ndarray) -> Vector3:
    return Vector3(*arr)


def vec_ros_to_habitat(vec: np.ndarray) -> np.ndarray:
    """Convert vector in local coordinates from ros to habitat
    coordinate convention
    """
    return np.array([-vec[1], vec[2], -vec[0]])


def vec_habitat_to_ros(vec: np.ndarray) -> np.ndarray:
    return np.array([-vec[2], -vec[0], vec[1]])


def quat_ros_to_habitat(quat: np.ndarray) -> np.ndarray:
    raise np.array([-quat[1], quat[2], -quat[0], quat[3]])


def quat_habitat_to_ros(quat: np.ndarray) -> np.ndarray:
    return np.array([-quat[2], -quat[0], quat[1], quat[3]])
