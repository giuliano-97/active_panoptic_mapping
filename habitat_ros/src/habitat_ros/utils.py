import numpy as np
from geometry_msgs.msg import Vector3, Quaternion


def vector3_to_numpy(msg: Vector3) -> np.ndarray:
    return np.array([msg.x, msg.y, msg.z])


def numpy_to_vector3(arr: np.ndarray) -> Vector3:
    return Vector3(*arr)


def quaternion_to_numpy(msg: Quaternion) -> np.ndarray:
    return np.array([msg.x, msg.y, msg.z, msg.w])


def vec_ros_to_habitat(vec: np.ndarray) -> np.ndarray:
    """Convert vector in local coordinates from ros to habitat
    coordinate convention
    """
    return np.array([-vec[1], vec[2], -vec[0]])


def vec_habitat_to_ros(vec: np.ndarray) -> np.ndarray:
    return np.array([-vec[2], -vec[0], vec[1]])


def quat_ros_to_habitat(quat: np.ndarray) -> np.ndarray:
    return np.array([-quat[1], quat[2], -quat[0], quat[3]])


def quat_habitat_to_ros(quat: np.ndarray) -> np.ndarray:
    return np.array([-quat[2], -quat[0], quat[1], quat[3]])


REPLICA_CLASS_ID_TO_NAME = {
    1: "backpack",
    2: "base-cabinet",
    3: "basket",
    4: "bathtub",
    5: "beam",
    6: "beanbag",
    7: "bed",
    8: "bench",
    9: "bike",
    10: "bin",
    11: "blanket",
    12: "blinds",
    13: "book",
    14: "bottle",
    15: "box",
    16: "bowl",
    17: "camera",
    18: "cabinet",
    19: "candle",
    20: "chair",
    21: "chopping-board",
    22: "clock",
    23: "cloth",
    24: "clothing",
    25: "coaster",
    26: "comforter",
    27: "computer-keyboard",
    28: "cup",
    29: "cushion",
    30: "curtain",
    31: "ceiling",
    32: "cooktop",
    33: "countertop",
    34: "desk",
    35: "desk-organizer",
    36: "desktop-computer",
    37: "door",
    38: "exercise-ball",
    39: "faucet",
    40: "floor",
    41: "handbag",
    42: "hair-dryer",
    43: "handrail",
    44: "indoor-plant",
    45: "knife-block",
    46: "kitchen-utensil",
    47: "lamp",
    48: "laptop",
    49: "major-appliance",
    50: "mat",
    51: "microwave",
    52: "monitor",
    53: "mouse",
    54: "nightstand",
    55: "pan",
    56: "panel",
    57: "paper-towel",
    58: "phone",
    59: "picture",
    60: "pillar",
    61: "pillow",
    62: "pipe",
    63: "plant-stand",
    64: "plate",
    65: "pot",
    66: "rack",
    67: "refrigerator",
    68: "remote-control",
    69: "scarf",
    70: "sculpture",
    71: "shelf",
    72: "shoe",
    73: "shower-stall",
    74: "sink",
    75: "small-appliance",
    76: "sofa",
    77: "stair",
    78: "stool",
    79: "switch",
    80: "table",
    81: "table-runner",
    82: "tablet",
    83: "tissue-paper",
    84: "toilet",
    85: "toothbrush",
    86: "towel",
    87: "tv-screen",
    88: "tv-stand",
    89: "umbrella",
    90: "utensil-holder",
    91: "vase",
    92: "vent",
    93: "wall",
    94: "wall-cabinet",
    95: "wall-plug",
    96: "wardrobe",
    97: "window",
    98: "rug",
    99: "logo",
    100: "bag",
    101: "set-of-clothing",
}


REPLICA_CLASS_ID_TO_NYU40 = {
    1: 37,
    2: 40,
    3: 40,
    4: 39,
    5: 38,
    6: 39,
    7: 40,
    8: 39,
    9: 40,
    10: 40,
    11: 40,
    12: 13,
    13: 23,
    14: 40,
    15: 29,
    16: 40,
    17: 40,
    18: 3,
    19: 40,
    20: 5,
    21: 40,
    22: 40,
    23: 21,
    24: 21,
    25: 40,
    26: 40,
    27: 40,
    28: 40,
    29: 18,
    30: 16,
    31: 21,
    32: 12,
    33: 12,
    34: 14,
    35: 40,
    36: 40,
    37: 8,
    38: 40,
    39: 34,
    40: 2,
    41: 3,
    42: 40,
    43: 40,
    44: 40,
    45: 40,
    46: 40,
    47: 35,
    48: 40,
    49: 39,
    50: 20,
    51: 40,
    52: 40,
    53: 40,
    54: 32,
    55: 40,
    56: 40,
    57: 40,
    58: 40,
    59: 11,
    60: 38,
    61: 18,
    62: 38,
    63: 39,
    64: 40,
    65: 40,
    66: 39,
    67: 24,
    68: 40,
    69: 40,
    70: 40,
    71: 10,
    72: 21,
    73: 39,
    74: 34,
    75: 40,
    76: 6,
    77: 38,
    78: 39,
    79: 40,
    80: 7,
    81: 40,
    82: 40,
    83: 26,
    84: 33,
    85: 40,
    86: 27,
    87: 25,
    88: 39,
    89: 40,
    90: 40,
    91: 40,
    92: 38,
    93: 1,
    94: 3,
    95: 40,
    96: 3,
    97: 9,
    98: 20,
    99: 40,
    100: 37,
    101: 21,
}