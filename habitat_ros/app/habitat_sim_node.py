#!/usr/bin/env python3

from dataclasses import dataclass
from queue import Queue
from threading import Thread, Lock

import cv2
import numpy as np
import rospy
import tf
import tf2_ros
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from geometry_msgs.msg import (
    Twist,
    Point,
    Pose,
    PoseStamped,
    Quaternion,
    Transform,
    TransformStamped,
)
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from trajectory_msgs.msg import MultiDOFJointTrajectory

from habitat_ros.async_simulator import AsyncSimulator
from habitat_ros.pid_position_controller import (
    PIDPositionController,
    PIDPositionControllerParameters,
)
from habitat_ros.utils.conversions import (
    numpy_to_vector3,
    vector3_to_numpy,
    vec_habitat_to_ros,
    vec_ros_to_habitat,
    quaternion_to_numpy,
    quat_ros_to_habitat,
    quat_habitat_to_ros,
)


def make_depth_camera_info_msg(header, height, width):
    r"""
    Create camera info message for depth camera.
    :param header: header to create the message
    :param height: height of depth image
    :param width: width of depth image
    :returns: camera info message of type CameraInfo.
    """
    # code modifed upon work by Bruce Cui
    camera_info_msg = CameraInfo()
    camera_info_msg.header = header
    fx, fy = width / 2, height / 2
    cx, cy = width / 2, height / 2

    camera_info_msg.width = width
    camera_info_msg.height = height
    camera_info_msg.distortion_model = "plumb_bob"
    camera_info_msg.K = np.float32([fx, 0, cx, 0, fy, cy, 0, 0, 1])
    camera_info_msg.D = np.float32([0, 0, 0, 0, 0])
    camera_info_msg.P = [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0]
    return camera_info_msg


def convert_instance_to_semantic_segmentation(segmentation, instance_id_to_class_id):
    ids, inv = np.unique(segmentation, return_inverse=True)
    return np.array(
        [instance_id_to_class_id[x] if x in instance_id_to_class_id else 0 for x in ids]
    )[inv].reshape(segmentation.shape)


@dataclass
class Waypoint:
    position: np.ndarray = np.zeros((3,))
    yaw: float = 0.0
    is_goal: bool = False


class HabitatSimNode:
    def __init__(self, node_name: str):
        # Initialize node
        self.node_name = node_name
        rospy.init_node(self.node_name)

        self.scene_file_path = rospy.get_param("~scene_file", None)
        self.sensor_height = rospy.get_param("~sensor_height", None)
        self.image_width = rospy.get_param("~image_width", 320)
        self.image_height = rospy.get_param("~image_height", 240)
        self.sensor_rate = rospy.get_param("~sensor_rate", 30)
        self.sim_rate = rospy.get_param("~sim_rate", 60)
        self.control_rate = rospy.get_param("~control_rate", 40)
        self.enable_physics = rospy.get_param("~enable_physics", False)
        self.use_embodied_agent = rospy.get_param("~use_embodied_agent", False)
        self.wait = rospy.get_param("~wait", True)

        self.waypoints = []

        self.async_sim = AsyncSimulator(
            scene_file_path=self.scene_file_path,
            image_width=self.image_width,
            image_height=self.image_height,
            sensor_height=self.sensor_height,
            sim_rate=self.sim_rate,
            enable_physics=self.enable_physics,
            use_embodied_agent=self.use_embodied_agent,
        )

        # Instantiate and configure position controller to track pose commands
        # TODO: read pid controller params from ros
        self.position_controller = PIDPositionController(
            PIDPositionControllerParameters()
        )
        self.position_controller_thread = Thread(target=self._run_position_controller)
        self.position_controller_lock = Lock()

        # TODO: make these configurable or read from rosparam server
        self.global_frame_name = "world"
        self.agent_frame_name = "base_link"
        self.sensor_frame_name = "depth_cam"

        self.cv_bridge = CvBridge()

        # Configure publishers to sensor topics
        self.rgb_pub = rospy.Publisher("~rgb", Image, queue_size=100)
        self.depth_pub = rospy.Publisher("~depth", Image, queue_size=100)
        self.instance_pub = rospy.Publisher("~instance", Image, queue_size=100)
        self.semantic_pub = rospy.Publisher("~semantic", Image, queue_size=100)
        # fmt: off
        self.rgb_3rd_person_pub = rospy.Publisher("~rgb_3rd_person", Image, queue_size=100)
        # fmt: on
        self.pose_pub = rospy.Publisher("~pose", PoseStamped, queue_size=100)
        self.odom_pub = rospy.Publisher("~odom", Odometry, queue_size=100)
        self.tf_broadcaster = tf.TransformBroadcaster()

        # Configure static transform broadcaster for static transforms
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster()

        self.sensor_to_agent_transform = TransformStamped(
            header=Header(stamp=rospy.Time.now(), frame_id=self.agent_frame_name),
            child_frame_id=self.sensor_frame_name,
            transform=Transform(
                translation=numpy_to_vector3(
                    vec_habitat_to_ros(np.array([0, self.sensor_height, 0]))
                ),
                # The ros convention for camera frame orientation is z forward, y down, x right
                rotation=Quaternion(0.5, -0.5, 0.5, -0.5),
            ),
        )
        self.static_tf_broadcaster.sendTransform([self.sensor_to_agent_transform])

        # Configure subscriber for command topics
        self.cmd_vel_sub = rospy.Subscriber(
            "~cmd_vel",
            Twist,
            self.cmd_vel_callback,
            queue_size=10,
        )

        self.cmd_pose_sub = rospy.Subscriber(
            "~cmd_pose",
            Pose,
            self.cmd_pose_callback,
            queue_size=10,
        )

        self.cmd_trajectory_sub = rospy.Subscriber(
            "~cmd_trajectory",
            MultiDOFJointTrajectory,
            self.cmd_trajectory_callback,
            queue_size=10,
        )

    def cmd_vel_callback(self, cmd_vel_msg: Twist):
        # Convert from ros to habitat coordinate convention
        linear_vel = vec_ros_to_habitat(vector3_to_numpy(cmd_vel_msg.linear))
        angular_vel = vec_ros_to_habitat(vector3_to_numpy(cmd_vel_msg.angular))

        # Set velocity control command
        duration = 1 / self.control_rate
        self.async_sim.set_vel_control(linear_vel, angular_vel, duration)

    def cmd_pose_callback(self, pose_msg: Pose):
        target_position = vector3_to_numpy(pose_msg.position)
        target_orientation = quaternion_to_numpy(pose_msg.orientation)
        target_yaw = tf.transformations.euler_from_quaternion(target_orientation)[2]

        with self.position_controller_lock:
            self.position_controller.set_target(target_position, target_yaw)

    def cmd_trajectory_callback(self, trajectory_msg: MultiDOFJointTrajectory):
        # Add waypoints to internal waypoints queue
        for point in trajectory_msg.points:
            for transform in point.transforms:
                position = vector3_to_numpy(transform.translation)
                orientation = quaternion_to_numpy(transform.rotation)
                yaw = tf.transformations.euler_from_quaternion(orientation)[2]
                waypoint = Waypoint(position, yaw, False)
                self.waypoints.put(waypoint)

    def _publish_odometry_and_sensor_observations(self):
        (
            position,
            orientation,
            observations,
        ) = self.async_sim.get_agent_pose_and_sensor_observations()

        # Convert position and orientation to the ROS coordinate convention
        position = vec_habitat_to_ros(position)
        orientation = quat_habitat_to_ros(orientation)

        # Wrap observations into messages

        # Use timer timestamp
        timestamp = rospy.Time.now()

        # Common header - so that all the observations have the same timestamp
        sensor_msgs_header = Header(
            stamp=timestamp,
            frame_id=self.sensor_frame_name,
        )

        rgb = cv2.cvtColor(observations["color_sensor"], cv2.COLOR_RGBA2BGR)
        rgb_msg = self.cv_bridge.cv2_to_imgmsg(rgb.astype(np.uint8), encoding="bgr8")
        rgb_msg.header = sensor_msgs_header

        rgb_3rd_person = cv2.cvtColor(
            observations["color_sensor_3rd_person"], cv2.COLOR_RGBA2BGR
        )
        rgb_3rd_person_msg = self.cv_bridge.cv2_to_imgmsg(
            rgb_3rd_person.astype(np.uint8), encoding="bgr8"
        )
        rgb_3rd_person_msg.header = sensor_msgs_header

        depth = observations["depth_sensor"]
        depth_msg = self.cv_bridge.cv2_to_imgmsg(
            depth.astype(np.float32),
            encoding="passthrough",
        )
        depth_msg.header = sensor_msgs_header

        instance = observations["semantic_sensor"]
        instance_msg = self.cv_bridge.cv2_to_imgmsg(instance.astype(np.uint16))
        instance_msg.header = sensor_msgs_header

        semantic = convert_instance_to_semantic_segmentation(
            instance,
            self.async_sim.instance_id_to_class_id,  # Hack - TODO: implement getter
        )
        semantic_msg = self.cv_bridge.cv2_to_imgmsg(semantic.astype(np.uint16))
        semantic_msg.header = sensor_msgs_header

        pose_msg = PoseStamped()
        pose_msg.header = Header(stamp=timestamp, frame_id=self.agent_frame_name)
        pose_msg.pose.position = Point(*position)
        pose_msg.pose.orientation = Quaternion(*orientation)

        odom_msg = Odometry()
        odom_msg.header = Header(stamp=timestamp, frame_id=self.global_frame_name)
        odom_msg.child_frame_id = self.agent_frame_name
        odom_msg.pose.pose.position = Point(*position)
        odom_msg.pose.pose.orientation = Quaternion(*orientation)

        # Publish observations
        self.rgb_pub.publish(rgb_msg)
        self.depth_pub.publish(depth_msg)
        self.rgb_3rd_person_pub.publish(rgb_3rd_person_msg)
        self.instance_pub.publish(instance_msg)
        self.semantic_pub.publish(semantic_msg)
        self.pose_pub.publish(pose_msg)
        self.odom_pub.publish(odom_msg)
        self.tf_broadcaster.sendTransform(
            translation=position,
            rotation=orientation,
            time=timestamp,
            child=self.agent_frame_name,
            parent=self.global_frame_name,
        )

    def _run_position_controller(self):
        _r = rospy.Rate(self.control_rate)
        while not rospy.is_shutdown():
            # Get current agent pose
            with self.position_controller_lock:
                if not self.position_controller.is_goal_reached():
                    position, orientation = self.async_sim.get_agent_pose()
                    position_ros = vec_habitat_to_ros(position)
                    orientation_ros = quat_habitat_to_ros(orientation)
                    # fmt: off
                    yaw_ros = tf.transformations.euler_from_quaternion(orientation_ros)[2]
                    # fmt: on

                    (
                        linear_vel_ros,
                        yaw_rate_ros,
                    ) = self.position_controller.compute_control_cmd(
                        position_ros, yaw_ros
                    )
                    linear_vel = vec_ros_to_habitat(linear_vel_ros)
                    angular_vel = vec_ros_to_habitat(np.array([0, 0, yaw_rate_ros]))

                    duration = 1 / self.control_rate
                    self.async_sim.set_vel_control(linear_vel, angular_vel, duration)
                elif not self.position_controller.is_target_set():
                    if not self.waypoints.empty():
                        next_waypoint = self.waypoints.get()
                        self.position_controller.set_target(
                            next_waypoint.position, next_waypoint.yaw
                        )

            _r.sleep()

    def _start_simulator_and_controller_threads(self):
        # Start simulator thread
        self.async_sim.start()

        # Start position controller thread
        self.position_controller_thread.start()

    def run(self):
        r"""
        Start loop in which at every iteration sensor observations
        are published. Blocking.
        """
        try:
            self._start_simulator_and_controller_threads()
            if self.wait:
                rospy.sleep(10.0)
            # Start the simulator in a separate thread
            _r = rospy.Rate(self.sensor_rate)
            while not rospy.is_shutdown():
                self._publish_odometry_and_sensor_observations()
                _r.sleep()
        except Exception as e:
            print(str(e))


if __name__ == "__main__":
    habitat_sim_node = HabitatSimNode("habitat_sim_node")
    habitat_sim_node.run()
