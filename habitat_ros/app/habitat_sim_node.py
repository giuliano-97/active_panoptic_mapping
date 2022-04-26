#!/usr/bin/env python3


import rospy
import tf
from geometry_msgs.msg import Twist, Pose
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint

from habitat_ros.async_simulator import AsyncSimulator
from habitat_ros.pid_position_controller import (
    PIDPositionController,
    PIDPositionControllerParameters,
)
from habitat_ros.utils.conversions import (
    vector3_to_numpy,
    vec_ros_to_habitat,
    quaternion_to_numpy,
    quat_ros_to_habitat,
)


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

        # Instantiate and configure position controller to track pose
        # and trajectory commands
        # TODO: read pid controller params from ros
        self.position_controller = PIDPositionController(
            PIDPositionControllerParameters()
        )

        # Configure subscriber for command topics
        self.cmd_vel_sub = rospy.Subscriber(
            "cmd_vel",
            Twist,
            self.cmd_vel_callback,
            queue_size=10,
        )

        self.cmd_pose_sub = rospy.Subscriber(
            "cmd_pose",
            Pose,
            self.cmd_pose_callback,
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
        target_position = vec_ros_to_habitat(vector3_to_numpy(pose_msg.position))

        target_orientation = quat_ros_to_habitat(
            quaternion_to_numpy(pose_msg.orientation)
        )
        target_yaw = tf.transformations.euler_from_quaternion(target_orientation)[2]

        # FIXME: not thread safe?
        self.position_controller.set_target(target_position, target_yaw)

    def simulate(self):
        try:
            if self.wait:
                rospy.sleep(10.0)
            # Start the simulator in a separate thread
            self.async_sim.start()
            _r = rospy.Rate(self.sensor_rate)
            while not rospy.is_shutdown():
                self.async_sim.publish_sensor_observations_and_odometry()
                _r.sleep()
        except Exception as e:
            print(str(e))


if __name__ == "__main__":
    habitat_sim_node = HabitatSimNode("habitat_sim_node")
    habitat_sim_node.simulate()
