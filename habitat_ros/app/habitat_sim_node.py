#!/usr/bin/env python3


import rospy
from geometry_msgs.msg import Twist

from habitat_ros.async_simulator import AsyncSimulator
from habitat_ros.utils.conversions import (
    vector3_to_numpy,
    vec_ros_to_habitat,
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
        self.wait = rospy.get_param("~wait", True)

        self.async_sim = AsyncSimulator(
            scene_file_path=self.scene_file_path,
            image_width=self.image_width,
            image_height=self.image_height,
            sensor_height=self.sensor_height,
            sim_rate=self.sim_rate,
            enable_physics=self.enable_physics,
        )

        # Configure subscriber for command topics
        self.cmd_vel_sub = rospy.Subscriber(
            "cmd_vel",
            Twist,
            self.cmd_vel_callback,
            queue_size=100,
        )

    def cmd_vel_callback(self, cmd_vel_msg: Twist):
        # Convert from ros to habitat coordinate convention
        linear_vel = vec_ros_to_habitat(vector3_to_numpy(cmd_vel_msg.linear))
        angular_vel = vec_ros_to_habitat(vector3_to_numpy(cmd_vel_msg.angular))

        # Set velocity control command
        self.async_sim.set_vel_control(linear_vel, angular_vel)

    def simulate(self):
        try:
            # Start the simulator in a separate thread
            self.async_sim.start()
            if self.wait:
                rospy.loginfo("Waiting for pano seg predictor node to be ready.")
                rospy.sleep(10.0)
            _r = rospy.Rate(self.sensor_rate)
            while not rospy.is_shutdown():
                self.async_sim.publish_sensor_observations_and_odometry()
                _r.sleep()
        except Exception as e:
            print(str(e))


if __name__ == "__main__":
    habitat_sim_node = HabitatSimNode("habitat_sim_node")
    habitat_sim_node.simulate()
