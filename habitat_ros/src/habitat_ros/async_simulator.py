from re import S
from threading import Thread, Lock

from cv_bridge import CvBridge
import habitat_sim
import magnum
import numpy as np
import rospy
import tf
import tf2_ros
from geometry_msgs.msg import (
    Point,
    Quaternion,
    PoseStamped,
    TransformStamped,
    Transform,
)
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header

from .utils.conversions import (
    numpy_to_vector3,
    vec_habitat_to_ros,
    quat_habitat_to_ros,
    REPLICA_CLASS_ID_TO_NYU40,
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


def make_agent_cfg(image_width: int, image_height: int, sensor_height: float):
    agent_cfg = habitat_sim.AgentConfiguration()

    # Note: all sensors must have the same resolution
    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = np.array([[image_height], [image_width]])
    color_sensor_spec.position = [0.0, sensor_height, 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = np.array([[image_height], [image_width]])
    depth_sensor_spec.position = [0.0, sensor_height, 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_sensor_spec)

    color_sensor_3rd_person_spec = habitat_sim.CameraSensorSpec()
    color_sensor_3rd_person_spec.uuid = "color_sensor_3rd_person"
    color_sensor_3rd_person_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_3rd_person_spec.resolution = np.array([[image_height], [image_width]])
    color_sensor_3rd_person_spec.position = [0.0, sensor_height + 0.4, 0.3]
    color_sensor_3rd_person_spec.orientation = [-np.pi / 4, 0, 0]
    color_sensor_3rd_person_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_3rd_person_spec)

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = np.array([[image_height], [image_width]])
    semantic_sensor_spec.position = [0.0, sensor_height, 0.0]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(semantic_sensor_spec)

    agent_cfg.sensor_specifications = sensor_specs

    return agent_cfg


def make_sim_cfg(scene_file_path: str, enable_physics: bool = True):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = scene_file_path
    sim_cfg.enable_physics = enable_physics
    return sim_cfg


def get_instance_id_to_category_id_map(scene, is_replica: bool):
    instance_id_to_category_id_map = {0: 0}
    for obj in scene.objects:
        if obj is None:
            continue
        if obj.id is None:
            continue
        if obj.category is None:
            continue
        instance_id = int(obj.id.split("_")[-1])
        category_id = obj.category.index()
        if is_replica:
            category_id = REPLICA_CLASS_ID_TO_NYU40[category_id]
        instance_id_to_category_id_map[instance_id] = category_id
    return instance_id_to_category_id_map


class AsyncSimulator(Thread):
    r"""Wrapper around habitat_sim.Simulator which continuously
    steps the simulator physics on a separate thread and provides
    an interface to query sensor observations on demand.
    """

    # TODO: This constructor should be refactored
    def __init__(
        self,
        scene_file_path: str,
        image_width: int,
        image_height: int,
        sensor_height: float,
        sim_rate: float,
        enable_physics: bool = False,
        use_embodied_agent: bool = False,
        # TODO: there should be a better way to infer the dataset
        is_replica: bool = True,
    ):
        super().__init__()

        # Sensor parameters
        self.image_width = image_width
        self.image_height = image_height
        self.sensor_height = sensor_height

        # Simulation params
        self.scene_file_path = scene_file_path
        self.sim_rate = float(sim_rate)
        self.time_step = float(1.0 / self.sim_rate)
        self.enable_physics = enable_physics
        self.is_replica = is_replica

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
        self.camera_info_pub = rospy.Publisher("~camera_info", CameraInfo, queue_size=100)
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

        # Generate sim config
        self.cfg = habitat_sim.Configuration(
            make_sim_cfg(
                scene_file_path=self.scene_file_path,
                enable_physics=self.enable_physics,
            ),
            [make_agent_cfg(self.image_width, self.image_height, self.sensor_height)],
        )

        # Init simulator
        self.sim = habitat_sim.Simulator(self.cfg)

        # Init agent
        self.agent = self.sim.initialize_agent(
            agent_id=0,
            initial_state=habitat_sim.AgentState(position=np.array([0.0, 0.0, 0.0])),
        )

        # Get the physics object attributes manager
        if use_embodied_agent:
            # TODO: parameter parsing scattered everywhere is bad
            robot_config_dir = rospy.get_param("~robot_config_dir")

            obj_templates_mgr = self.sim.get_object_template_manager()

            # Instantiate object
            # FIXME: load urdf with mesh instead
            # fmt: off
            locobot_template_id = obj_templates_mgr.load_configs(robot_config_dir)[0]
            # fmt: on

            self.agent_body = self.sim.add_object(
                locobot_template_id, self.agent.scene_node
            )

        # Create velocity control object
        self.vel_control = habitat_sim.physics.VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.ang_vel_is_local = True

        # Cache instance to class ids mapping
        self.instance_id_to_class_id = get_instance_id_to_category_id_map(
            self.sim.semantic_scene, self.is_replica
        )

        # Simulator lock to avoid reading sensor data or updating
        # the velocity command while the simulation is being stepped
        self.sim_lock = Lock()

    def cv2_to_depthmsg(self, depth_img: np.ndarray) -> Image:
        r"""
        Converts a Habitat depth image to a ROS Image message.
        """
        if len(depth_img.shape) > 2:
            depth_img_in_m = np.squeeze(depth_img, axis=2)
        else:
            depth_img_in_m = depth_img
        depth_msg = self.cv_bridge.cv2_to_imgmsg(
            depth_img_in_m.astype(np.float32),
            encoding="passthrough",
        )
        return depth_msg

    def _convert_instance_to_semantic_segmentation(self, segmentation):
        ids, inv = np.unique(segmentation, return_inverse=True)
        return np.array(
            [
                self.instance_id_to_class_id[x]
                if x in self.instance_id_to_class_id
                else 0
                for x in ids
            ]
        )[inv].reshape(segmentation.shape)

    def publish_sensor_observations_and_odometry(self):
        # Acquire the lock to get the sensor info then release it
        # so the simulation thread is not waiting for too long
        with self.sim_lock:
            timestamp = rospy.Time.now()

            # Collect sensor observations
            observations = self.sim.get_sensor_observations()

            # Get agent pose info
            agent_orientation = quat_habitat_to_ros(
                np.roll(np.array(self.agent.state.rotation.components), -1),
            )
            agent_translation = vec_habitat_to_ros(self.agent.state.position)

        # Wrap observations into messages

        # Common header - so that all the observations have the same timestamp
        sensor_msgs_header = Header(
            stamp=timestamp,
            frame_id=self.sensor_frame_name,
        )

        rgb = observations["color_sensor"]
        if rgb.shape[2] > 3:
            rgb = rgb[:, :, :3]
        rgb_msg = self.cv_bridge.cv2_to_imgmsg(rgb.astype(np.uint8), encoding="bgr8")
        rgb_msg.header = sensor_msgs_header

        rgb_3rd_person = observations["color_sensor_3rd_person"]
        if rgb_3rd_person.shape[2] > 3:
            rgb_3rd_person = rgb_3rd_person[:, :, :3]
        rgb_3rd_person_msg = self.cv_bridge.cv2_to_imgmsg(
            rgb_3rd_person.astype(np.uint8), encoding="bgr8"
        )
        rgb_3rd_person_msg.header = sensor_msgs_header

        depth = observations["depth_sensor"]
        depth_msg = self.cv2_to_depthmsg(depth)
        depth_msg.header = sensor_msgs_header

        instance = observations["semantic_sensor"]
        instance_msg = self.cv_bridge.cv2_to_imgmsg(instance.astype(np.uint16))
        instance_msg.header = sensor_msgs_header

        semantic = self._convert_instance_to_semantic_segmentation(instance)
        semantic_msg = self.cv_bridge.cv2_to_imgmsg(semantic.astype(np.uint16))
        semantic_msg.header = sensor_msgs_header

        camera_info_msg = make_depth_camera_info_msg(
            sensor_msgs_header, self.image_height, self.image_width
        )

        pose_msg = PoseStamped()
        pose_msg.header = Header(stamp=timestamp, frame_id=self.agent_frame_name)
        pose_msg.pose.position = Point(*agent_translation)
        pose_msg.pose.orientation = Quaternion(*agent_orientation)

        odom_msg = Odometry()
        odom_msg.header = Header(stamp=timestamp, frame_id=self.global_frame_name)
        odom_msg.child_frame_id = self.agent_frame_name
        odom_msg.pose.pose.position = Point(*agent_translation)
        odom_msg.pose.pose.orientation = Quaternion(*agent_orientation)

        # Publish observations
        self.rgb_pub.publish(rgb_msg)
        self.depth_pub.publish(depth_msg)
        self.rgb_3rd_person_pub.publish(rgb_3rd_person_msg)
        self.camera_info_pub.publish(camera_info_msg)
        self.instance_pub.publish(instance_msg)
        self.semantic_pub.publish(semantic_msg)
        self.pose_pub.publish(pose_msg)
        self.odom_pub.publish(odom_msg)
        self.tf_broadcaster.sendTransform(
            translation=agent_translation,
            rotation=agent_orientation,
            time=timestamp,
            child=self.agent_frame_name,
            parent=self.global_frame_name,
        )

    def set_vel_control(self, linear: np.ndarray, angular: np.ndarray):
        with self.sim_lock:

            self.vel_control.linear_velocity = magnum.Vector3(linear)

            self.vel_control.angular_velocity = magnum.Vector3(angular)

    def _step(self):
        with self.sim_lock:
            cur_agent_pose = habitat_sim.RigidState(
                habitat_sim.utils.common.quat_to_magnum(self.agent.state.rotation),
                self.agent.state.position,
            )

            # Integrate current agent pose to the next using cmd_vel
            next_agent_pose = self.vel_control.integrate_transform(
                self.time_step,
                cur_agent_pose,
            )

            # FIXME: no collision checking implemented!
            # If the agent is embodied, collision checks should be done using
            # Simulator.perform_discrete_collision_check (?)

            # Set the agent state
            self.agent.set_state(
                habitat_sim.AgentState(
                    position=next_agent_pose.translation,
                    rotation=habitat_sim.utils.common.quat_from_magnum(
                        next_agent_pose.rotation
                    ),
                )
            )

            # Step physics
            self.sim.step_physics(self.time_step)

    def run(self):
        _r = rospy.Rate(self.sim_rate)
        while not rospy.is_shutdown():
            self._step()
            _r.sleep()
