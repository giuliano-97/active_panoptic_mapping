from threading import Thread, Lock
from typing import Optional, Tuple

import habitat_sim
import magnum
import numpy as np
import quaternion
import rospy

from habitat_ros.utils import REPLICA_CLASS_ID_TO_NYU40


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
        initial_position: np.ndarray = np.zeros((3,)),
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
            initial_state=habitat_sim.AgentState(position=initial_position),
        )

        # Get the physics object attributes manager
        if use_embodied_agent:
            # TODO: refactor and pass this as a config param
            robot_asset_dir = rospy.get_param("~robot_asset_dir", None)
            # If the asset dir is not none, load the robot asset
            if robot_asset_dir is not None:
                obj_templates_mgr = self.sim.get_object_template_manager()

                # Instantiate object
                # fmt: off
                locobot_template_id = obj_templates_mgr.load_configs(robot_asset_dir)[0]
                # fmt: on

                # Instantiate and attach object to agent
                self.agent_body = self.sim.add_object(
                    locobot_template_id, self.agent.scene_node
                )
            # Otherwise approximate robot as a sphere
            else:
                # Get primitive attribute manager
                prim_attr_mgr = self.sim.get_asset_template_manager()

                # Get solid icosphere template handle
                # fmt: off
                icosphere_template_handle = prim_attr_mgr.get_template_handles("icosphereSolid")[0]
                # fmt: on

                # Instantiate and attach object to agent
                rigid_object_manager = self.sim.get_rigid_object_manager()
                self.agent_body = rigid_object_manager.add_object_by_template_handle(
                    icosphere_template_handle,  # obj template handle
                    self.agent.scene_node,  # attachment node
                )

        # Create velocity control object
        self.vel_control = habitat_sim.physics.VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.ang_vel_is_local = True
        self.vel_control_requested_duration = None
        self.vel_control_duration = 0.0

        # Cache instance to class ids mapping
        self.instance_id_to_class_id = get_instance_id_to_category_id_map(
            self.sim.semantic_scene, self.is_replica
        )

        # Simulator lock to avoid reading sensor data or updating
        # the velocity command while the simulation is being stepped
        self.sim_lock = Lock()

    def _reset_vel_control(self):
        self.vel_control.linear_velocity = magnum.Vector3.zero_init()
        self.vel_control.angular_velocity = magnum.Vector3.zero_init()
        self.vel_control_requested_duration = None
        self.vel_control_duration = 0.0

    def _get_agent_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Get current agent position and orientation as numpy arrays.
        Not thread safe.
        """
        agent_position = self.agent.state.position
        agent_orientation = np.roll(
            quaternion.as_float_array(self.agent.state.rotation), -1
        )
        return agent_position, agent_orientation

    def get_agent_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Thread-safe public interface to get the current agent pose.
        """
        with self.sim_lock:
            return self._get_agent_pose()

    def get_agent_pose_and_sensor_observations(self):
        with self.sim_lock:
            observations = self.sim.get_sensor_observations()
            agent_position, agent_orientation = self._get_agent_pose()

        return agent_position, agent_orientation, observations

    def set_vel_control(
        self,
        linear: np.ndarray,
        angular: np.ndarray,
        duration: Optional[float] = None,
    ):
        with self.sim_lock:

            self.vel_control.linear_velocity = magnum.Vector3(linear)

            self.vel_control.angular_velocity = magnum.Vector3(angular)

            self.vel_control_requested_duration = duration
            self.vel_control_duration = 0.0

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

            if self.vel_control_requested_duration is not None:
                self.vel_control_duration += self.time_step
                if self.vel_control_duration >= self.vel_control_requested_duration:
                    self._reset_vel_control()

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
