from pathlib import Path

import rospy
import rospkg
import roslaunch
from std_srvs.srv import SetBool, Empty
from panoptic_mapping_msgs.srv import SaveLoadMap


class PlanningExperimentManager:
    def __init__(self):
        self.service_timeout = 15
        rospy.init_node("active_panoptic_mapping_experiment_manager")

        self.experiment_name = rospy.get_param("~experiment_name")
        self.n_reps = int(rospy.get_param("~n_reps", 10))
        self.duration = int(rospy.get_param("~duration", 1200))  # 20 minutes by default
        self.save_map_every_n_sec = int(rospy.get_param("~save_map_every_n_sec", 60))
        self.planner_config = rospy.get_param("~planner_config")
        self.mapper_config = rospy.get_param("~mapper_config")
        self.out_dir_path = Path(rospy.get_param("~out_dir"))

        self.out_dir_path.mkdir(parents=True, exist_ok=True)
        self.map_file_name_template = "{:06d}.panmap"

        self.package_path = Path(rospkg.RosPack().get_path("active_panoptic_mapping"))
        self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(self.uuid)

    def run(self):
        for n in range(self.n_reps):
            self._run_once(n)

    def _run_once(self, rep_number=0):
        # Prepare output directory
        run_out_dir_path = self.out_dir_path / f"run_{str(rep_number)}"
        run_out_dir_path.mkdir(exist_ok=True)
        saved_maps_dir_path = run_out_dir_path / "maps"
        saved_maps_dir_path.mkdir(exist_ok=True)
        logs_dir_path = run_out_dir_path / "logs"
        logs_dir_path.mkdir(exist_ok=True)
        bags_dir_path = run_out_dir_path / "bags"
        bags_dir_path.mkdir(exist_ok=True)

        launch_file_path = self.package_path / "launch" / "run.launch"

        run_planner_srv = rospy.ServiceProxy(
            "/active_panoptic_mapping_node/toggle_running", SetBool
        )

        save_map_srv = rospy.ServiceProxy(
            "/active_panoptic_mapping_node/save_map", SaveLoadMap
        )

        finish_mapping_srv = rospy.ServiceProxy(
            "/active_panoptic_mapping_node/finish_mapping", Empty
        )

        pub_sensor_data_srv = rospy.ServiceProxy(
            "/habitat_sim_node/toggle_pub_sensor_data", SetBool
        )

        cli_args = [
            str(launch_file_path),
            "visualize:=true",
            "has_screen:=false",
            "record:=true",
            f"planner_config:={self.planner_config}",
            f"out_dir:={str(run_out_dir_path)}",
        ]

        roslaunch_args = cli_args[1:]
        roslaunch_file = [
            (roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)
        ]
        parent = roslaunch.parent.ROSLaunchParent(self.uuid, roslaunch_file)

        parent.start()

        # Sleep for 15 seconds - wait for the other components to be ready
        # TODO(albanesg): there must be a better way to do this
        # rospy.sleep(15)

        # Wait for sim to be ready
        pub_sensor_data_srv.wait_for_service(timeout=self.service_timeout)

        # Start publishing sensor data
        pub_sensor_data_srv(True)

        # Wait for mapper+planner
        run_planner_srv.wait_for_service(timeout=self.service_timeout)
        save_map_srv.wait_for_service(timeout=self.service_timeout)
        finish_mapping_srv.wait_for_service(timeout=self.service_timeout)

        # Trigger planner
        run_planner_srv(True)

        n_checkpoints = self.duration // self.save_map_every_n_sec
        for i in range(n_checkpoints):
            rospy.sleep(self.save_map_every_n_sec)
            map_file_path = saved_maps_dir_path / (
                self.map_file_name_template.format((i + 1) * self.save_map_every_n_sec)
            )
            save_map_srv(str(map_file_path))

        rospy.sleep(self.duration - n_checkpoints * self.save_map_every_n_sec)

        # Stop the planner
        run_planner_srv(False)

        # Stop publishing sensor data
        pub_sensor_data_srv(False)

        rospy.loginfo(
            "Waiting for mapper to finish processing the remaining frames in the queue."
        )
        rospy.sleep(5)

        rospy.loginfo("Saving final map.")
        map_file_path = saved_maps_dir_path / self.map_file_name_template.format(
            self.duration
        )
        save_map_srv(str(map_file_path))

        # Finish mapping
        finish_mapping_srv()

        # Close service proxies
        pub_sensor_data_srv.close()
        run_planner_srv.close()
        finish_mapping_srv.close()
        save_map_srv.close()

        # End experiment
        parent.shutdown()


if __name__ == "__main__":
    em = PlanningExperimentManager()
    try:
        em.run()
        rospy.signal_shutdown("Experiment completed successfully.")
    except Exception as err:
        rospy.logfatal(f"An unexpected error occured: {str(err)}")
        rospy.signal_shutdown("Experiment aborted.")
