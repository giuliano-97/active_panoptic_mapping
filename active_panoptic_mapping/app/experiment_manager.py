import os
from pathlib import Path

import rospy
import rospkg
import roslaunch
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool, Empty
from tqdm import tqdm
from panoptic_mapping_msgs.srv import SaveLoadMap


class PlanningExperimentManager:
    def __init__(self):
        rospy.init_node("active_panoptic_mapping_experiment_manager")

        self.timeout = rospy.get_param("timeout", 30)
        self.experiment_name = rospy.get_param("~experiment_name")
        self.n_reps = rospy.get_param("~n_reps", 10)
        self.duration = int(rospy.get_param("~duration", 1200))  # 20 minutes by default
        self.save_map_every_n_sec = int(rospy.get_param("~save_map_every_n_sec", 60))
        self.planner_config = rospy.get_param("~planner_config")
        self.mapper_config = rospy.get_param("~mapper_config")
        self.out_dir_path = Path(rospy.get_param("~out_dir"))

        self.out_dir_path.mkdir(parents=True, exist_ok=True)
        self.map_file_name_template = "{:06d}.panmap"
        self.n_checkpoints = self.duration // self.save_map_every_n_sec

        self.package_path = Path(rospkg.RosPack().get_path("active_panoptic_mapping"))

        self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(self.uuid)

    def run(self):
        for i in range(self.n_reps):
            if not rospy.is_shutdown():
                self._run_once(i)
            else:
                rospy.loginfo("Shutdown detected. Early exit")
                return

    def _run_once(self, run_id: int):
        rospy.loginfo(f"Starting run {run_id} of experiment {self.experiment_name}.")

        # Prepare output directory
        run_out_dir = self.out_dir_path / f"run_{run_id}"
        run_out_dir.mkdir(exist_ok=True, parents=True)
        saved_maps_dir_path = run_out_dir / "maps"
        saved_maps_dir_path.mkdir(exist_ok=True)
        logs_dir_path = run_out_dir / "logs"
        logs_dir_path.mkdir(exist_ok=True)
        bags_dir_path = run_out_dir / "bags"
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
            f"out_dir:={str(run_out_dir)}",
        ]

        roslaunch_args = cli_args[1:]
        roslaunch_file = [
            (roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)
        ]
        parent = roslaunch.parent.ROSLaunchParent(self.uuid, roslaunch_file)

        parent.start()

        try:
            # Wait for mapper and planner
            save_map_srv.wait_for_service(timeout=self.timeout)
            finish_mapping_srv.wait_for_service(timeout=self.timeout)
            run_planner_srv.wait_for_service(timeout=self.timeout)

            # Hack: wait until mapper starts processing so when the planning starts
            # we are sure that the map is ready
            rospy.loginfo("Waiting for mapper to start processing.")
            rospy.wait_for_message(
                "/active_panoptic_mapping_node/visualization/tracking/color",
                Image,
                timeout=self.timeout,
            )

            rospy.loginfo("Starting planner.")
            run_planner_srv(True)

            for i in range(self.n_checkpoints):
                rospy.sleep(self.save_map_every_n_sec)
                map_file_path = saved_maps_dir_path / (
                    self.map_file_name_template.format(
                        (i + 1) * self.save_map_every_n_sec
                    )
                )
                save_map_srv(str(map_file_path))

            rospy.sleep(self.duration - self.n_checkpoints * self.save_map_every_n_sec)

            # Stop the planner
            run_planner_srv(False)

            rospy.loginfo(
                "Waiting for mapper to finish processing the remaining frames in the queue."
            )
            for i in tqdm(range(5)):
                rospy.sleep(1)

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
        except rospy.ROSException as e:
            rospy.logwarn(
                f"An ROS exception was raise: {str(e)}. Run {run_id} aborted."
            )
        except Exception as e:
            rospy.logwarn(
                f"An unexpected error occurred: {str(e)}. Run {run_id} aborted"
            )

        # End experiment
        parent.shutdown()


if __name__ == "__main__":
    em = PlanningExperimentManager()
    em.run()
