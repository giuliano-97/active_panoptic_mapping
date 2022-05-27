from pathlib import Path

import rospy
import rospkg
import roslaunch
from std_srvs.srv import SetBool

from panoptic_mapping_msgs.srv import SaveLoadMap


class PlanningExperimentManager:
    def __init__(self):
        rospy.init_node("ipp_experiment_manager")

        self.run_planner_srv = rospy.ServiceProxy(
            "/active_panoptic_mapping_node/toggle_running", SetBool
        )

        self.save_map_srv = rospy.ServiceProxy(
            "/active_panoptic_mapping_node/save_map", SaveLoadMap
        )

        self.duration = int(rospy.get_param("~duration", 1800))
        self.save_map_every_n_sec = int(rospy.get_param("~save_map_every_n_sec", 120))
        self.planner_config = rospy.get_param("~planner_config")
        self.mapper_config = rospy.get_param("~mapper_config")
        self.out_dir_path = Path(rospy.get_param("~out_dir"))

        self.out_dir_path.mkdir(parents=True, exist_ok=True)
        self.saved_maps_dir_path = self.out_dir_path / "maps"
        self.saved_maps_dir_path.mkdir(exist_ok=True)
        self.logs_dir_path = self.out_dir_path / "logs"
        self.logs_dir_path.mkdir(exist_ok=True)
        self.bags_dir_path = self.out_dir_path / "bags"
        self.bags_dir_path.mkdir(exist_ok=True)
        self.map_file_name_template = "{:06d}.panmap"

        self.package_path = Path(rospkg.RosPack().get_path("active_panoptic_mapping"))
        self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(self.uuid)

    def run(self):
        launch_file_path = self.package_path / "launch" / "run.launch"

        cli_args = [
            str(launch_file_path),
            "visualize:=true",
            "has_screen:=false",
            "record:=true",
            f"planner_config:={self.planner_config}",
            f"out_dir:={str(self.out_dir_path)}",
        ]

        roslaunch_args = cli_args[1:]
        roslaunch_file = [
            (roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)
        ]
        parent = roslaunch.parent.ROSLaunchParent(self.uuid, roslaunch_file)

        parent.start()

        # Sleep for 15 seconds - wait for the other components to be ready
        # TODO(albanesg): there must be a better way to do this
        rospy.sleep(15)

        # Trigger planner
        self.run_planner_srv(True)

        n_checkpoints = self.duration // self.save_map_every_n_sec
        for i in range(n_checkpoints):
            rospy.sleep(self.save_map_every_n_sec)
            map_file_path = self.saved_maps_dir_path / (
                self.map_file_name_template.format((i + 1) * self.save_map_every_n_sec)
            )
            self.save_map_srv(str(map_file_path))

        rospy.sleep(self.duration - n_checkpoints * self.save_map_every_n_sec)

        # Save final map
        map_file_path = self.saved_maps_dir_path / self.map_file_name_template.format(
            self.duration
        )
        self.save_map_srv(str(map_file_path))

        # End experiment
        parent.shutdown()


if __name__ == "__main__":
    em = PlanningExperimentManager()
    em.run()
    rospy.signal_shutdown("Experiment completed successfully.")
