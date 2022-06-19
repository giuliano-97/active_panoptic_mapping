from pathlib import Path

import rospy
import rospkg
import roslaunch


class MappingExperimentManager:
    def __init__(self):
        rospy.init_node("mapping_experiment_manager")

        self.dataset = rospy.get_param("~dataset")
        if self.dataset != "scannet":
            raise NotImplementedError("Only dataset ScanNet supported!")

        self.data_dir_path = Path(rospy.get_param("~data_dir"))
        if not self.data_dir_path.is_dir():
            raise FileNotFoundError(f"{self.data_dir_path} is not a valid directory!")

        self.out_dir_path = Path(rospy.get_param("~out_dir"))

        scans = rospy.get_param("~scans", None)
        if scans is not None:
            self.scan_dirs = []
            for scan in scans:
                scan_dir_path = self.data_dir_path / scan
                if (
                    scan_dir_path.is_dir()
                    and scan_dir_path.joinpath("color").is_dir()
                    and scan_dir_path.joinpath("depth").is_dir()
                    and scan_dir_path.joinpath("pose").is_dir()
                    and scan_dir_path.joinpath("panoptic_pred").is_dir()
                ):
                    self.scan_dirs.append(scan_dir_path)

        else:
            self.scan_dirs = [
                p
                for p in self.data_dir_path.iterdir()
                if p.is_dir()
                and p.joinpath("color").is_dir()
                and p.joinpath("depth").is_dir()
                and p.joinpath("pose").is_dir()
                and p.joinpath("panoptic_pred").is_dir()
            ]
        if len(self.scan_dirs) < 1:
            rospy.logfatal(f"No scan directories were found in {self.data_dir_path}")

        self.mapper_configs = rospy.get_param("~mapper_configs", [])

        self.package_path = Path(
            rospkg.RosPack().get_path("panoptic_mapping_evaluation")
        )

        # TODO: make this configurable?
        self.mapper_configs_dir_path = self.package_path / "config" / "mapper"

        self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(self.uuid)

    def _run_experiment(self, mapper_config_file_path, scan_dir_path, out_dir_path):
        rospy.loginfo(
            "\n**************************************************************"
            f"\nStarting experiment:"
            f"\n\tScan: {scan_dir_path.name}. "
            f"\n\tMapper config: {mapper_config_file_path.name}"
            "\n**************************************************************"
        )
        launch_file_path = (
            self.package_path / "launch" / "run_experiment_scannet.launch"
        )

        cli_args = [
            str(launch_file_path),
            f"config:={str(mapper_config_file_path)}",
            f"data_dir:={str(scan_dir_path.parent)}",
            f"scan_id:={str(scan_dir_path.name)}",
            f"out_dir:={str(out_dir_path)}",
        ]

        roslaunch_args = cli_args[1:]
        roslaunch_file = [
            (roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)
        ]
        parent = roslaunch.parent.ROSLaunchParent(self.uuid, roslaunch_file)

        parent.start()

        parent.spin()

        rospy.loginfo("Experiment completed successfully.")

    def run_experiments(self):
        for scan_dir_path in self.scan_dirs:
            for mapper_config in self.mapper_configs:
                experiment_out_dir_path = (
                    self.out_dir_path / scan_dir_path.name / mapper_config
                )
                experiment_out_dir_path.mkdir(exist_ok=True, parents=True)
                mapper_config_file_path = self.mapper_configs_dir_path.joinpath(
                    mapper_config
                ).with_suffix(".yaml")
                self._run_experiment(
                    mapper_config_file_path,
                    scan_dir_path,
                    experiment_out_dir_path,
                )


if __name__ == "__main__":
    em = MappingExperimentManager()
    em.run_experiments()
