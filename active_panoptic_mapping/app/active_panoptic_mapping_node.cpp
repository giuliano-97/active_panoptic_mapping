#include <chrono>
#include <memory>
#include <thread>

#include <active_3d_planning_ros/module/module_factory_ros.h>
#include <active_3d_planning_ros/planner/ros_planner.h>
#include <glog/logging.h>
#include <panoptic_mapping_ros/panoptic_mapper.h>
#include <ros/ros.h>

#include "active_panoptic_mapping/planner/panoptic_map.h"

int main(int argc, char** argv) {
  // Always add these arguments for proper logging.
  config_utilities::RequiredArguments ra(
      &argc, &argv, {"--logtostderr", "--colorlogtostderr"});

  // Setup logging.
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  google::ParseCommandLineFlags(&argc, &argv, false);

  // Start Ros.
  ros::init(argc, argv, "active_panoptic_mapping_node",
            ros::init_options::NoSigintHandler);
  ros::NodeHandle nh("");
  ros::NodeHandle nh_private("~");

  // Setup the mapper.
  auto mapper =
      std::make_shared<panoptic_mapping::PanopticMapper>(nh, nh_private);

  // Setup the planner.
  active_3d_planning::ros::ModuleFactoryROS factory;
  active_3d_planning::Module::ParamMap param_map;
  active_3d_planning::ros::RosPlanner::setupFactoryAndParams(
      &factory, &param_map, nh_private);
  active_3d_planning::ros::RosPlanner planner(nh, nh_private, &factory,
                                              &param_map);

  // Link the planner to the mapper.
  auto map =
      dynamic_cast<active_3d_planning::map::PanopticMap*>(&planner.getMap());
  if (!map) {
    LOG(FATAL) << "The active panoptic mapping node can only be run with a "
                  "panoptic map.";
    return 1;
  }
  map->setMap(mapper);

  // Spin.
  ros::AsyncSpinner spinner(mapper->getConfig().ros_spinner_threads);
  spinner.start();
  planner.planningLoop();
  ros::waitForShutdown();

  return 0;
}
