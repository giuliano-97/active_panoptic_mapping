cmake_minimum_required(VERSION 3.0.2)
project(active_panoptic_mapping_core)

set(CMAKE_CXX_STANDARD 17)
find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)
catkin_package()
catkin_python_setup()

cs_add_library(${PROJECT_NAME}
  src/mapper/single_tsdf_panoptic_id_tracker.cpp
  src/mapper/single_tsdf_ground_truth_id_tracker.cpp
  src/planner/panoptic_map.cpp
  src/planner/trajectory_evaluator/voxel_class_evaluator.cpp
  src/planner/trajectory_evaluator/panoptic_uncertainty_evaluator.cpp
  src/planner/trajectory_evaluator/uncertainty_weighted_tsdf_entropy_evaluator.cpp
  src/planner/trajectory_evaluator/uncertaint_weighted_voxel_weight_evaluator.cpp
)


cs_install()
cs_export()
