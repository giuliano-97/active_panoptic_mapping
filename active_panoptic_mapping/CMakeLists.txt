cmake_minimum_required(VERSION 3.0.2)
project(active_panoptic_mapping)

set(CMAKE_CXX_STANDARD 17)
find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)
catkin_package()

cs_add_library(${PROJECT_NAME}
  src/planner/panoptic_map.cpp
  src/planner/trajectory_evaluator/voxel_class_evaluator.cpp
  src/planner/trajectory_evaluator/panoptic_uncertainty_evaluator.cpp
  src/planner/trajectory_evaluator/uncertainty_weighted_tsdf_entropy_evaluator.cpp
  src/planner/trajectory_evaluator/uncertaint_weighted_voxel_weight_evaluator.cpp
)

cs_add_executable(active_panoptic_mapping_node
        app/active_panoptic_mapping_node.cpp
        )
target_link_libraries(active_panoptic_mapping_node ${PROJECT_NAME})

cs_install()
cs_export()
