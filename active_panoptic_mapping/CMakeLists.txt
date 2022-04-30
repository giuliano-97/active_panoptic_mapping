cmake_minimum_required(VERSION 3.0.2)
project(active_panoptic_mapping)

set(CMAKE_CXX_STANDARD 17)
find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)
catkin_package()

cs_add_library(${PROJECT_NAME}
  src/planner/panoptic_map.cpp
)

cs_add_executable(active_panoptic_mapping_node
        app/active_panoptic_mapping_node.cpp
        )
target_link_libraries(active_panoptic_mapping_node ${PROJECT_NAME})

cs_install()
cs_export()

#############
## Python ##
#############

catkin_python_setup()

catkin_install_python(PROGRAMS
  app/pano_seg_predictor.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


