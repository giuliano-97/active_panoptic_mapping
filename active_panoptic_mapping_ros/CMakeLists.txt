cmake_minimum_required(VERSION 3.0.2)
project(active_panoptic_mapping_ros)

set(CMAKE_CXX_STANDARD 17)
find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)
catkin_package()

cs_add_executable(active_panoptic_mapping_node
        app/active_panoptic_mapping_node.cpp
)

cs_install()
cs_export()

catkin_install_python(
  PROGRAMS
    app/panoptic_segmentation_node.py
  DESTINATION 
    ${CATKIN_PACKAGE_BIN_DESTINATION}
)
