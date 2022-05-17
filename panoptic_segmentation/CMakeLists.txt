cmake_minimum_required(VERSION 3.0.2)
project(panoptic_segmentation)

set(CMAKE_CXX_STANDARD 17)
find_package(catkin REQUIRED COMPONENTS rospy nav_msgs)

#############
## Python ##
#############
catkin_package()

catkin_python_setup()

catkin_install_python(
  PROGRAMS
    app/panoptic_segmentation_node.py
  DESTINATION 
    ${CATKIN_PACKAGE_BIN_DESTINATION}
)


