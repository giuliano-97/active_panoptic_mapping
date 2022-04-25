cmake_minimum_required(VERSION 3.0.2)
project(habitat_ros)

find_package(catkin REQUIRED COMPONENTS rospy nav_msgs)

# Install global modules and scripts
catkin_python_setup()

# Catkin specific configuration
catkin_package(CATKIN_DEPENDS rospy)

# do not wildcard install files since the root folder of the package will contain a debian folder for releasing
catkin_install_python(
  PROGRAMS
    app/habitat_sim_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
