cmake_minimum_required(VERSION 3.0.2)
project(active_panoptic_mapping_utils)

set(CMAKE_CXX_STANDARD 17)
find_package(catkin_simple REQUIRED)

# This must be run before catkin_simple or message generation will fail
catkin_python_setup()

catkin_simple(ALL_DEPS_REQUIRED)

catkin_package()

cs_add_library(${PROJECT_NAME} src/evaluation_data_exporter.cpp)

cs_add_executable(evaluation_data_export_node
                  app/evaluation_data_export_node.cpp)
target_link_libraries(evaluation_data_export_node ${PROJECT_NAME})

catkin_install_python(
  PROGRAMS 
    app/evaluation_manager.py 
    app/evaluation_manager_2.py 
    app/mapping_experiment_manager.py
    app/planning_experiment_manager.py
    app/scannet_data_player.py
  DESTINATION
    ${CATKIN_PACKAGE_BIN_DESTINATION})


cs_install()
cs_export()
