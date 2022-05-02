cmake_minimum_required(VERSION 3.0.2)
project(panoptic_mapping_evaluation)

set(CMAKE_CXX_STANDARD 17)
find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)
catkin_python_setup()

catkin_package()


cs_add_library(${PROJECT_NAME} src/evaluation_data_exporter.cpp)

cs_add_executable(evaluation_data_export_node
                  app/evaluation_data_export_node.cpp)
target_link_libraries(evaluation_data_export_node ${PROJECT_NAME})

# ##############################################################################
# Install
# ##############################################################################

cs_install()

catkin_install_python(PROGRAMS app/evaluation_manager_node.py DESTINATION
                      ${CATKIN_PACKAGE_BIN_DESTINATION})

cs_export()
