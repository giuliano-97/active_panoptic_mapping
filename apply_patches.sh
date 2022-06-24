#!/usr/bin/env bash

cd "$(dirname "$(realpath "$0")")";

patch -R ../opencv3_catkin/CMakeLists.txt disable_opencv3_catkin_openexr.patch

patch -R ../eigen_catkin/use-system-installation-of-eigen.cmake use_eigen_catkin.patch