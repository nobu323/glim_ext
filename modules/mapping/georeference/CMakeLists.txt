cmake_minimum_required(VERSION 3.5.2)
project(georeference)

set(CMAKE_CXX_STANDARD 17)

find_package(glim REQUIRED)
find_package(GTSAM REQUIRED)
find_package(spdlog REQUIRED)

if($ENV{ROS_VERSION} EQUAL 2)
  # ROS2
  find_package(ament_cmake_auto REQUIRED)
  ament_auto_find_build_dependencies() # Will find geometry_msgs, nav_msgs, sensor_msgs if in package.xml

  ament_auto_add_library(georeference SHARED
    src/glim_ext/geodetic.cpp
    src/glim_ext/georeference_module_ros.cpp
  )
elseif($ENV{ROS_VERSION} EQUAL 1)
  # ROS1
  find_package(catkin REQUIRED COMPONENTS glim geometry_msgs nav_msgs sensor_msgs)
  catkin_package(
    INCLUDE_DIRS include thirdparty/gtsam_points/include # Assuming gtsam_points is still relevant or a placeholder
    LIBRARIES ${glim_LIBRARIES} ${GTSAM_LIBRARIES}
  )

  add_library(georeference SHARED
    src/glim_ext/geodetic.cpp
    src/glim_ext/georeference_module_ros.cpp
  )
endif()

target_include_directories(georeference PRIVATE
  include
  ${GTSAM_INCLUDE_DIRS}
  ${glim_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS} # This will be empty for ROS2, which is fine
)
target_link_libraries(georeference
  glim_ext
  ${GTSAM_LIBRARIES}
  ${glim_LIBRARIES}
  spdlog::spdlog
  ${catkin_LIBRARIES} # This will be empty for ROS2, which is fine
)

# Install rules for ROS2
if(ament_cmake_auto_FOUND)
  ament_auto_package()
endif()
