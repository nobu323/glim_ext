cmake_minimum_required(VERSION 3.5.2)
project(georeference)

set(CMAKE_CXX_STANDARD 17)

# Core dependencies for all builds
find_package(glim REQUIRED)
find_package(GTSAM REQUIRED)
find_package(spdlog REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)

# Print Eigen include path for debugging
message(STATUS "Eigen3 include dir: ${EIGEN3_INCLUDE_DIR}")

# Source files (common for all build types)
set(SOURCES
  src/glim_ext/geodetic.cpp
  src/glim_ext/georeference_module_ros.cpp
)

# ROS version-specific configuration
if($ENV{ROS_VERSION} EQUAL 2)
  # ROS2 build
  
  # Define GLIM_ROS2 for conditional compilation
  add_definitions(-DGLIM_ROS2)
  
  find_package(ament_cmake_auto REQUIRED)
  ament_auto_find_build_dependencies()
  
  ament_auto_add_library(${PROJECT_NAME} SHARED ${SOURCES})
  
  target_include_directories(${PROJECT_NAME} PRIVATE
    include
    ${EIGEN3_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${GTSAM_INCLUDE_DIR}
    ${glim_INCLUDE_DIRS}
  )

  target_link_libraries(${PROJECT_NAME}
    glim_ext
    ${glim_LIBRARIES}
    ${GTSAM_LIBRARIES}
    ${Boost_LIBRARIES}
    spdlog::spdlog
  )


elseif($ENV{ROS_VERSION} EQUAL 1)
  # ROS1 build
  find_package(catkin REQUIRED COMPONENTS glim roscpp geometry_msgs nav_msgs sensor_msgs)
  
  catkin_package(
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
    CATKIN_DEPENDS glim roscpp geometry_msgs nav_msgs sensor_msgs
  )
  
  target_include_directories(${PROJECT_NAME} PRIVATE
    include
    ${EIGEN3_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${GTSAM_INCLUDE_DIR}
    ${glim_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
  )  
  add_library(${PROJECT_NAME} SHARED ${SOURCES})
  
  target_link_libraries(${PROJECT_NAME}
    glim_ext
    ${glim_LIBRARIES}
    ${GTSAM_LIBRARIES}
    ${Boost_LIBRARIES}
    spdlog::spdlog
    ${catkin_LIBRARIES}
  )
endif()
