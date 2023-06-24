cmake_minimum_required(VERSION 3.0.2)
project(rescue_vision)

find_package(catkin REQUIRED COMPONENTS qt_build
roscpp
rospy
cv_bridge
message_runtime
message_generation
image_transport
std_msgs
roslib
sensor_msgs
image_transport
)
find_package(OpenCV 4.5.2 REQUIRED)

catkin_package(
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ library
add_executable(findc src/findc.cpp)
add_executable(hazmat src/hazmat.cpp)
add_executable(qr_detect src/qr_detect.cpp)
add_executable(thermal src/thermal.cpp)
add_executable(victimboard src/victimboard.cpp src/robit_master_vision.cpp)

target_link_libraries(findc 
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS} 
)

target_link_libraries(hazmat
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS} 
  ${OpenCV_DNN_LIBS}
)

target_link_libraries(qr_detect
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS} 
  ${OpenCV_DNN_LIBS}
)

target_link_libraries(thermal
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS} 
  ${OpenCV_DNN_LIBS}
)

target_link_libraries(victimboard
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS} 
  ${OpenCV_DNN_LIBS}
)
